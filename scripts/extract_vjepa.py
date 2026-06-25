"""
Phase 1.2 — Extract V-JEPA 2.1 per-frame features from SoccerNet 224p videos.

For each game/half we slide a short clip window over the match and run the
V-JEPA 2.1 ViT-L encoder once per SoccerNet feature-frame (2 fps), mean-pooling
the encoder's spatiotemporal patch tokens into a single 1024-d vector. The result
is saved as <half>_VJEPA21_L.npy of shape (T, 1024), MIRRORING the layout of the
precomputed <half>_ResNET_TF2.npy so the rest of the pipeline (dataset / probe /
train / spot) reuses it unchanged — only feat_dim changes (2048 -> 1024).

Key facts (see plan):
  - V-JEPA 2.1 is NOT in HF transformers; load via torch.hub:
        torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
  - Encoder input: (B, num_frames, C, 384, 384), ImageNet-normalized.
  - Encoder output: (B, num_tokens, 1024) -> mean over tokens -> (B, 1024).
  - SoccerNet features are 2 fps; the raw .mkv is 25 fps. Frame i (i/2 seconds)
    maps to video frame round(i / FPS * video_fps); we take a tubelet-friendly
    window of NUM_FRAMES around it.

The script is resumable: halves whose .npy already exists are skipped.

Run (after videos are downloaded):
    python -m scripts.extract_vjepa
    python -m scripts.extract_vjepa --data_dir /kaggle/working/soccernet --out_dir /kaggle/working/soccernet
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from footy.config import VJEPA_GAME_PATHS, default_data_dir

# ── Constants ─────────────────────────────────────────────────────────────────
FPS = 2                      # SoccerNet feature frame rate (matches src/dataset.py)
HUB_REPO = "facebookresearch/vjepa2"
HUB_MODEL = "vjepa2_1_vit_large_384"
RESOLUTION = 384
NUM_FRAMES = 16              # frames per clip window fed to the encoder (multiple of tubelet_size=2)
FEAT_DIM = 1024              # ViT-L hidden size
FEATURE_TAG = "VJEPA21_L"    # output filename tag: <half>_VJEPA21_L.npy

# ImageNet normalization (V-JEPA default preprocessing)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_encoder(device: torch.device):
    """Load the V-JEPA 2.1 ViT-L encoder from torch.hub."""
    model = torch.hub.load(HUB_REPO, HUB_MODEL)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def encode_windows(model, windows: torch.Tensor) -> np.ndarray:
    """
    windows: (B, NUM_FRAMES, 3, RES, RES) already normalized, on device.
    Returns: (B, FEAT_DIM) mean-pooled token features as float32 numpy.
    """
    # Prefer the documented feature accessor; fall back to forward().last_hidden_state.
    if hasattr(model, "get_vision_features"):
        tokens = model.get_vision_features(windows)
    else:
        out = model(windows)
        tokens = getattr(out, "last_hidden_state", out)

    # tokens: (B, num_tokens, FEAT_DIM) -> mean-pool over the token axis.
    if tokens.dim() == 3:
        feats = tokens.mean(dim=1)
    elif tokens.dim() == 2:
        feats = tokens  # already pooled
    else:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(tokens.shape)}")
    return feats.float().cpu().numpy()


# ── Video → normalized clip windows ───────────────────────────────────────────

def _open_decoder(video_path: Path):
    """Return (decoder, total_frames, video_fps). Uses torchcodec, falls back to decord."""
    try:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(video_path))
        meta = dec.metadata
        n = meta.num_frames
        fps = float(meta.average_fps)
        return ("torchcodec", dec), n, fps
    except Exception:
        import decord  # type: ignore

        vr = decord.VideoReader(str(video_path))
        n = len(vr)
        fps = float(vr.get_avg_fps())
        return ("decord", vr), n, fps


def _get_window_frames(decoder, center_vframe: int, total_frames: int) -> torch.Tensor:
    """
    Fetch NUM_FRAMES raw frames centered on center_vframe, clamped to [0, total_frames).
    Returns uint8/float tensor (NUM_FRAMES, 3, H, W).
    """
    half = NUM_FRAMES // 2
    idxs = [min(max(center_vframe - half + k, 0), total_frames - 1) for k in range(NUM_FRAMES)]

    kind, dec = decoder
    if kind == "torchcodec":
        frames = dec.get_frames_at(indices=idxs).data  # (T, C, H, W)
        return frames
    else:  # decord
        import torch as _t

        arr = dec.get_batch(idxs).asnumpy()  # (T, H, W, C)
        return _t.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)


def _preprocess(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    frames: (T, C, H, W) uint8 or float. Resize to RESOLUTION, scale to [0,1],
    ImageNet-normalize. Returns (T, 3, RES, RES) float on device.
    """
    import torch.nn.functional as F

    x = frames.float()
    if x.max() > 1.5:           # uint8-range -> [0,1]
        x = x / 255.0
    x = F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return x.to(device)


# ── Per-half extraction ───────────────────────────────────────────────────────

def expected_T(game_dir: Path, half: int) -> int | None:
    """Length T from the matching ResNet feature file, so V-JEPA T stays aligned."""
    resnet = game_dir / f"{half}_ResNET_TF2.npy"
    if resnet.exists():
        return int(np.load(resnet, mmap_mode="r").shape[0])
    return None


def extract_half(model, game_dir: Path, half: int, device: torch.device, batch: int) -> np.ndarray:
    video_path = game_dir / f"{half}_224p.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    decoder, total_frames, video_fps = _open_decoder(video_path)

    T = expected_T(game_dir, half)
    if T is None:
        # No ResNet reference; derive T from video duration at 2 fps.
        T = int(total_frames / video_fps * FPS)
    print(f"    half {half}: video {total_frames} frames @ {video_fps:.2f} fps -> T={T}")

    feats = np.zeros((T, FEAT_DIM), dtype=np.float32)

    pending_idx: list[int] = []
    pending_clips: list[torch.Tensor] = []

    def flush():
        if not pending_clips:
            return
        windows = torch.stack(pending_clips, dim=0)  # (B, NUM_FRAMES, 3, RES, RES)
        out = encode_windows(model, windows)
        for i, vec in zip(pending_idx, out):
            feats[i] = vec
        pending_idx.clear()
        pending_clips.clear()

    for i in range(T):
        center_vframe = round(i / FPS * video_fps)
        frames = _get_window_frames(decoder, center_vframe, total_frames)
        clip = _preprocess(frames, device)  # (NUM_FRAMES, 3, RES, RES)
        pending_idx.append(i)
        pending_clips.append(clip)
        if len(pending_clips) == batch:
            flush()
        if i % 500 == 0 and i:
            print(f"      {i}/{T} frames")
    flush()

    return feats


def run(data_dir: str, out_dir: str, batch: int, games: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {HUB_MODEL} from torch.hub ...")
    model = load_encoder(device)

    for game in games:
        game_dir = Path(data_dir) / game
        out_game_dir = Path(out_dir) / game
        out_game_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {game} ===")
        for half in (1, 2):
            out_path = out_game_dir / f"{half}_{FEATURE_TAG}.npy"
            if out_path.exists():
                print(f"    half {half}: {out_path.name} exists, skipping")
                continue
            if not (game_dir / f"{half}_224p.mkv").exists():
                print(f"    half {half}: no video, skipping")
                continue
            feats = extract_half(model, game_dir, half, device, batch)
            np.save(out_path, feats)
            print(f"    saved {out_path}  shape={feats.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=default_data_dir(),
                        help="Where the *_224p.mkv videos live")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write *_VJEPA21_L.npy (default: same as data_dir)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Clip windows per encoder forward pass")
    parser.add_argument("--games", nargs="*", default=None,
                        help="Game paths (default: footy.config.VJEPA_GAME_PATHS)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    games = args.games or VJEPA_GAME_PATHS
    run(args.data_dir, out_dir, args.batch, games)
