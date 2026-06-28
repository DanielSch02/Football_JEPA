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

from footy.config import VJEPA_GAME_PATHS, VJEPA_GAME_PATHS_FULL, default_data_dir, resolve_game_dir

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


# Real Meta CDN for the pretrained weights. The vjepa2 repo's backbones.py ships
# with a testing override (`VJEPA_BASE_URL = "http://localhost:8300"`) that clobbers
# the real URL, so torch.hub.load fails with ConnectionRefused. We patch it back.
VJEPA_CDN = "https://dl.fbaipublicfiles.com/vjepa2"


# ── Model ─────────────────────────────────────────────────────────────────────

def load_encoder(device: torch.device):
    """
    Load the V-JEPA 2.1 ViT-L encoder.

    We download/cache the vjepa2 repo source, import its backbones module, patch
    the broken `VJEPA_BASE_URL = "http://localhost:8300"` testing placeholder back
    to the real Meta CDN, then call the entrypoint *directly from that module* so
    the patch is guaranteed to be in effect. The entrypoint returns an
    (encoder, predictor) tuple; we keep only the encoder.
    """
    import sys

    # Ensure the repo source is cached. torch.hub.list() downloads the repo (if
    # needed) and runs no entrypoint — avoiding the private _get_cache_or_reload
    # whose signature varies across torch versions.
    owner_repo = HUB_REPO.split("/")
    repo_dir = Path(torch.hub.get_dir()) / f"{owner_repo[0]}_{owner_repo[1]}_main"
    if not (repo_dir / "src" / "hub" / "backbones.py").exists():
        torch.hub.list(HUB_REPO, force_reload=False, trust_repo=True)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import src.hub.backbones as backbones  # the vjepa2 repo's own 'src' package

    if str(getattr(backbones, "VJEPA_BASE_URL", "")).startswith("http://localhost"):
        print(f"    patching VJEPA_BASE_URL -> {VJEPA_CDN}")
        backbones.VJEPA_BASE_URL = VJEPA_CDN

    entry = getattr(backbones, HUB_MODEL)
    out = entry(pretrained=True)
    encoder = out[0] if isinstance(out, (tuple, list)) else out
    return encoder.to(device).eval()


@torch.no_grad()
def encode_windows(model, windows: torch.Tensor) -> np.ndarray:
    """
    windows: (B, NUM_FRAMES, 3, RES, RES) already normalized, on device.
    Returns: (B, FEAT_DIM) mean-pooled token features as float32 numpy.

    The V-JEPA 2.1 torch.hub encoder's patch_embed is a Conv3d expecting input in
    (B, C, T, H, W) order, so we transpose frames<->channels before the forward.
    It returns all patch tokens (B, num_tokens, embed_dim); we mean-pool tokens.
    """
    x = windows.transpose(1, 2)   # (B, NUM_FRAMES, 3, H, W) -> (B, 3, NUM_FRAMES, H, W)
    tokens = model(x)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    tokens = getattr(tokens, "last_hidden_state", tokens)

    if tokens.dim() == 3:
        feats = tokens.mean(dim=1)        # (B, embed_dim)
    elif tokens.dim() == 2:
        feats = tokens
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


def _get_frames_at(decoder, idxs: list[int]) -> torch.Tensor:
    """Fetch the given video-frame indices. Returns (len(idxs), 3, H, W)."""
    kind, dec = decoder
    if kind == "torchcodec":
        return dec.get_frames_at(indices=idxs).data  # (N, C, H, W)
    else:  # decord
        import torch as _t

        arr = dec.get_batch(idxs).asnumpy()  # (N, H, W, C)
        return _t.from_numpy(arr).permute(0, 3, 1, 2)  # (N, C, H, W)


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
    """
    Tiled extraction: split the half into non-overlapping NUM_FRAMES-frame clips
    (one clip ≈ NUM_FRAMES/FPS seconds). Run the encoder once per clip and
    broadcast each clip's pooled vector to the SoccerNet frame-rows it covers.
    This does ~NUM_FRAMES× fewer forward passes than one window per frame.
    """
    video_path = game_dir / f"{half}_224p.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # Random-access video decode from a network-backed mount (/kaggle/input) is
    # ~6x slower than from local disk. Copy the .mkv to fast local /tmp first,
    # decode there, delete after. Skips the copy if already local.
    import shutil, tempfile, os
    local_video = video_path
    tmp_copy = None
    if str(video_path).startswith("/kaggle/input"):
        tmp_copy = Path(tempfile.gettempdir()) / f"_extract_{half}_{os.getpid()}.mkv"
        print(f"    copying video to local {tmp_copy} ...")
        shutil.copy(video_path, tmp_copy)
        local_video = tmp_copy

    try:
        decoder, total_frames, video_fps = _open_decoder(local_video)
        return _extract_from_decoder(model, game_dir, half, decoder, total_frames, video_fps, device, batch)
    finally:
        if tmp_copy is not None and tmp_copy.exists():
            tmp_copy.unlink()


def _extract_from_decoder(model, game_dir, half, decoder, total_frames, video_fps, device, batch):

    T = expected_T(game_dir, half)
    if T is None:
        T = int(total_frames / video_fps * FPS)

    # Non-overlapping clips over the T SoccerNet frames.
    clip_starts = list(range(0, T, NUM_FRAMES))
    print(f"    half {half}: video {total_frames} frames @ {video_fps:.2f} fps "
          f"-> T={T}, {len(clip_starts)} clips of {NUM_FRAMES}")

    feats = np.zeros((T, FEAT_DIM), dtype=np.float32)

    pending_starts: list[int] = []
    pending_clips: list[torch.Tensor] = []

    def flush():
        if not pending_clips:
            return
        windows = torch.stack(pending_clips, dim=0)  # (B, NUM_FRAMES, 3, RES, RES)
        out = encode_windows(model, windows)         # (B, FEAT_DIM)
        for start, vec in zip(pending_starts, out):
            end = min(start + NUM_FRAMES, T)
            feats[start:end] = vec                   # broadcast to the rows this clip covers
        pending_starts.clear()
        pending_clips.clear()

    for n, start in enumerate(clip_starts):
        # SoccerNet frames start..start+NUM_FRAMES-1 -> their video-frame indices.
        sn_frames = range(start, min(start + NUM_FRAMES, T))
        vidx = [min(round(i / FPS * video_fps), total_frames - 1) for i in sn_frames]
        # pad a short final clip up to NUM_FRAMES by repeating the last index.
        while len(vidx) < NUM_FRAMES:
            vidx.append(vidx[-1])

        frames = _get_frames_at(decoder, vidx)          # (NUM_FRAMES, 3, H, W)
        clip = _preprocess(frames, device)              # (NUM_FRAMES, 3, RES, RES)
        pending_starts.append(start)
        pending_clips.append(clip)
        if len(pending_clips) == batch:
            flush()
        if n % 20 == 0:
            print(f"      clip {n}/{len(clip_starts)}")
    flush()

    return feats


def run(data_dir: str, out_dir: str, batch: int, games: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {HUB_MODEL} from torch.hub ...")
    model = load_encoder(device)

    for game in games:
        # Input videos may be flattened by Kaggle's dataset save -> resolve actual dir.
        game_dir = resolve_game_dir(data_dir, game)
        # Output keeps the canonical england_epl/<season>/<game> layout.
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
                        help="Explicit game paths (e.g. a chunk); overrides --full")
    parser.add_argument("--full", action="store_true",
                        help="Use the full 25-game baseline set (default: 3-game smoke set)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    if args.games:
        games = args.games
    elif args.full:
        games = VJEPA_GAME_PATHS_FULL
    else:
        games = VJEPA_GAME_PATHS
    run(args.data_dir, out_dir, args.batch, games)
