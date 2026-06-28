"""
SoccerNet Action Spotting v2 clip dataset.

Design decisions:
- Clip length: 30 frames (15 s at 2 fps), centered on the event anchor.
  At 2 fps the raw video equivalent is ~375 raw frames. 15 s captures a full
  corner sequence (cross → header → clearance) or a build-up to a goal.
  7.5 s (15 frames) was too short for sequences that unfold over ~10 s.
- Background negatives: randomly sampled windows whose center is at least
  CLIP_LEN frames away from every annotated event in the same half.
  Ratio: 1:1 event-to-background per half, giving balanced classes overall.
  We do NOT bias towards "hard" near-event negatives here — the sliding window
  at inference time will stress-test that; training on uniform negatives avoids
  teaching the model to be uncertain near events.
- Class design: all 17 SoccerNet v2 event types + 1 background = 18 classes.
  No collapsing. The confusion matrix will reveal natural confusions.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

FPS = 2
CLIP_LEN = 30          # frames; 30 frames × 0.5 s = 15 s of context per clip
HALF_CLIP = CLIP_LEN // 2
MIN_NEG_DIST = CLIP_LEN  # minimum distance from any event to call it background

# Feature backends. Each tag maps to its precomputed .npy filename suffix and its
# per-frame feature dimension. The probe's feat_dim must match the chosen tag.
#   ResNET_TF2 : SoccerNet's precomputed ResNet features (the baseline).
#   VJEPA21_L  : V-JEPA 2.1 ViT-L features produced by scripts/extract_vjepa.py.
DEFAULT_FEATURE_TAG = "ResNET_TF2"
FEATURE_DIMS = {
    "ResNET_TF2": 2048,
    "VJEPA21_L": 1024,
}

# All 17 SoccerNet v2 event labels (discovered from the full label taxonomy).
# Background is class 0; events are 1-17.
BACKGROUND = "Background"
EVENT_LABELS = [
    BACKGROUND,
    "Ball out of play",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Kick-off",
    "Offside",
    "Red card",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Throw-in",
    "Yellow card",
    "Yellow->red card",
    "Penalty",
]
NUM_CLASSES = len(EVENT_LABELS)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(EVENT_LABELS)}


def _parse_position_ms(annotation: dict) -> int:
    """Return the 'position' field (ms from half start) as an int."""
    return int(annotation["position"])


def _ms_to_frame(ms: int) -> int:
    return int(ms / 1000 * FPS)


def _gametime_to_frame(game_time: str) -> int:
    """'1 - MM:SS' → frame index within that half."""
    _, clock = game_time.split(" - ")
    mm, ss = map(int, clock.split(":"))
    return (mm * 60 + ss) * FPS


def load_half(
    data_dir: str,
    game: str,
    half: int,
    feature_tag: str = DEFAULT_FEATURE_TAG,
) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
        features: float32 array of shape (T, D)
        events:   list of dicts with keys 'frame' and 'label_idx'
    """
    from footy.config import resolve_game_dir
    game_path = resolve_game_dir(data_dir, game)
    feat_path = game_path / f"{half}_{feature_tag}.npy"
    label_path = game_path / "Labels-v2.json"

    features = np.load(feat_path).astype(np.float32)

    with open(label_path) as f:
        raw = json.load(f)

    events = []
    for ann in raw["annotations"]:
        half_id = int(ann["gameTime"][0])
        if half_id != half:
            continue
        label = ann["label"]
        if label not in LABEL_TO_IDX:
            # unknown label → skip (e.g. future schema additions)
            continue
        # Use position (ms) for sub-second precision; fall back to gameTime
        frame = _ms_to_frame(int(ann["position"])) if ann["position"] != "0" or ann["gameTime"].endswith("00:00") else _gametime_to_frame(ann["gameTime"])
        frame = min(frame, len(features) - 1)
        events.append({"frame": frame, "label_idx": LABEL_TO_IDX[label]})

    return features, events


def _extract_clip(features: np.ndarray, center: int) -> np.ndarray:
    """Extract a fixed-length clip, padding with zeros at boundaries."""
    T = len(features)
    start = center - HALF_CLIP
    end = start + CLIP_LEN
    if start >= 0 and end <= T:
        return features[start:end]
    # boundary — zero-pad
    clip = np.zeros((CLIP_LEN, features.shape[1]), dtype=np.float32)
    src_start = max(0, start)
    src_end = min(T, end)
    dst_start = src_start - start
    clip[dst_start: dst_start + (src_end - src_start)] = features[src_start:src_end]
    return clip


def _sample_negatives(
    features: np.ndarray,
    events: list[dict],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Sample n background center-frames, all at least MIN_NEG_DIST from any event."""
    event_frames = {e["frame"] for e in events}
    T = len(features)
    candidates = [
        f for f in range(HALF_CLIP, T - HALF_CLIP)
        if all(abs(f - ef) >= MIN_NEG_DIST for ef in event_frames)
    ]
    if len(candidates) < n:
        return candidates
    return rng.sample(candidates, n)


class SoccerNetClipDataset(Dataset):
    """
    Yields (clip, label_idx) pairs.

    clip:      float32 tensor of shape (CLIP_LEN, feat_dim)
    label_idx: int in [0, NUM_CLASSES)  — 0 is Background
    """

    def __init__(
        self,
        data_dir: str,
        split: str,                    # "train" | "valid" | "test"
        games: Optional[list[str]] = None,
        neg_ratio: float = 1.0,        # background samples per event sample
        seed: int = 42,
        feature_tag: str = DEFAULT_FEATURE_TAG,
    ):
        from SoccerNet.utils import getListGames

        self.data_dir = data_dir
        self.feature_tag = feature_tag
        self.rng = random.Random(seed)
        self.samples: list[tuple[np.ndarray, int]] = []  # (clip, label_idx)

        if games is None:
            games = getListGames(split, task="spotting")

        from footy.config import resolve_game_dir
        missing = 0
        for game in games:
            game_dir = resolve_game_dir(data_dir, game)
            for half in (1, 2):
                feat_path = game_dir / f"{half}_{feature_tag}.npy"
                label_path = game_dir / "Labels-v2.json"
                if not feat_path.exists() or not label_path.exists():
                    missing += 1
                    continue

                features, events = load_half(data_dir, game, half, feature_tag)

                # Positive samples — one clip per annotation
                for ev in events:
                    clip = _extract_clip(features, ev["frame"])
                    self.samples.append((clip, ev["label_idx"]))

                # Background samples
                n_neg = max(1, int(len(events) * neg_ratio))
                for center in _sample_negatives(features, events, n_neg, self.rng):
                    clip = _extract_clip(features, center)
                    self.samples.append((clip, 0))  # 0 = Background

        if missing:
            print(f"[dataset] Skipped {missing} halves (features not downloaded yet)")

        self.rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip, label_idx = self.samples[idx]
        return torch.from_numpy(clip), label_idx

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {lbl: 0 for lbl in EVENT_LABELS}
        for _, idx in self.samples:
            counts[EVENT_LABELS[idx]] += 1
        return counts


if __name__ == "__main__":
    # Quick smoke test on the one downloaded game
    DATA_DIR = "./data/soccernet"
    GAME = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

    ds = SoccerNetClipDataset(DATA_DIR, split="train", games=[GAME])
    print(f"Total samples: {len(ds)}")
    print("\nClass distribution:")
    for lbl, cnt in ds.class_counts().items():
        if cnt:
            print(f"  {lbl:<25} {cnt}")

    clip, label = ds[0]
    print(f"\nSample 0: clip={clip.shape}, label={EVENT_LABELS[label]} ({label})")
    clip, label = ds[10]
    print(f"Sample 10: clip={clip.shape}, label={EVENT_LABELS[label]} ({label})")
