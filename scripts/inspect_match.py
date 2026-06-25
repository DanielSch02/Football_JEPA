"""
Downloads one match half's Labels-v2.json + ResNET_TF2.npy features,
then shows that annotations and feature frames line up.
"""

import json
import unittest.mock
import numpy as np

# Patch telemetry before SoccerNet is imported so the GA ping doesn't block
with unittest.mock.patch("google_measurement_protocol.report", return_value=[]):
    from SoccerNet.Downloader import SoccerNetDownloader

import sys
# Keep telemetry silent for all subsequent calls
sys.modules["google_measurement_protocol"].report = lambda *_, **__: []

GAME = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
HALF = 1          # 1 or 2
FPS  = 2          # SoccerNet precomputed features are at 2 fps
DATA_DIR = "./data/soccernet"

# ── Download ──────────────────────────────────────────────────────────────────

d = SoccerNetDownloader(DATA_DIR)
d.password = "s0cc3rn3t"

# Labels need your NDA password; features use the public "SoccerNet" password
d.downloadGame(game=GAME, files=["Labels-v2.json"], spl="train")
d.downloadGame(game=GAME, files=[f"{HALF}_ResNET_TF2.npy"], spl="train")

# ── Load labels ───────────────────────────────────────────────────────────────

label_path = f"{DATA_DIR}/{GAME}/Labels-v2.json"
with open(label_path) as f:
    labels = json.load(f)

half_events = [a for a in labels["annotations"] if int(a["gameTime"][0]) == HALF]

print(f"\n=== Half {HALF} events ({len(half_events)} total) ===")
for ev in half_events:
    print(f"  {ev['gameTime']}  {ev['label']}")

# ── Load features ─────────────────────────────────────────────────────────────

feat_path = f"{DATA_DIR}/{GAME}/{HALF}_ResNET_TF2.npy"
features = np.load(feat_path)   # shape: (num_frames, feature_dim)
print(f"\nFeature array shape: {features.shape}  ({FPS} fps → {features.shape[0]/FPS/60:.1f} min)")

# ── Line them up ─────────────────────────────────────────────────────────────

def timestamp_to_frame(game_time_str: str, fps: int) -> int:
    # game_time format: "1 - MM:SS"
    _, clock = game_time_str.split(" - ")
    mm, ss = map(int, clock.split(":"))
    total_seconds = mm * 60 + ss
    return int(total_seconds * fps)

print(f"\n=== Annotation → frame alignment ===")
for ev in half_events:
    frame_idx = timestamp_to_frame(ev["gameTime"], FPS)
    # clamp to valid range
    frame_idx = min(frame_idx, features.shape[0] - 1)
    vec = features[frame_idx]
    print(
        f"  '{ev['label']}' at {ev['gameTime'].split(' - ')[1]}"
        f" → frame {frame_idx}"
        f" → feature vector shape {vec.shape}, "
        f"min={vec.min():.3f} max={vec.max():.3f} mean={vec.mean():.3f}"
    )
