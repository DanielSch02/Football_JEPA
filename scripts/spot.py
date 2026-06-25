"""
Milestone 3: slide the trained probe across a full untrimmed match half,
peak-pick into predicted timestamps, evaluate with SoccerNet Average-mAP.

Usage (run from project root):
    python -m scripts.spot
    python -m scripts.spot --game "england_epl/2014-2015/..." --half 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from footy.dataset import (
    CLIP_LEN, HALF_CLIP, FPS, EVENT_LABELS, NUM_CLASSES,
    LABEL_TO_IDX, FEATURE_DIMS, load_half, _extract_clip,
)
from footy.probe import AttentiveProbe

RESULTS_DIR = Path("results")


def checkpoint_path(feature_tag: str) -> Path:
    """Mirror scripts/train.py: ResNET_TF2 -> probe.pt; other tags -> probe_<tag>.pt."""
    if feature_tag == "ResNET_TF2":
        return RESULTS_DIR / "probe.pt"
    return RESULTS_DIR / f"probe_{feature_tag}.pt"


# ── Sliding window inference ──────────────────────────────────────────────────

def score_half(
    model: AttentiveProbe,
    features: np.ndarray,
    device: torch.device,
    stride: int = 1,
) -> np.ndarray:
    """
    Slide a window of CLIP_LEN over the feature array.
    Returns score_matrix of shape (T, NUM_CLASSES) with softmax probabilities,
    where T = len(features).
    """
    model.eval()
    T = len(features)
    scores = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    centers = range(HALF_CLIP, T - HALF_CLIP, stride)
    batch_clips, batch_centers = [], []

    def flush():
        if not batch_clips:
            return
        x = torch.tensor(np.stack(batch_clips), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
        for center, p in zip(batch_centers, probs):
            scores[center] += p
            counts[center] += 1
        batch_clips.clear()
        batch_centers.clear()

    BATCH = 128
    for c in centers:
        clip = _extract_clip(features, c)
        batch_clips.append(clip)
        batch_centers.append(c)
        if len(batch_clips) == BATCH:
            flush()
    flush()

    # Fill un-scored boundary frames from nearest scored neighbour
    for t in range(T):
        if counts[t] == 0:
            # find nearest scored frame
            nearest = min((abs(t - c) for c in centers), default=0)
            nc = min(centers, key=lambda c: abs(c - t)) if len(list(centers)) else t
            scores[t] = scores[nc]
            counts[t] = 1

    scores /= counts[:, None].clip(1)
    return scores  # (T, NUM_CLASSES)


# ── Peak picking ──────────────────────────────────────────────────────────────

def pick_peaks(
    scores: np.ndarray,
    class_idx: int,
    min_distance: int = FPS * 30,   # 30 s minimum gap between same-class peaks
    threshold: float = 0.1,
) -> list[tuple[int, float]]:
    """
    Return list of (frame_index, confidence) for predicted events of class_idx,
    sorted by descending confidence.
    """
    signal = scores[:, class_idx]
    peaks, props = find_peaks(signal, distance=min_distance, height=threshold)
    results = [(int(p), float(signal[p])) for p in peaks]
    results.sort(key=lambda x: -x[1])
    return results


# ── Average-mAP (SoccerNet protocol) ─────────────────────────────────────────

def frame_to_ms(frame: int) -> int:
    return int(frame / FPS * 1000)


def compute_ap(
    predictions: list[tuple[int, float]],
    ground_truth_frames: list[int],
    tolerance_frames: int,
) -> float:
    """
    Compute Average Precision for one class in one half.
    A prediction is a TP if within tolerance_frames of an unmatched GT event.
    """
    if not ground_truth_frames:
        return float("nan")
    if not predictions:
        return 0.0

    matched = set()
    tp, fp = [], []
    for frame, conf in sorted(predictions, key=lambda x: -x[1]):
        hit = False
        for i, gt in enumerate(ground_truth_frames):
            if i not in matched and abs(frame - gt) <= tolerance_frames:
                matched.add(i)
                hit = True
                break
        tp.append(1 if hit else 0)
        fp.append(0 if hit else 1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall    = tp_cum / len(ground_truth_frames)

    # area under P-R curve (trapezoid)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def evaluate_spotting(
    model: AttentiveProbe,
    data_dir: str,
    game: str,
    half: int,
    device: torch.device,
    tolerances_sec: tuple[int, ...] = (5, 10, 30, 60),
    feature_tag: str = "ResNET_TF2",
) -> dict:
    features, events = load_half(data_dir, game, half, feature_tag)
    scores = score_half(model, features, device)

    results = {}
    for tol_s in tolerances_sec:
        tol_frames = tol_s * FPS
        aps = []
        for cls_idx in range(1, NUM_CLASSES):    # skip Background
            cls_name = EVENT_LABELS[cls_idx]
            gt_frames = [e["frame"] for e in events if e["label_idx"] == cls_idx]
            preds = pick_peaks(scores, cls_idx)
            ap = compute_ap(preds, gt_frames, tol_frames)
            if not np.isnan(ap):
                aps.append((cls_name, ap))
        mean_ap = np.mean([ap for _, ap in aps]) if aps else 0.0
        results[f"mAP@{tol_s}s"] = mean_ap
        results[f"per_class@{tol_s}s"] = aps

    return results, scores, events


def print_spotting_results(results: dict, game: str, half: int):
    print(f"\n=== Spotting results: {game}  Half {half} ===")
    for tol in (5, 10, 30, 60):
        key = f"mAP@{tol}s"
        if key in results:
            print(f"  {key}: {results[key]:.4f}")

    # Show per-class AP for one tolerance
    tol = 60
    print(f"\nPer-class AP @ {tol}s tolerance:")
    for cls_name, ap in sorted(results.get(f"per_class@{tol}s", []), key=lambda x: -x[1]):
        bar = "#" * int(ap * 20)
        print(f"  {cls_name:<25} {ap:.3f}  {bar}")


def demo_query(scores: np.ndarray, events: list[dict], event_name: str):
    """Print a ranked list of predicted timestamps for a given event class."""
    if event_name not in LABEL_TO_IDX:
        print(f"Unknown event: {event_name}")
        return
    cls_idx = LABEL_TO_IDX[event_name]
    preds = pick_peaks(scores, cls_idx)

    print(f"\n=== Query: '{event_name}' ===")
    print(f"{'Rank':<6} {'Time':>8} {'Confidence':>12}  {'GT match?':>10}")
    gt_frames = [e["frame"] for e in events if e["label_idx"] == cls_idx]
    tolerance = 60 * FPS
    for rank, (frame, conf) in enumerate(preds[:10], 1):
        mm = frame // (FPS * 60)
        ss = (frame // FPS) % 60
        match = any(abs(frame - g) <= tolerance for g in gt_frames)
        print(f"{rank:<6} {mm:02d}:{ss:02d}     {conf:>12.4f}  {'YES' if match else 'no':>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/soccernet")
    parser.add_argument("--game",     default="england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley")
    parser.add_argument("--half",     type=int, default=1)
    parser.add_argument("--query",    default="Corner",
                        help="Event class to demo-query")
    parser.add_argument("--feature_tag", default="ResNET_TF2", choices=list(FEATURE_DIMS),
                        help="Feature backend: ResNET_TF2 (baseline) or VJEPA21_L")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = checkpoint_path(args.feature_tag)
    if not checkpoint.exists():
        print(f"No checkpoint found at {checkpoint}. Run train.py --feature_tag {args.feature_tag} first.")
        exit(1)

    model = AttentiveProbe(feat_dim=FEATURE_DIMS[args.feature_tag]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded checkpoint from {checkpoint}  (feature_tag={args.feature_tag})")

    results, scores, events = evaluate_spotting(
        model, args.data_dir, args.game, args.half, device, feature_tag=args.feature_tag
    )
    print_spotting_results(results, args.game, args.half)
    demo_query(scores, events, args.query)
