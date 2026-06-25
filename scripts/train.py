"""
Train the attentive probe on downloaded SoccerNet halves.
Outputs:
  - per-class accuracy table (stdout)
  - confusion matrix saved to results/confusion_matrix.png
  - model checkpoint saved to results/probe.pt
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.dataset import SoccerNetClipDataset, EVENT_LABELS, NUM_CLASSES, FEATURE_DIMS
from src.probe import AttentiveProbe

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def checkpoint_path(feature_tag: str) -> Path:
    """ResNET_TF2 -> results/probe.pt (baseline); other tags -> results/probe_<tag>.pt."""
    if feature_tag == "ResNET_TF2":
        return RESULTS_DIR / "probe.pt"
    return RESULTS_DIR / f"probe_{feature_tag}.pt"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feat_dim = FEATURE_DIMS[args.feature_tag]
    ckpt_path = checkpoint_path(args.feature_tag)
    print(f"Feature backend: {args.feature_tag} (feat_dim={feat_dim})  ->  {ckpt_path}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = SoccerNetClipDataset(args.data_dir, split="train", games=args.games,
                                    feature_tag=args.feature_tag)
    val_ds   = SoccerNetClipDataset(args.data_dir, split="valid", games=args.games,
                                    feature_tag=args.feature_tag)

    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    print("\nTrain class distribution:")
    for lbl, cnt in train_ds.class_counts().items():
        if cnt:
            print(f"  {lbl:<25} {cnt}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AttentiveProbe(feat_dim=feat_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Class-weighted loss to handle imbalance (Background dominates)
    counts = np.zeros(NUM_CLASSES)
    for _, label in train_ds.samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    weights = weights / weights.sum() * NUM_CLASSES  # normalise to sum to num_classes

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for clips, labels in train_loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={total_loss/total:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print("Loading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    all_preds, all_labels = predict_all(model, val_loader, device)
    print_per_class_accuracy(all_preds, all_labels)
    plot_confusion_matrix(all_preds, all_labels)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total, total_loss / total


def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            preds = model(clips).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def print_per_class_accuracy(preds, labels):
    preds  = np.array(preds)
    labels = np.array(labels)
    present = sorted(set(labels))

    print("\n=== Per-class accuracy ===")
    print(f"{'Class':<25} {'Correct':>7} {'Total':>7} {'Acc':>7}")
    print("-" * 50)
    for idx in present:
        mask = labels == idx
        n = mask.sum()
        correct = (preds[mask] == idx).sum()
        print(f"{EVENT_LABELS[idx]:<25} {correct:>7} {n:>7} {correct/n:>7.3f}")


def plot_confusion_matrix(preds, labels):
    present = sorted(set(labels))
    names   = [EVENT_LABELS[i] for i in present]
    cm = confusion_matrix(labels, preds, labels=present)
    # Normalise rows to [0,1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(max(8, len(present)), max(6, len(present))))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")

    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"\nConfusion matrix saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/soccernet")
    parser.add_argument("--games",    nargs="*", default=None,
                        help="Restrict to specific game paths (default: full split)")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--feature_tag", default="ResNET_TF2", choices=list(FEATURE_DIMS),
                        help="Feature backend: ResNET_TF2 (baseline) or VJEPA21_L")
    args = parser.parse_args()
    train(args)
