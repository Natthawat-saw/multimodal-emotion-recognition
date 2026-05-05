"""
train.py — Main training script

Usage:
    python src/train.py --img_train data/fer2013/train \
                        --img_test  data/fer2013/test  \
                        --audio_root data/ravdess/audio_speech_actors_01-24 \
                        --save_dir  checkpoints/
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import build_loaders
from data.fer2013 import FER2013Dataset, scan_ravdess
from data.ravdess import scan_ravdess
from models.framework import EmotionModel
from models.fusion import clip_contrastive_loss
from utils.metrics import evaluate, plot_confusion


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_alpha(epoch: int) -> float:
    """Staged contrastive loss weight schedule."""
    if epoch < 5:
        return 0.08 * (epoch + 1)   # warm-up: 0.08 → 0.40
    elif epoch < 10:
        return 0.4
    return 0.25


def train_one_epoch(model, loader, optimizer, ce, alpha, device) -> float:
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for imgs, auds, labels in pbar:
        imgs, auds, labels = imgs.to(device), auds.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, v, a = model(img=imgs, mel=auds)

        loss_ce = ce(logits, labels)
        loss_clip = clip_contrastive_loss(v, a)
        loss = loss_ce + alpha * loss_clip

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n += len(labels)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{loss_ce.item():.4f}",
            clip=f"{loss_clip.item():.4f}",
        )

    return total_loss / max(1, n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_train",   required=True, help="FER2013 train/ folder")
    p.add_argument("--img_test",    required=True, help="FER2013 test/ folder")
    p.add_argument("--audio_root",  required=True, help="RAVDESS audio folder")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--proj_dim",    type=int,   default=256)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--max_per_class",type=int,  default=300)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    img_train_ds = FER2013Dataset(
        args.img_train, max_per_class=args.max_per_class, seed=args.seed
    )
    img_val_ds = FER2013Dataset(args.img_test)
    audio_by_label = scan_ravdess(args.audio_root)

    train_loader, val_loader = build_loaders(
        img_train_ds,
        img_val_ds,
        audio_by_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────
    model = EmotionModel(proj_dim=args.proj_dim).to(device)
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────
    best_f1, best_path = -1.0, ""

    for epoch in range(args.epochs):
        alpha = get_alpha(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, ce, alpha, device)

        model.eval()
        with torch.no_grad():
            val_acc, val_f1, val_report, y_true, y_pred = evaluate(
                model, val_loader, device
            )

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f} | α={alpha:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(
                args.save_dir, f"best_epoch{epoch+1}_f1{val_f1:.4f}.pt"
            )
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved → {best_path}")

    print(f"\nBest F1: {best_f1:.4f}  |  checkpoint: {best_path}")
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, _, report, y_true, y_pred = evaluate(model, val_loader, device)
    print(report)
    plot_confusion(y_true, y_pred, save_path="assets/confusion_matrix.png")


if __name__ == "__main__":
    main()
