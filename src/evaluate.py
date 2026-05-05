"""
evaluate.py — Load a checkpoint and run evaluation or single-sample inference

Usage (full test set):
    python src/evaluate.py --ckpt checkpoints/best.pt \
                           --img_test  data/fer2013/test \
                           --audio_root data/ravdess/audio_speech_actors_01-24

Usage (single sample):
    python src/evaluate.py --ckpt checkpoints/best.pt \
                           --image path/to/face.jpg \
                           --audio path/to/speech.wav
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from data.fer2013 import FER2013Dataset, CLASS_NAMES, default_transform
from data.ravdess import scan_ravdess, wav_to_mel, pad_or_crop, TARGET_T
from data.paired import build_loaders
from models.framework import EmotionModel
from utils.metrics import evaluate, bootstrap_metrics, plot_confusion


def load_model(ckpt_path: str, device: str, proj_dim: int = 256) -> EmotionModel:
    model = EmotionModel(proj_dim=proj_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def predict(model, img_path: str, audio_path: str, device: str) -> tuple:
    """
    Single-sample inference.

    Returns:
        emotion_label (str), probabilities (Tensor [num_classes])
    """
    transform = default_transform()
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    mel = pad_or_crop(wav_to_mel(audio_path), TARGET_T).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, _ = model(img=img, mel=mel)

    probs = F.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return CLASS_NAMES[idx], probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True, help="Path to .pt checkpoint")
    p.add_argument("--proj_dim",   type=int, default=256)
    p.add_argument("--device",     default=None, help="cuda / cpu (auto-detect if omitted)")

    # Full evaluation mode
    p.add_argument("--img_test",   help="FER2013 test/ folder")
    p.add_argument("--audio_root", help="RAVDESS audio folder")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=2)
    p.add_argument("--bootstrap",  type=int, default=50, help="Bootstrap iterations (0 = skip)")

    # Single sample mode
    p.add_argument("--image", help="Path to a single face image")
    p.add_argument("--audio", help="Path to a single .wav file")

    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt, device, proj_dim=args.proj_dim)
    print(f"Loaded: {args.ckpt}")

    # ── Single sample inference ────────────────────────────────
    if args.image and args.audio:
        label, probs = predict(model, args.image, args.audio, device)
        print(f"\nPredicted emotion : {label}")
        print("Probabilities:")
        for name, prob in zip(CLASS_NAMES, probs.tolist()):
            bar = "█" * int(prob * 30)
            print(f"  {name:<10} {prob:.3f}  {bar}")
        return

    # ── Full evaluation ────────────────────────────────────────
    if not (args.img_test and args.audio_root):
        print("Provide --img_test + --audio_root for full eval, or --image + --audio for single sample.")
        return

    img_val_ds = FER2013Dataset(args.img_test)
    audio_by_label = scan_ravdess(args.audio_root)
    _, val_loader = build_loaders(
        img_val_ds, img_val_ds,     # train_ds unused
        audio_by_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    acc, f1, report, y_true, y_pred = evaluate(model, val_loader, device)
    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro : {f1:.4f}")
    print(report)

    if args.bootstrap > 0:
        stats = bootstrap_metrics(y_true, y_pred, n_bootstrap=args.bootstrap)
        print("=== Bootstrap Evaluation ===")
        for metric, (mean, std) in stats.items():
            print(f"{metric:<6}: {mean:.4f} ± {std:.4f}")

    plot_confusion(y_true, y_pred)


if __name__ == "__main__":
    main()
