import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def evaluate(model, loader, device) -> tuple:
    """
    Run inference on a DataLoader and return metrics.

    Returns:
        acc, f1_macro, report (str), y_true (list), y_pred (list)
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, auds, labels in loader:
            imgs, auds = imgs.to(device), auds.to(device)
            logits, _, _ = model(img=imgs, mel=auds)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    return acc, f1, report, y_true, y_pred


def bootstrap_metrics(
    y_true,
    y_pred,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict:
    """
    Estimate mean ± std of metrics via bootstrap resampling.

    Returns:
        dict with keys: acc, prec, rec, f1
        each value is (mean, std)
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    res = {"acc": [], "prec": [], "rec": [], "f1": []}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        res["acc"].append(accuracy_score(yt, yp))
        res["prec"].append(precision_score(yt, yp, average="macro", zero_division=0))
        res["rec"].append(recall_score(yt, yp, average="macro", zero_division=0))
        res["f1"].append(f1_score(yt, yp, average="macro", zero_division=0))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}


def plot_confusion(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
