# pipeline_utils.py
# Shared experiment protocol and plotting helpers for all fake-news GNN models.

import csv
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from src.utils.data_loader import *
except ImportError:
    from utils.data_loader import *

METRIC_NAMES = ["accuracy", "f1", "precision", "recall", "auc", "ap"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_split(dataset, seed: int):
    from torch.utils.data import random_split

    generator = torch.Generator().manual_seed(seed)
    n_train = int(len(dataset) * 0.20)
    n_val = int(len(dataset) * 0.10)
    n_test = len(dataset) - n_train - n_val
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )


def _safe_auc_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = np.nan
    return auc, ap


def metrics_from_probs(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    auc, ap = _safe_auc_ap(y_true, y_prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "auc": auc,
        "ap": ap,
    }


@torch.no_grad()
def predict_loader(model, loader, device, forward_fn=None):
    model.eval()
    y_true_all, y_prob_all = [], []
    total_loss, total_samples = 0.0, 0

    for data in loader:
        data = data.to(device)
        out = forward_fn(data) if forward_fn is not None else model(data)
        y = data.y.view(-1).long()
        loss = torch.nn.functional.nll_loss(out, y, reduction="sum")
        prob = torch.softmax(out, dim=1)[:, 1]

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_prob_all.extend(prob.detach().cpu().numpy().tolist())
        total_loss += float(loss.item())
        total_samples += int(y.numel())

    avg_loss = total_loss / max(total_samples, 1)
    metrics = metrics_from_probs(y_true_all, y_prob_all)
    return metrics, avg_loss, y_true_all, y_prob_all


def train_one_epoch(model, loader, optimizer, device, forward_fn=None):
    model.train()
    total_loss, total_samples = 0.0, 0
    y_true_all, y_prob_all = [], []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = forward_fn(data) if forward_fn is not None else model(data)
        y = data.y.view(-1).long()
        loss = torch.nn.functional.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        n = int(y.numel())
        total_loss += float(loss.item()) * n
        total_samples += n
        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_prob_all.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    return metrics_from_probs(y_true_all, y_prob_all), avg_loss


def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_seed_artifacts(seed_dir, history, y_true, y_prob):
    """Save raw per-seed artifacts so compare_all.py can redraw combined plots."""
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        seed_dir / "predictions.csv",
        [
            {"y_true": int(y), "y_prob": float(p), "y_pred": int(p >= 0.5)}
            for y, p in zip(y_true, y_prob)
        ],
        ["y_true", "y_prob", "y_pred"],
    )
    history_rows = []
    n = max(len(history.get("train_loss", [])), len(history.get("val_loss", [])))
    for i in range(n):
        history_rows.append(
            {
                "epoch": i + 1,
                "train_loss": history.get("train_loss", [np.nan] * n)[i],
                "val_loss": history.get("val_loss", [np.nan] * n)[i],
            }
        )
    save_csv(seed_dir / "history.csv", history_rows, ["epoch", "train_loss", "val_loss"])


def summarize_runs(results):
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray([r[metric] for r in results], dtype=np.float64)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return summary


def print_summary(model_name, dataset_name, results):
    summary = summarize_runs(results)
    print("\n" + "=" * 70)
    print(f"{model_name} | DATASET={dataset_name}")
    print(f"MEAN ± STD OVER {len(results)} SEEDS")
    print("=" * 70)
    labels = {
        "accuracy": "Accuracy ", "f1": "F1       ", "precision": "Precision",
        "recall": "Recall   ", "auc": "AUC      ", "ap": "AP       ",
    }
    for m in METRIC_NAMES:
        print(f"{labels[m]}: {summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}")
    print("=" * 70)
    return summary


def plot_error_bar(summary, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [("Accuracy", "accuracy"), ("F1", "f1"), ("Precision", "precision"),
             ("Recall", "recall"), ("AUC", "auc"), ("AP", "ap")]
    names = [x[0] for x in pairs]
    means = [summary[x[1]]["mean"] for x in pairs]
    stds = [summary[x[1]]["std"] for x in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6, edgecolor="black")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, min(mean + std + .015, 1.06),
                f"{mean:.4f} ± {std:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name}: 5-Seed Performance")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.tight_layout()
    fig.savefig(out_dir / "error_bar_all_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history, model_name, out_dir, seed):
    out_dir = Path(out_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title(f"{model_name}: Loss Curve (Seed {seed})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=.4)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_prob, model_name, out_dir):
    out_dir = Path(out_dir)
    pred = (np.asarray(y_prob) >= .5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Real / 0","Fake / 1"],
           yticklabels=["Real / 0","Fake / 1"], xlabel="Predicted Label",
           ylabel="True Label", title=f"{model_name}: Confusion Matrix")
    threshold = cm.max()/2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > threshold else "black", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_prob, model_name, out_dir):
    out_dir = Path(out_dir)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(5,5))
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1.5, label="Random")
    ax.set(xlim=(0,1), ylim=(0,1.05), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title=f"{model_name}: ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=.4)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
