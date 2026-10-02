"""
compare_all.py

Create one consolidated figure from the per-model results produced by the
shared pipeline.

Outputs:
  results/combined/<dataset>/<feature>/
    all_models_comparison.png
    combined_metrics.csv
    combined_roc.csv
    combined_pr.csv
    combined_loss.csv

The combined ROC/PR curves are mean +/- std over the 5 seeds. Metrics are
mean +/- std over the 5 seeds. Loss curves are mean +/- std over available
epochs. Confusion matrices are mean counts over the 5 seed test splits.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


DEFAULT_MODELS = {
    "GNN-SAGE": Path("results/gnn/{dataset}/{feature}/sage"),
    "GCNFN": Path("results/gcnfn/{dataset}/{feature}"),
    "BiGCN": Path("results/bigcn/{dataset}/{feature}"),
    "GNN-CL": Path("results/gnncl/{dataset}/{feature}"),
}


def read_csv(path):
    return pd.read_csv(path)


def load_model(root):
    root = Path(root)
    summary_path = root / "summary.csv"
    per_seed_path = root / "metrics_per_seed.csv"
    if not summary_path.exists() or not per_seed_path.exists():
        return None

    summary = read_csv(summary_path)
    per_seed = read_csv(per_seed_path)

    seed_dirs = []
    for d in sorted(root.glob("seed_*")):
        pred = d / "predictions.csv"
        hist = d / "history.csv"
        if pred.exists() and hist.exists():
            try:
                seed = int(d.name.split("_")[-1])
                seed_dirs.append((seed, d))
            except ValueError:
                pass

    return {
        "root": root,
        "summary": summary,
        "per_seed": per_seed,
        "seed_dirs": seed_dirs,
    }


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def compute_mean_roc(model, grid):
    curves, aucs = [], []
    for seed, seed_dir in model["seed_dirs"]:
        df = read_csv(seed_dir / "predictions.csv")
        y_true = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        curves.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(y_true, y_prob))

    if not curves:
        return None
    curves = np.vstack(curves)
    return {
        "fpr": grid,
        "tpr_mean": curves.mean(axis=0),
        "tpr_std": curves.std(axis=0),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def compute_mean_pr(model, grid):
    curves, aps = [], []
    for seed, seed_dir in model["seed_dirs"]:
        df = read_csv(seed_dir / "predictions.csv")
        y_true = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns recall in descending order in
        # common sklearn versions; sort before interpolation.
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        recall_unique, unique_idx = np.unique(recall_sorted, return_index=True)
        precision_unique = precision_sorted[unique_idx]
        curves.append(np.interp(grid, recall_unique, precision_unique))
        aps.append(float(np.trapezoid(precision, recall * -1) if recall[0] > recall[-1] else np.trapezoid(precision, recall)))

    if not curves:
        return None
    curves = np.vstack(curves)
    return {
        "recall": grid,
        "precision_mean": curves.mean(axis=0),
        "precision_std": curves.std(axis=0),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
    }


def compute_mean_loss(model):
    histories = []
    for seed, seed_dir in model["seed_dirs"]:
        df = read_csv(seed_dir / "history.csv")
        histories.append(df.set_index("epoch"))
    if not histories:
        return None

    max_epoch = max(int(h.index.max()) for h in histories)
    epochs = np.arange(1, max_epoch + 1)
    train = np.full((len(histories), max_epoch), np.nan)
    val = np.full((len(histories), max_epoch), np.nan)
    for i, h in enumerate(histories):
        for epoch in h.index:
            j = int(epoch) - 1
            train[i, j] = float(h.loc[epoch, "train_loss"])
            val[i, j] = float(h.loc[epoch, "val_loss"])

    return {
        "epoch": epochs,
        "train_mean": np.nanmean(train, axis=0),
        "train_std": np.nanstd(train, axis=0),
        "val_mean": np.nanmean(val, axis=0),
        "val_std": np.nanstd(val, axis=0),
    }


def compute_mean_cm(model):
    cms = []
    for seed, seed_dir in model["seed_dirs"]:
        df = read_csv(seed_dir / "predictions.csv")
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        cms.append(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    if not cms:
        return None
    return np.mean(np.stack(cms), axis=0)


def save_table(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    else:
        pd.DataFrame(data).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"])
    parser.add_argument("--feature", default="bert", choices=["profile", "spacy", "bert", "content"])
    parser.add_argument("--out_dir", default="results/combined")
    parser.add_argument(
        "--models", nargs="*",
        default=list(DEFAULT_MODELS.keys()),
        help="Model display names. Existing result folders are included automatically.",
    )
    args = parser.parse_args()

    loaded = {}
    for name in args.models:
        if name not in DEFAULT_MODELS:
            continue
        root = Path(str(DEFAULT_MODELS[name]).format(dataset=args.dataset, feature=args.feature))
        model = load_model(root)
        if model is not None:
            loaded[name] = model
        else:
            print(f"[WARN] Skip {name}: missing {root / 'summary.csv'} or per-seed artifacts.")

    if not loaded:
        raise FileNotFoundError(
            "No model results found. Run the model pipelines first. "
            "Each model must create summary.csv, metrics_per_seed.csv and seed_*/predictions.csv."
        )

    out_root = Path(args.out_dir) / args.dataset / args.feature
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------ 1) Metrics table ------------------
    rows = []
    metric_order = ["accuracy", "f1", "precision", "recall", "auc", "ap"]
    for name, model in loaded.items():
        row = {"model": name}
        for metric in metric_order:
            r = model["summary"].loc[model["summary"]["metric"] == metric]
            if len(r):
                row[f"{metric}_mean"] = float(r.iloc[0]["mean"])
                row[f"{metric}_std"] = float(r.iloc[0]["std"])
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    save_table(metrics_df, out_root / "combined_metrics.csv")

    # ------------------ Curves ------------------
    roc_grid = np.linspace(0.0, 1.0, 201)
    pr_grid = np.linspace(0.0, 1.0, 201)

    roc_data = {}
    pr_data = {}
    loss_data = {}
    cm_data = {}
    for name, model in loaded.items():
        roc_data[name] = compute_mean_roc(model, roc_grid)
        pr_data[name] = compute_mean_pr(model, pr_grid)
        loss_data[name] = compute_mean_loss(model)
        cm_data[name] = compute_mean_cm(model)

    # Save curve data for reproducibility.
    roc_rows, pr_rows, loss_rows = [], [], []
    for name, d in roc_data.items():
        if d:
            for x, m, s in zip(d["fpr"], d["tpr_mean"], d["tpr_std"]):
                roc_rows.append({"model": name, "fpr": x, "tpr_mean": m, "tpr_std": s})
    for name, d in pr_data.items():
        if d:
            for x, m, s in zip(d["recall"], d["precision_mean"], d["precision_std"]):
                pr_rows.append({"model": name, "recall": x, "precision_mean": m, "precision_std": s})
    for name, d in loss_data.items():
        if d:
            for e, tm, ts, vm, vs in zip(d["epoch"], d["train_mean"], d["train_std"], d["val_mean"], d["val_std"]):
                loss_rows.append({"model": name, "epoch": e, "train_mean": tm, "train_std": ts, "val_mean": vm, "val_std": vs})
    save_table(roc_rows, out_root / "combined_roc.csv")
    save_table(pr_rows, out_root / "combined_pr.csv")
    save_table(loss_rows, out_root / "combined_loss.csv")

    # ------------------ 2) One comprehensive figure ------------------
    n = len(loaded)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    ax_metrics, ax_roc, ax_pr = axes[0]
    ax_loss = axes[1, 0]
    cm_axes = [axes[1,1], axes[1,2], axes[2,0], axes[2,1], axes[2,2]]

    # Metrics grouped bar.
    x = np.arange(len(metric_order))
    width = 0.8 / max(n, 1)
    for i, row in metrics_df.iterrows():
        vals = [row[f"{m}_mean"] for m in metric_order]
        errs = [row[f"{m}_std"] for m in metric_order]
        offset = (i - (n - 1)/2) * width
        ax_metrics.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=row["model"])
    ax_metrics.set_xticks(x, [m.upper() if m != "f1" else "F1" for m in metric_order])
    ax_metrics.set_ylim(0, 1.15)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Model comparison: mean ± std over 5 seeds")
    ax_metrics.grid(axis="y", linestyle="--", alpha=.35)
    ax_metrics.legend(fontsize=8)

    # Combined ROC.
    for name, d in roc_data.items():
        if d:
            ax_roc.plot(d["fpr"], d["tpr_mean"], linewidth=2,
                        label=f"{name} (AUC={d['auc_mean']:.4f}±{d['auc_std']:.4f})")
            ax_roc.fill_between(d["fpr"], np.maximum(0, d["tpr_mean"]-d["tpr_std"]),
                                np.minimum(1, d["tpr_mean"]+d["tpr_std"]), alpha=.12)
    ax_roc.plot([0,1],[0,1], linestyle="--", linewidth=1.2, label="Random")
    ax_roc.set(xlim=(0,1), ylim=(0,1.05), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax_roc.set_title("Combined ROC (mean ± std across seeds)")
    ax_roc.grid(True, linestyle="--", alpha=.35)
    ax_roc.legend(fontsize=8, loc="lower right")

    # Combined PR.
    for name, d in pr_data.items():
        if d:
            ax_pr.plot(d["recall"], d["precision_mean"], linewidth=2,
                       label=f"{name} (AP≈{d['ap_mean']:.4f})")
            ax_pr.fill_between(d["recall"], np.maximum(0, d["precision_mean"]-d["precision_std"]),
                               np.minimum(1, d["precision_mean"]+d["precision_std"]), alpha=.12)
    ax_pr.set(xlim=(0,1), ylim=(0,1.05), xlabel="Recall", ylabel="Precision")
    ax_pr.set_title("Combined Precision–Recall")
    ax_pr.grid(True, linestyle="--", alpha=.35)
    ax_pr.legend(fontsize=8, loc="lower left")

    # Combined loss: 8 lines, train dashed / validation solid.
    for name, d in loss_data.items():
        if d:
            ax_loss.plot(d["epoch"], d["train_mean"], linestyle="--", linewidth=1.4, label=f"{name} Train")
            ax_loss.plot(d["epoch"], d["val_mean"], linewidth=2.0, label=f"{name} Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("NLL Loss")
    ax_loss.set_title("Mean loss curves across seeds")
    ax_loss.grid(True, linestyle="--", alpha=.35)
    ax_loss.legend(fontsize=7, ncol=2)

    # Confusion matrices: one axis per model, unused axes hidden.
    for ax in cm_axes:
        ax.axis("off")
    for ax, (name, cm) in zip(cm_axes, cm_data.items()):
        if cm is None:
            continue
        ax.axis("on")
        im = ax.imshow(cm)
        ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.04)
        ax.set_xticks([0,1], ["Real / 0", "Fake / 1"])
        ax.set_yticks([0,1], ["Real / 0", "Fake / 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\nMean confusion matrix over seeds")
        threshold = float(cm.max()) / 2 if cm.size else 0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r,c]:.1f}", ha="center", va="center",
                        color="white" if cm[r,c] > threshold else "black", fontweight="bold")

    fig.suptitle(
        f"Fake News Detection — {args.dataset.upper()} / {args.feature.upper()}",
        fontsize=18, fontweight="bold", y=.995,
    )
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(out_root / "all_models_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Also produce a dedicated ROC figure, matching the user's example.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d in roc_data.items():
        if d:
            ax.plot(d["fpr"], d["tpr_mean"], linewidth=2,
                    label=f"{name} (AUC={d['auc_mean']:.4f}±{d['auc_std']:.4f})")
            ax.fill_between(d["fpr"], np.maximum(0, d["tpr_mean"]-d["tpr_std"]),
                            np.minimum(1, d["tpr_mean"]+d["tpr_std"]), alpha=.12)
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1.2, label="Random")
    ax.set(xlim=(0,1), ylim=(0,1.05), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("All Models — ROC Comparison")
    ax.grid(True, linestyle="--", alpha=.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_root / "combined_roc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Dedicated metric figure.
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in metrics_df.iterrows():
        vals = [row[f"{m}_mean"] for m in metric_order]
        errs = [row[f"{m}_std"] for m in metric_order]
        ax.errorbar(metric_order, vals, yerr=errs, marker="o", linewidth=2, capsize=4, label=row["model"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("All Models — Mean ± Std")
    ax.grid(axis="y", linestyle="--", alpha=.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "combined_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 80)
    print("COMBINED RESULTS")
    print("=" * 80)
    display_cols = ["model"] + [f"{m}_mean" for m in metric_order]
    print(metrics_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
