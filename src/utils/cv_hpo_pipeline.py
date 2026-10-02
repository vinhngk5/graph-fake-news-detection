import json
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader, DenseDataLoader

from src.models.bigcn import Net as BiGCN
from src.models.gcnfn import Net as GCNFN
from src.models.gnn import Model as GNN
from src.models.gnncl import make_model as GNNCL
from src.utils.data_loader import *  # noqa: F401,F403
from src.utils.pipeline_utils import predict_loader, set_seed, train_one_epoch

# ============================================================
# Global settings
# ============================================================

SEED = 42
EPOCH = 200
PATIENCE = 10
N_SPLITS = 5

# Metric used to select the best hyperparameters.
SELECTION_METRIC = "f1"


# ============================================================
# Helpers
# ============================================================

def to_numpy_indices(indices):
    """Convert torch/list/np indices to a 1-D NumPy array."""
    if torch.is_tensor(indices):
        return indices.detach().cpu().numpy()
    return np.asarray(indices)


def prepare_cv_datasets(dataset):
    """
    Keep the official UPFD test split untouched.

    Original:  train / val / test
    New:       development = train + val, test = original test

    The graphs themselves remain independent.
    """
    train_idx = to_numpy_indices(dataset.train_idx)
    val_idx = to_numpy_indices(dataset.val_idx)
    test_idx = to_numpy_indices(dataset.test_idx)

    # Official train + validation -> development
    dev_global_indices = np.concatenate([train_idx, val_idx]).astype(np.int64)
    test_global_indices = test_idx.astype(np.int64)

    dev_dataset = Subset(dataset, dev_global_indices.tolist())
    test_dataset = Subset(dataset, test_global_indices.tolist())

    # Labels for StratifiedKFold
    dev_labels = np.asarray(
        [int(dataset.get(int(idx)).y.item()) for idx in dev_global_indices]
    )

    return (
        dev_dataset,
        test_dataset,
        dev_labels,
        dev_global_indices,
        test_global_indices,
    )


def get_device(args):
    return torch.device(args.device if torch.cuda.is_available() else "cpu")


def get_loader_cls(model_name):
    return DenseDataLoader if model_name == "gnncl" else DataLoader


def print_dataset_info(dataset, dev_dataset, test_dataset, dev_labels, args, device):
    """Print dataset information."""
    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Full dataset   : {len(dataset)}")
    print(f"Development    : {len(dev_dataset)}")
    print(f"Test           : {len(test_dataset)}")
    print(f"Features       : {dataset.num_features}")
    print(f"Classes        : {dataset.num_classes}")
    print(f"Device         : {device}")
    print(
        f"Development labels: "
        f"class 0 = {(dev_labels == 0).sum()}, "
        f"class 1 = {(dev_labels == 1).sum()}"
    )
    print("=" * 70)
    print(args)


def apply_params_to_args(args, params):
    """
    Create a new args object and override only the hyperparameters
    used by the current experiment.

    Supports: wd -> weight_decay
    """
    new_args = deepcopy(args)

    for key, value in params.items():
        setattr(new_args, "weight_decay" if key == "wd" else key, value)

    return new_args


def get_metric(metrics, name="f1"):
    """Get a metric robustly from the metric dictionary."""
    aliases = {
        "f1": ["f1", "macro_f1", "F1", "Macro-F1"],
        "accuracy": ["accuracy", "acc", "Accuracy"],
        "precision": ["precision", "Precision"],
        "recall": ["recall", "Recall"],
    }

    for key in aliases.get(name, [name]):
        if key in metrics:
            return float(metrics[key])

    raise KeyError(
        f"Cannot find metric '{name}'. "
        f"Available metrics: {list(metrics.keys())}"
    )


def generate_param_combinations(param_grid):
    """
    Yield one hyperparameter configuration (dict) at a time.

    Example: {"nhid": 128, "lr": 0.001, "wd": 0.0001}
    """
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]

    for combination in product(*values):
        yield dict(zip(keys, combination))


# ============================================================
# Cache (skip configurations that were already run)
# ============================================================

def _json_default(obj):
    """Convert NumPy scalars/arrays so json can serialize them."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def params_key(params):
    """Stable string key identifying one hyperparameter configuration."""
    return json.dumps(params, sort_keys=True, default=_json_default)


def load_cache(cache_path, meta):
    """Load finished configurations. Ignore records made with different settings."""
    cache = {}
    if not cache_path.exists():
        return cache

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # line half-written when the run was interrupted
            if record.get("meta") != meta:
                continue
            cache[params_key(record["params"])] = record["result"]

    return cache


def append_cache(cache_path, meta, params, result):
    """Append one finished configuration to the cache file."""
    with open(cache_path, "a", encoding="utf-8") as f:
        record = {"meta": meta, "params": params, "result": result}
        f.write(json.dumps(record, default=_json_default) + "\n")


# ============================================================
# Dataset setup
# ============================================================

def _finalize_dataset(dataset, args, device):
    """Shared tail of every setup_*_dataset function."""
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes

    dev_dataset, test_dataset, dev_labels, _, _ = prepare_cv_datasets(dataset)

    print_dataset_info(dataset, dev_dataset, test_dataset, dev_labels, args, device)

    return dataset, dev_dataset, test_dataset, dev_labels, args


def setup_BiGCN_dataset(args):
    device = get_device(args)

    dataset = FNNDataset(
        root="data/",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=DropEdge(args.TDdroprate, args.BUdroprate),
    )

    return _finalize_dataset(dataset, args, device)


def setup_GCNFN_dataset(args):
    device = get_device(args)

    dataset = FNNDataset(
        root="data",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=ToUndirected(),
    )

    return _finalize_dataset(dataset, args, device)


def setup_GNN_dataset(args):
    device = get_device(args)

    dataset = FNNDataset(
        root="data",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=ToUndirected(),
    )

    return _finalize_dataset(dataset, args, device)


def setup_GNNCL_dataset(args):
    device = get_device(args)

    max_nodes = args.max_nodes or (500 if args.dataset == "politifact" else 200)

    dataset = FNNDataset(
        root="data",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=T.ToDense(max_nodes),
        pre_transform=ToUndirected(),
    )

    return _finalize_dataset(dataset, args, device)


# ============================================================
# Model setup
# ============================================================

def setup_BiGCN_model(args, dataset, device):
    return BiGCN(
        dataset.num_features,
        args.nhid,
        args.nhid,
        td_dropout=args.TDdroprate,
        bu_dropout=args.BUdroprate,
    ).to(device)


def setup_GCNFN_model(args, dataset, device):
    return GCNFN(
        dataset.num_features,
        dataset.num_classes,
        nhid=args.nhid,
        concat=args.concat,
    ).to(device)


def setup_GNN_model(args, dataset, device):
    return GNN(args).to(device)


def setup_GNNCL_model(args, dataset, device):
    return GNNCL(args, dataset).to(device)


# ============================================================
# Registries
# ============================================================

SETUP_DATASET_REGISTRY = {
    "bigcn": setup_BiGCN_dataset,
    "gcnfn": setup_GCNFN_dataset,
    "gnn": setup_GNN_dataset,
    "gnncl": setup_GNNCL_dataset,
}

SETUP_MODEL_REGISTRY = {
    "bigcn": setup_BiGCN_model,
    "gcnfn": setup_GCNFN_model,
    "gnn": setup_GNN_model,
    "gnncl": setup_GNNCL_model,
}


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(model_name, model, args):
    """Create the optimizer according to the original model setup."""
    if model_name == "bigcn":
        bu_params = list(model.BUrumorGCN.conv1.parameters()) + list(
            model.BUrumorGCN.conv2.parameters()
        )
        bu_param_ids = {id(p) for p in bu_params}
        base_params = [p for p in model.parameters() if id(p) not in bu_param_ids]

        return torch.optim.Adam(
            [
                {"params": base_params},
                {"params": bu_params, "lr": args.lr / 5.0},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    return torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(
    model_name,
    full_dataset,
    fold_train_dataset,
    fold_val_dataset,
    args,
    device,
    fold=None,
):
    """
    Train ONE fresh model for ONE CV fold.

    The best checkpoint is selected by minimum validation loss.
    """
    set_seed(SEED)

    model = SETUP_MODEL_REGISTRY[model_name](args, full_dataset, device)

    loader_cls = get_loader_cls(model_name)

    num_workers = getattr(args, "num_workers", 0)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_loader = loader_cls(fold_train_dataset, shuffle=True, **loader_kwargs)
    val_loader = loader_cls(fold_val_dataset, shuffle=False, **loader_kwargs)
    optimizer = create_optimizer(model_name, model, args)

    # Early stopping / best checkpoint
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    # Đặt trước vòng lặp epoch
    forward_fn = (lambda d: model(d)[0]) if model_name == "gnncl" else None

    for epoch in range(1, EPOCH + 1):
        train_metrics, train_loss = train_one_epoch(
            model, train_loader, optimizer, device, forward_fn=forward_fn
        )

        val_metrics, val_loss, _, _ = predict_loader(
            model, val_loader, device, forward_fn=forward_fn
        )
        val_f1 = get_metric(val_metrics, "f1")
        val_acc = get_metric(val_metrics, "accuracy")

        # Select best checkpoint by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_f1": best_val_f1,
        "val_accuracy": best_val_acc,
    }


# ============================================================
# Hyperparameter search
# ============================================================

def find_best_params(model_name, args, param_grid):
    # Device / reproducibility
    set_seed(SEED)
    device = get_device(args)

    # Load dataset once
    (
        full_dataset,
        dev_dataset,
        test_dataset,
        dev_labels,
        args,
    ) = SETUP_DATASET_REGISTRY[model_name](args)

    # Cross-validation
    X = np.arange(len(dev_dataset))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Generate configurations
    param_combinations = list(generate_param_combinations(param_grid))
    n_configs = len(param_combinations)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Model          : {model_name}")
    print(f"Configurations : {n_configs}")
    print(f"CV folds       : {N_SPLITS}")

    # Paths + cache
    out_dir = Path("results") / f"{model_name}/{args.dataset}/{args.feature}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{model_name}_hpo_cache.jsonl"
    results_path = out_dir / f"{model_name}_best.csv"

    # Settings that affect results; cache is only reused if they match
    meta = {
        "seed": SEED,
        "epoch": EPOCH,
        "patience": PATIENCE,
        "n_splits": N_SPLITS,
    }
    cache = load_cache(cache_path, meta)
    print(f"Cached configs : {len(cache)}")

    # Search
    all_results = []
    best_mean_f1 = -float("inf")
    best_params = None
    best_result = None

    for config_id, params in enumerate(param_combinations, start=1):
        print("\n" + "-" * 70)
        print(f"Configuration {config_id}/{n_configs}")
        print(params)

        key = params_key(params)

        if key in cache:
            # Already evaluated -> skip training, reuse stored result
            result = cache[key]
            print(
                f"SKIPPED (already run): "
                f"mean_f1={result['mean_f1']:.4f} +/- {result['std_f1']:.4f}"
            )
        else:
            fold_f1_scores = []
            fold_acc_scores = []
            fold_val_losses = []
            fold_best_epochs = []

            # Original args + current hyperparameters
            fold_args = apply_params_to_args(args, params)

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, dev_labels), start=1):
                fold_train_dataset = Subset(dev_dataset, train_idx)
                fold_val_dataset = Subset(dev_dataset, val_idx)

                print(
                    f"\nFold {fold}/{N_SPLITS}: "
                    f"train={len(train_idx)}, val={len(val_idx)}"
                )

                fold_result = train_one_fold(
                    model_name=model_name,
                    full_dataset=full_dataset,
                    fold_train_dataset=fold_train_dataset,
                    fold_val_dataset=fold_val_dataset,
                    args=fold_args,
                    device=device,
                    fold=fold,
                )

                fold_f1_scores.append(fold_result["val_f1"])
                fold_acc_scores.append(fold_result["val_accuracy"])
                fold_val_losses.append(fold_result["val_loss"])
                fold_best_epochs.append(fold_result["best_epoch"])

                print(f"  val_loss   = {fold_result['val_loss']:.4f}")
                print(f"  val_f1     = {fold_result['val_f1']:.4f}")
                print(f"  val_acc    = {fold_result['val_accuracy']:.4f}")
                print(f"  best_epoch = {fold_result['best_epoch']}")

            # Aggregate CV results
            result = {
                **params,
                "mean_f1": float(np.mean(fold_f1_scores)),
                "std_f1": float(np.std(fold_f1_scores)),
                "mean_accuracy": float(np.mean(fold_acc_scores)),
                "std_accuracy": float(np.std(fold_acc_scores)),
                "mean_val_loss": float(np.mean(fold_val_losses)),
                "median_best_epoch": int(np.median(fold_best_epochs)),
                "fold_f1": fold_f1_scores,
                "fold_accuracy": fold_acc_scores,
                "fold_val_loss": fold_val_losses,
                "fold_best_epoch": fold_best_epochs,
            }

            # Save immediately so an interrupted run can resume
            append_cache(cache_path, meta, params, result)
            cache[key] = result

            print(f"\nMean F1       = {result['mean_f1']:.4f} +/- {result['std_f1']:.4f}")
            print(f"Mean Accuracy = {result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}")
            print(f"Mean Val Loss = {result['mean_val_loss']:.4f}")

        all_results.append(result)

        # HPO criterion = maximum mean CV F1 (applies to cached results too)
        if result["mean_f1"] > best_mean_f1:
            best_mean_f1 = result["mean_f1"]
            best_params = params.copy()
            best_result = result
            print("\n*** NEW BEST CONFIGURATION ***")

    # Save ONLY the best result
    best_df = pd.DataFrame([best_result])
    best_df.to_csv(results_path, index=False)

    # Final HPO result
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print(best_params)
    print(f"Mean CV F1       = {best_result['mean_f1']:.4f}")
    print(f"Std CV F1        = {best_result['std_f1']:.4f}")
    print(f"Mean CV Accuracy = {best_result['mean_accuracy']:.4f}")
    print(f"Mean CV Loss     = {best_result['mean_val_loss']:.4f}")
    print(f"Results saved to: {results_path}")

    return {
        "best_params": best_params,
        "best_result": best_result,
        "results": best_df,
        "full_dataset": full_dataset,
        "dev_dataset": dev_dataset,
        "test_dataset": test_dataset,
        "dev_labels": dev_labels,
        "args": args,
    }