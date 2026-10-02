import argparse
from pathlib import Path
import copy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from src.utils.data_loader import *
from src.utils.pipeline_utils import (
    set_seed, make_split, train_one_epoch, predict_loader,
    save_csv, print_summary, save_seed_artifacts,
    plot_error_bar, plot_loss, plot_confusion_matrix, plot_roc,
)


class Net(nn.Module):
    """
    Source-faithful GCNFN implementation.

    Note: the uploaded source actually uses GATConv for both graph layers,
    even though the model/file is called GCNFN. This code preserves that
    architecture rather than silently replacing it with GCNConv.
    """
    def __init__(self, num_features, num_classes, nhid=128, concat=True):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.nhid = nhid
        self.concat = concat

        self.conv1 = GATConv(num_features, nhid * 2)
        self.conv2 = GATConv(nhid * 2, nhid * 2)

        self.fc0 = nn.Linear(num_features, nhid) if concat else None
        self.fc_graph = nn.Linear(nhid * 2, nhid)
        self.fc_concat = nn.Linear(nhid * 2, nhid)
        self.fc2 = nn.Linear(nhid, num_classes)

    def _root_features(self, data):
        # UPFD stores the source/news/root node first. Prefer root_index
        # when present; otherwise fall back to the first node in each graph.
        if hasattr(data, "root_index"):
            roots = data.root_index.view(-1).long()
            return data.x[roots]

        root_nodes = []
        batch = data.batch
        for graph_id in range(data.num_graphs):
            idx = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)[0]
            root_nodes.append(idx)
        return data.x[torch.stack(root_nodes)]

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc_graph(x))
        x = F.dropout(x, p=0.5, training=self.training)

        if self.concat:
            news = self._root_features(data)
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc_concat(x))

        return F.log_softmax(self.fc2(x), dim=-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"])
    p.add_argument("--feature", default="bert", choices=["profile", "spacy", "bert", "content"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--nhid", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 777, 101, 112])
    p.add_argument("--out_dir", default="results/gcnfn")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = FNNDataset(
        root="data",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=ToUndirected(),
    )

    print(args)
    print(f"Dataset size={len(dataset)}, features={dataset.num_features}, classes={dataset.num_classes}")
    print(f"Device={device}")

    out_root = Path(args.out_dir) / args.dataset / args.feature
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_plot_run = None

    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"GCNFN | SEED {seed}")
        print("=" * 70)

        set_seed(seed)
        train_set, val_set, test_set = make_split(dataset, seed)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = Net(
            dataset.num_features,
            dataset.num_classes,
            nhid=args.nhid,
            concat=args.concat,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_state = cp.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, args.epochs + 1):
            train_metrics, train_loss = train_one_epoch(
                model, train_loader, optimizer, device
            )
            val_metrics, val_loss, _, _ = predict_loader(
                model, val_loader, device
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['f1']:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0
                best_state = cp.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                    break

        model.load_state_dict(best_state)

        test_metrics, test_loss, y_true, y_prob = predict_loader(
            model, test_loader, device
        )

        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            **test_metrics,
        }
        all_results.append(result)

        save_seed_artifacts(out_root / f"seed_{seed}", history, y_true, y_prob)

        print(
            f"Seed {seed} | "
            f"acc={test_metrics['accuracy']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, "
            f"auc={test_metrics['auc']:.4f}, "
            f"ap={test_metrics['ap']:.4f}"
        )

        if best_plot_run is None or best_val_loss < best_plot_run["best_val_loss"]:
            best_plot_run = {
                "seed": seed,
                "best_val_loss": best_val_loss,
                "history": history,
                "y_true": y_true,
                "y_prob": y_prob,
            }

    summary = print_summary("GCNFN", args.dataset, all_results)

    save_csv(
        out_root / "metrics_per_seed.csv",
        all_results,
        ["seed", "best_epoch", "best_val_loss", "test_loss"] + list(summary.keys()),
    )
    save_csv(
        out_root / "summary.csv",
        [
            {"metric": m, "mean": summary[m]["mean"], "std": summary[m]["std"]}
            for m in summary
        ],
        ["metric", "mean", "std"],
    )

    plot_error_bar(summary, "GCNFN", out_root)
    plot_loss(best_plot_run["history"], "GCNFN", out_root, best_plot_run["seed"])
    plot_confusion_matrix(
        best_plot_run["y_true"], best_plot_run["y_prob"], "GCNFN", out_root
    )
    plot_roc(best_plot_run["y_true"], best_plot_run["y_prob"], "GCNFN", out_root)

    print(f"\nResults saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
