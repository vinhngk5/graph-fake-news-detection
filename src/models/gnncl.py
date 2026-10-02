import argparse
from pathlib import Path
import copy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from src.utils.data_loader import *
from src.utils.pipeline_utils import (
    set_seed, make_split, train_one_epoch, predict_loader,
    save_csv, summarize_runs, print_summary, save_seed_artifacts,
    plot_error_bar, plot_loss, plot_confusion_matrix, plot_roc,
)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.lin = (
            nn.Linear(2 * hidden_channels + out_channels, out_channels)
            if lin else None
        )

    def _bn(self, layer_no, x):
        b, n, c = x.size()
        x = x.reshape(-1, c)
        x = getattr(self, f"bn{layer_no}")(x)
        return x.reshape(b, n, c)

    def forward(self, x, adj, mask=None):
        x1 = self._bn(1, F.relu(self.conv1(x, adj, mask)))
        x2 = self._bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self._bn(3, F.relu(self.conv3(x2, adj, mask)))
        x = torch.cat([x1, x2, x3], dim=-1)
        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, max_nodes):
        super().__init__()

        num_nodes = int(torch.ceil(torch.tensor(0.25 * max_nodes)).item())
        self.gnn1_pool = GNN(in_channels, hidden_channels, num_nodes)
        self.gnn1_embed = GNN(in_channels, hidden_channels, hidden_channels, lin=False)

        num_nodes = int(torch.ceil(torch.tensor(0.25 * num_nodes)).item())
        self.gnn2_pool = GNN(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)
        self.lin1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


def make_model(args, dataset):
    max_nodes = args.max_nodes
    if max_nodes is None:
        max_nodes = 500 if args.dataset == "politifact" else 200
    return Net(dataset.num_features, args.nhid, dataset.num_classes, max_nodes)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"])
    p.add_argument("--feature", default="bert", choices=["profile", "spacy", "bert", "content"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--nhid", type=int, default=64,
                   help="Kept for CLI compatibility; source architecture uses 64.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 777, 101, 112])
    p.add_argument("--out_dir", default="results/gnncl")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    max_nodes = args.max_nodes or (500 if args.dataset == "politifact" else 200)

    # The source GNN-CL uses ToDense + ToUndirected.
    dataset = FNNDataset(
        root="data",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=T.ToDense(max_nodes),
        pre_transform=ToUndirected(),
    )

    print(args)
    print(f"Dataset size={len(dataset)}, features={dataset.num_features}, classes={dataset.num_classes}")
    print(f"Device={device}, max_nodes={max_nodes}")

    out_root = Path(args.out_dir) / args.dataset / args.feature
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_plot_run = None

    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"GNN-CL | SEED {seed}")
        print("=" * 70)

        set_seed(seed)
        train_set, val_set, test_set = make_split(dataset, seed)

        train_loader = DenseDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DenseDataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DenseDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = make_model(args, dataset).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_state = cp.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, args.epochs + 1):
            train_metrics, train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                forward_fn=lambda d: model(d)[0],
            )
            val_metrics, val_loss, _, _ = predict_loader(
                model,
                val_loader,
                device,
                forward_fn=lambda d: model(d)[0],
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
            model,
            test_loader,
            device,
            forward_fn=lambda d: model(d)[0],
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

        # Choose the run with lowest validation loss for auxiliary plots.
        if best_plot_run is None or best_val_loss < best_plot_run["best_val_loss"]:
            best_plot_run = {
                "seed": seed,
                "best_val_loss": best_val_loss,
                "history": history,
                "y_true": y_true,
                "y_prob": y_prob,
            }

    summary = print_summary("GNN-CL", args.dataset, all_results)
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

    plot_error_bar(summary, "GNN-CL", out_root)
    plot_loss(best_plot_run["history"], "GNN-CL", out_root, best_plot_run["seed"])
    plot_confusion_matrix(
        best_plot_run["y_true"], best_plot_run["y_prob"], "GNN-CL", out_root
    )
    plot_roc(best_plot_run["y_true"], best_plot_run["y_prob"], "GNN-CL", out_root)

    print(f"\nResults saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
