import argparse
from pathlib import Path
import copy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from src.utils.data_loader import *
from src.utils.pipeline_utils import (
    set_seed, make_split, train_one_epoch, predict_loader,
    save_csv, print_summary, save_seed_artifacts,
    plot_error_bar, plot_loss, plot_confusion_matrix, plot_roc,
)


def _expand_root_feature(root_index, batch, x_feature):
    """
    Broadcast each graph's root representation to all its nodes without the
    Python loop used in the uploaded reference implementation.
    """
    num_graphs = int(batch.max().item()) + 1
    # root_index is expected to contain one root node index per graph.
    roots = root_index.view(-1).long()
    root_repr = x_feature[roots]
    return root_repr[batch]


class TDrumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x1 = x

        x = self.conv1(x, edge_index)
        x2 = x

        root_extend = _expand_root_feature(data.root_index, data.batch, x1)
        x = torch.cat((x, root_extend), dim=1)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = _expand_root_feature(data.root_index, data.batch, x2)
        x = torch.cat((x, root_extend), dim=1)

        return scatter_mean(x, data.batch, dim=0)


class BUrumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.float(), data.BU_edge_index
        x1 = x

        x = self.conv1(x, edge_index)
        x2 = x

        root_extend = _expand_root_feature(data.root_index, data.batch, x1)
        x = torch.cat((x, root_extend), dim=1)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = _expand_root_feature(data.root_index, data.batch, x2)
        x = torch.cat((x, root_extend), dim=1)

        return scatter_mean(x, data.batch, dim=0)


class Net(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, td_dropout=0.2, bu_dropout=0.2):
        super().__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, td_dropout)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, bu_dropout)
        self.fc = nn.Linear((out_feats + hid_feats) * 2, 2)

    def forward(self, data):
        td_x = self.TDrumorGCN(data)
        bu_x = self.BUrumorGCN(data)
        x = torch.cat((td_x, bu_x), dim=1)
        return F.log_softmax(self.fc(x), dim=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"])
    p.add_argument("--feature", default="bert", choices=["profile", "spacy", "bert", "content"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--nhid", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--TDdroprate", type=float, default=0.2)
    p.add_argument("--BUdroprate", type=float, default=0.2)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 777, 101, 112])
    p.add_argument("--out_dir", default="results/bigcn")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # The source BiGCN uses DropEdge(TDdroprate, BUdroprate).
    # Keep this transform as part of the dataset so the experiment preserves
    # the intended bidirectional-dropedge setup.
    dataset = FNNDataset(
        root="data/",
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=DropEdge(args.TDdroprate, args.BUdroprate),
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
        print(f"BiGCN | SEED {seed}")
        print("=" * 70)

        set_seed(seed)
        train_set, val_set, test_set = make_split(dataset, seed)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = Net(
            dataset.num_features,
            args.nhid,
            args.nhid,
            td_dropout=args.TDdroprate,
            bu_dropout=args.BUdroprate,
        ).to(device)

        # Preserve the source's lower learning rate for the BU branch.
        bu_param_ids = {
            id(p)
            for p in list(model.BUrumorGCN.conv1.parameters())
            + list(model.BUrumorGCN.conv2.parameters())
        }
        base_params = [
            p for p in model.parameters()
            if id(p) not in bu_param_ids
        ]
        bu_params = list(model.BUrumorGCN.conv1.parameters()) + list(
            model.BUrumorGCN.conv2.parameters()
        )

        optimizer = torch.optim.Adam(
            [
                {"params": base_params},
                {"params": bu_params, "lr": args.lr / 5.0},
            ],
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

    summary = print_summary("BiGCN", args.dataset, all_results)

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

    plot_error_bar(summary, "BiGCN", out_root)
    plot_loss(best_plot_run["history"], "BiGCN", out_root, best_plot_run["seed"])
    plot_confusion_matrix(
        best_plot_run["y_true"], best_plot_run["y_prob"], "BiGCN", out_root
    )
    plot_roc(best_plot_run["y_true"], best_plot_run["y_prob"], "BiGCN", out_root)

    print(f"\nResults saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
