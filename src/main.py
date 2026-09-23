import argparse
from pathlib import Path

from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="politifact",
                    choices=["politifact", "gossipcop"])
parser.add_argument("--feature", type=str, default="spacy",
                    choices=["profile", "spacy", "bert", "content"])

args = parser.parse_args()
data_root = Path(__file__).parent.parent / "data"

train_dataset = UPFD(data_root, args.dataset, args.feature, "train", ToUndirected())
val_dataset = UPFD(data_root, args.dataset, args.feature, "val", ToUndirected())
test_dataset = UPFD(data_root, args.dataset, args.feature, "test", ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)