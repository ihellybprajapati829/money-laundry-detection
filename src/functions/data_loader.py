import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_raw(raw_dir=RAW_DIR):
    """Load the three Elliptic files and return features, classes, edges."""
    raw = pd.read_csv(f"{raw_dir}/elliptic_txs_features.csv", header=None)
    num_features = raw.shape[1] - 2  # minus txId and time_step
    cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, num_features + 1)]
    raw.columns = cols
    classes = pd.read_csv(f"{raw_dir}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{raw_dir}/elliptic_txs_edgelist.csv")
    return raw, classes, edges

def get_labeled(features, classes):
    """Merge features with labels, keep labeled rows, add binary label (1 = illicit)."""
    df = features.merge(classes, on="txId", how="left")
    df = df[df["class"] != "unknown"].copy()
    df["label"] = (df["class"] == "1").astype(int)
    return df