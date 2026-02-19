"""Shared data loading, filtering, and PyG data object creation for GNN experiments."""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


GO_MIN_FREQ = 4
DATA_PATH = "../data/final_df_with_features_expanded.csv"


def load_and_filter(path=DATA_PATH, go_min_freq=GO_MIN_FREQ):
    """Load CSV and filter rare GO terms. Returns df, feature_cols, go_cols."""
    df = pd.read_csv(path)

    go_cols = [col for col in df.columns if col.startswith("GO:")]
    go_counts = df[go_cols].sum()
    keep_go = go_counts[go_counts >= go_min_freq].index.tolist()
    drop_go = go_counts[go_counts < go_min_freq].index.tolist()

    print(f"Total GO terms: {len(go_cols)}")
    print(f"Keeping (freq >= {go_min_freq}): {len(keep_go)}")
    print(f"Dropping (freq < {go_min_freq}): {len(drop_go)}")

    df = df.drop(columns=drop_go)

    remaining_go = [col for col in df.columns if col.startswith("GO:")]
    mask = df[remaining_go].sum(axis=1) > 0
    print(f"Rows before: {len(df)}, after removing zero-annotation rows: {mask.sum()}")
    df = df[mask].reset_index(drop=True)

    feature_cols = [col for col in df.columns if col not in ["sequence"] + remaining_go]
    return df, feature_cols, remaining_go


def get_esm_columns(df):
    """Return list of ESM embedding column names."""
    return [col for col in df.columns if col.startswith("ESM_Dim_")]


def build_pyg_data(df, feature_cols, go_cols, edge_index):
    """Build a PyG Data object from dataframe and edge index.

    Args:
        df: DataFrame with features and GO labels.
        feature_cols: List of feature column names for node features.
        go_cols: List of GO term column names for labels.
        edge_index: torch.LongTensor of shape [2, num_edges].

    Returns:
        PyG Data object with x, y, edge_index, and masks.
    """
    # Standardize node features
    scaler = StandardScaler()
    x = scaler.fit_transform(df[feature_cols].values)

    # Labels
    y = df[go_cols].values

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    print(f"PyG Data: {data}")
    return data


def create_masks(data, random_state=42):
    """Create stratified train/val/test masks (70/15/15) for multi-label node classification.

    Uses MultilabelStratifiedShuffleSplit for balanced label representation.
    """
    y_np = data.y.cpu().numpy()
    n = data.num_nodes
    indices = np.arange(n)

    # 70% train, 30% temp
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(indices.reshape(-1, 1), y_np))

    # 50/50 of temp → 15% val, 15% test
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel_idx, test_rel_idx = next(msss2.split(temp_idx.reshape(-1, 1), y_np[temp_idx]))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Masks: train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}")
    return data
