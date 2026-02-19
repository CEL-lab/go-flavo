"""Feature ablation study for XGBoost and DNN.

Addresses reviewer R4.5:
  - Report XGBoost ablation results (reviewer said they were "dismissed without justification")
  - Compare three feature sets: All features, ESM-2 only, BioPython only
  - Uses the same 70/15/15 stratified split as DNN, GNN, and per-GO analysis

Usage:
  python 6_ablation.py                          # full ablation (XGBoost + DNN)
  python 6_ablation.py --skip-xgboost           # DNN only (faster)
  python 6_ablation.py --skip-dnn               # XGBoost only
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)


# ── Data loading (same as 5_per_go_analysis.py and DNN/GNN scripts) ─────────

def load_and_filter(path="./data/final_df_with_features_expanded.csv", go_min_freq=4):
    """Load CSV, filter rare GO terms."""
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

    return df, remaining_go


def get_feature_sets(df, go_cols):
    """Define the three feature set configurations."""
    all_features = [c for c in df.columns if c not in ["sequence"] + go_cols]
    esm_features = [c for c in all_features if c.startswith("ESM_Dim_")]
    bio_features = [c for c in all_features if not c.startswith("ESM_Dim_")]

    print(f"Feature sets: All={len(all_features)}, ESM={len(esm_features)}, BioPython={len(bio_features)}")
    return {
        "All features": all_features,
        "ESM-2 only": esm_features,
        "BioPython only": bio_features,
    }


def create_splits(df, feature_cols, go_cols, random_state=42):
    """70/15/15 stratified multi-label split (same as DNN/GNN/per-GO analysis)."""
    X = df[feature_cols].values
    y = df[go_cols].values

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(X, y))

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    return train_idx, val_idx, test_idx


# ── XGBoost ablation ────────────────────────────────────────────────────────

def run_xgboost_ablation(df, feature_sets, go_cols, train_idx, test_idx):
    """Train XGBoost with each feature set and return results."""
    from xgboost import XGBClassifier
    from tqdm import tqdm

    y_train = df[go_cols].values[train_idx]
    y_test = df[go_cols].values[test_idx]

    results = []
    for name, feat_cols in feature_sets.items():
        print(f"\n--- XGBoost: {name} ({len(feat_cols)} features) ---")
        X_train = df[feat_cols].values[train_idx]
        X_test = df[feat_cols].values[test_idx]

        start = time.time()
        y_pred_test = np.zeros_like(y_test)
        y_pred_train = np.zeros_like(y_train)

        for i in range(len(go_cols)):
            pos_count = y_train[:, i].sum()
            if pos_count == 0:
                clf = XGBClassifier(eval_metric="logloss", random_state=42, base_score=1e-5, verbosity=0)
            else:
                clf = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
            clf.fit(X_train, y_train[:, i])
            y_pred_test[:, i] = clf.predict(X_test)
            y_pred_train[:, i] = clf.predict(X_train)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(go_cols)} GO terms done")

        elapsed = time.time() - start

        report_test = classification_report(y_test, y_pred_test, target_names=go_cols,
                                            zero_division=0, output_dict=True)
        report_train = classification_report(y_train, y_pred_train, target_names=go_cols,
                                             zero_division=0, output_dict=True)

        result = {
            "model": "XGBoost",
            "features": name,
            "num_features": len(feat_cols),
            "time_s": round(elapsed, 1),
            "micro_precision": round(report_test["micro avg"]["precision"], 4),
            "micro_recall": round(report_test["micro avg"]["recall"], 4),
            "micro_f1": round(report_test["micro avg"]["f1-score"], 4),
            "macro_f1": round(report_test["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_test["weighted avg"]["f1-score"], 4),
            "samples_f1": round(report_test["samples avg"]["f1-score"], 4),
            "train_micro_f1": round(report_train["micro avg"]["f1-score"], 4),
            "train_macro_f1": round(report_train["macro avg"]["f1-score"], 4),
        }
        results.append(result)

        gap = result["train_micro_f1"] - result["micro_f1"]
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Test  — Micro F1: {result['micro_f1']}, Macro F1: {result['macro_f1']}")
        print(f"  Train — Micro F1: {result['train_micro_f1']}, Macro F1: {result['train_macro_f1']}")
        print(f"  Gap (train-test Micro F1): {gap:+.4f}")

    return results


# ── DNN ablation ────────────────────────────────────────────────────────────

def run_dnn_ablation(df, feature_sets, go_cols, train_idx, val_idx, test_idx):
    """Train DNN with each feature set and return results."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class DNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, 128)
            self.layer2 = nn.Linear(128, 64)
            self.layer3 = nn.Linear(64, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.sigmoid(self.layer3(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = df[go_cols].values
    y_test_np = y[test_idx]
    results = []

    for name, feat_cols in feature_sets.items():
        print(f"\n--- DNN: {name} ({len(feat_cols)} features) ---")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(df[feat_cols].values[train_idx])
        X_val = scaler.transform(df[feat_cols].values[val_idx])
        X_test = scaler.transform(df[feat_cols].values[test_idx])

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        model = DNN(len(feat_cols), len(go_cols)).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start = time.time()
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(20):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    val_loss += criterion(model(bx), by).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/20, Val Loss: {val_loss:.4f}")

        elapsed = time.time() - start

        # Evaluate best model on test and train
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_pred_test = (model(X_test_t) > 0.5).int().cpu().numpy()
            y_pred_train = (model(X_train_t) > 0.5).int().cpu().numpy()

        report_test = classification_report(y_test_np, y_pred_test, target_names=go_cols,
                                            zero_division=0, output_dict=True)
        report_train = classification_report(y_train, y_pred_train, target_names=go_cols,
                                             zero_division=0, output_dict=True)

        result = {
            "model": "DNN",
            "features": name,
            "num_features": len(feat_cols),
            "time_s": round(elapsed, 1),
            "micro_precision": round(report_test["micro avg"]["precision"], 4),
            "micro_recall": round(report_test["micro avg"]["recall"], 4),
            "micro_f1": round(report_test["micro avg"]["f1-score"], 4),
            "macro_f1": round(report_test["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_test["weighted avg"]["f1-score"], 4),
            "samples_f1": round(report_test["samples avg"]["f1-score"], 4),
            "train_micro_f1": round(report_train["micro avg"]["f1-score"], 4),
            "train_macro_f1": round(report_train["macro avg"]["f1-score"], 4),
        }
        results.append(result)

        gap = result["train_micro_f1"] - result["micro_f1"]
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Test  — Micro F1: {result['micro_f1']}, Macro F1: {result['macro_f1']}")
        print(f"  Train — Micro F1: {result['train_micro_f1']}, Macro F1: {result['train_macro_f1']}")
        print(f"  Gap (train-test Micro F1): {gap:+.4f}")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature ablation study [R4.5]")
    parser.add_argument("--data-path", type=str, default="./data/final_df_with_features_expanded.csv")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-dnn", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # 1. Load data
    print("=" * 70)
    print("Step 1: Loading and filtering data")
    print("=" * 70)
    df, go_cols = load_and_filter(args.data_path)
    feature_sets = get_feature_sets(df, go_cols)

    # 2. Create splits
    print("\n" + "=" * 70)
    print("Step 2: Creating 70/15/15 stratified split")
    print("=" * 70)
    all_features = feature_sets["All features"]
    train_idx, val_idx, test_idx = create_splits(df, all_features, go_cols)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    all_results = []

    # 3. XGBoost ablation
    if not args.skip_xgboost:
        print("\n" + "=" * 70)
        print("Step 3: XGBoost ablation")
        print("=" * 70)
        xgb_results = run_xgboost_ablation(df, feature_sets, go_cols, train_idx, test_idx)
        all_results.extend(xgb_results)

    # 4. DNN ablation
    if not args.skip_dnn:
        print("\n" + "=" * 70)
        print("Step 4: DNN ablation")
        print("=" * 70)
        dnn_results = run_dnn_ablation(df, feature_sets, go_cols, train_idx, val_idx, test_idx)
        all_results.extend(dnn_results)

    # 5. Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Features':<18} {'#Feat':>6} {'Micro F1':>9} {'Macro F1':>9} "
          f"{'Weighted F1':>11} {'Samples F1':>11} {'Time':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['model']:<10} {r['features']:<18} {r['num_features']:>6} "
              f"{r['micro_f1']:>9.4f} {r['macro_f1']:>9.4f} {r['weighted_f1']:>11.4f} "
              f"{r['samples_f1']:>11.4f} {r['time_s']:>7.1f}s")

    # 5b. Train vs Test gap (overfitting diagnostic)
    print("\n" + "=" * 70)
    print("TRAIN vs TEST GAP (overfitting diagnostic)")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Features':<18} {'Train MiF1':>11} {'Test MiF1':>10} {'Gap':>8} "
          f"{'Train MaF1':>11} {'Test MaF1':>10} {'Gap':>8}")
    print("-" * 90)
    for r in all_results:
        mi_gap = r["train_micro_f1"] - r["micro_f1"]
        ma_gap = r["train_macro_f1"] - r["macro_f1"]
        print(f"{r['model']:<10} {r['features']:<18} {r['train_micro_f1']:>11.4f} "
              f"{r['micro_f1']:>10.4f} {mi_gap:>+8.4f} {r['train_macro_f1']:>11.4f} "
              f"{r['macro_f1']:>10.4f} {ma_gap:>+8.4f}")

    # 6. Save results
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
