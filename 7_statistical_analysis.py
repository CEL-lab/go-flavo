"""Per-GO category breakdown and statistical significance testing.

Addresses:
  6d / R4.5: Formal per-GO-category breakdown (MF, BP, CC) for manuscript
  6e / R4.4: Paired bootstrap significance tests for model comparisons

Usage:
  python 7_statistical_analysis.py                    # full analysis (category + significance)
  python 7_statistical_analysis.py --skip-significance # category breakdown only (fast, uses existing CSV)
  python 7_statistical_analysis.py --skip-category     # significance only
  python 7_statistical_analysis.py --n-bootstrap 1000  # fewer bootstrap iterations (faster)
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)


# ── Shared data loading (same as other scripts) ─────────────────────────────

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

    feature_cols = [col for col in df.columns if col not in ["sequence"] + remaining_go]
    return df, feature_cols, remaining_go


def create_splits(df, feature_cols, go_cols, random_state=42):
    """70/15/15 stratified multi-label split (same as DNN/GNN/ablation)."""
    X = df[feature_cols].values
    y = df[go_cols].values

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(X, y))

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# ── 6d: Per-GO category breakdown ───────────────────────────────────────────

def category_breakdown(per_go_csv, go_cache_path, output_dir):
    """Detailed per-GO-category breakdown from existing per_go_analysis.csv."""
    print("\n" + "=" * 70)
    print("6d: PER-GO CATEGORY BREAKDOWN (MF / BP / CC)")
    print("=" * 70)

    df = pd.read_csv(per_go_csv)
    models = df["model"].unique()

    # Load GO categories
    if os.path.exists(go_cache_path):
        with open(go_cache_path) as f:
            go_info = json.load(f)
    else:
        go_info = {}

    if "category" not in df.columns:
        df["category"] = df["go_term"].map(
            lambda g: go_info.get(g, {}).get("category", "unknown"))

    # ── Summary table (manuscript-ready) ──
    print("\n### Manuscript table: Performance by GO category\n")
    print(f"{'Category':<10} {'Count':>6}", end="")
    for model in models:
        print(f" | {model+' P':>10} {model+' R':>10} {model+' F1':>10}", end="")
    print()
    print("-" * (18 + len(models) * 34))

    category_results = {}
    for cat in ["MF", "BP", "CC"]:
        cat_data = df[df["category"] == cat]
        if len(cat_data) == 0:
            continue
        count = len(cat_data[cat_data["model"] == models[0]])
        print(f"{cat:<10} {count:>6}", end="")

        category_results[cat] = {"count": count}
        for model in models:
            m = cat_data[cat_data["model"] == model]
            mp = m["precision"].mean()
            mr = m["recall"].mean()
            mf = m["f1"].mean()
            print(f" | {mp:>10.4f} {mr:>10.4f} {mf:>10.4f}", end="")
            category_results[cat][model] = {
                "mean_precision": round(mp, 4),
                "mean_recall": round(mr, 4),
                "mean_f1": round(mf, 4),
                "std_f1": round(m["f1"].std(), 4),
            }
        print()

    # ── DNN advantage by category ──
    if len(models) >= 2:
        print("\n### DNN advantage over XGBoost by GO category\n")
        pivot = df.pivot(index="go_term", columns="model", values="f1")
        for cat in ["MF", "BP", "CC"]:
            cat_terms = df[df["category"] == cat]["go_term"].unique()
            cat_pivot = pivot.loc[pivot.index.isin(cat_terms)]
            if "DNN" in models and "XGBoost" in models:
                diff = cat_pivot["DNN"] - cat_pivot["XGBoost"]
                dnn_wins = (diff > 0).sum()
                xgb_wins = (diff < 0).sum()
                tied = (diff == 0).sum()
                print(f"  {cat}: DNN wins {dnn_wins}, XGBoost wins {xgb_wins}, tied {tied} "
                      f"(mean diff: {diff.mean():+.4f})")

    # ── F1 distribution by category ──
    print("\n### F1 distribution by category\n")
    for model in models:
        print(f"  {model}:")
        m = df[df["model"] == model]
        for cat in ["MF", "BP", "CC"]:
            cm = m[m["category"] == cat]
            if len(cm) == 0:
                continue
            print(f"    {cat}: F1=0: {(cm['f1']==0).sum()}, "
                  f"F1<0.5: {(cm['f1']<0.5).sum()}, "
                  f"F1>0.9: {(cm['f1']>0.9).sum()}, "
                  f"F1=1.0: {(cm['f1']==1.0).sum()}")

    # Save category results
    cat_json_path = os.path.join(output_dir, "category_breakdown.json")
    with open(cat_json_path, "w") as f:
        json.dump(category_results, f, indent=2)
    print(f"\nCategory breakdown saved to {cat_json_path}")

    return category_results


# ── 6e: Statistical significance testing ─────────────────────────────────────

def get_xgboost_predictions(X_train, X_test, y_train, go_cols):
    """Train XGBoost (all features) and return test predictions."""
    from xgboost import XGBClassifier

    print("\nTraining XGBoost (762 models) for significance testing...")
    start = time.time()
    y_pred = np.zeros((X_test.shape[0], len(go_cols)), dtype=int)

    for i in range(len(go_cols)):
        pos_count = y_train[:, i].sum()
        if pos_count == 0:
            clf = XGBClassifier(eval_metric="logloss", random_state=42, base_score=1e-5, verbosity=0)
        else:
            clf = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
        clf.fit(X_train, y_train[:, i])
        y_pred[:, i] = clf.predict(X_test)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(go_cols)} GO terms done")

    elapsed = time.time() - start
    print(f"XGBoost done in {elapsed:.1f}s")
    return y_pred


def get_dnn_predictions(X_test, go_cols, model_path="./data/best_dnn_model.pth"):
    """Load saved DNN and return test predictions."""
    import torch
    import torch.nn as nn

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
    model = DNN(X_test.shape[1], len(go_cols)).to(device)

    print(f"\nLoading DNN model from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred = (outputs > 0.5).int().cpu().numpy()

    print(f"DNN prediction done ({y_pred.shape[0]} samples)")
    return y_pred


def _fast_micro_f1(y_true, y_pred):
    """Compute micro F1 using numpy (much faster than sklearn for bootstrap)."""
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _fast_macro_f1(y_true, y_pred):
    """Compute macro F1 using vectorized numpy (per-label then average)."""
    tp = (y_true & y_pred).sum(axis=0)  # shape: (n_labels,)
    fp = (~y_true & y_pred).sum(axis=0)
    fn = (y_true & ~y_pred).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_p = tp + fp
        denom_r = tp + fn
        prec = np.where(denom_p > 0, tp / denom_p, 0.0)
        rec = np.where(denom_r > 0, tp / denom_r, 0.0)
        denom_f = prec + rec
        f1 = np.where(denom_f > 0, 2 * prec * rec / denom_f, 0.0)
    return f1.mean()


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, model_a_name, model_b_name,
                          n_bootstrap=10000, seed=42):
    """Paired bootstrap significance test for multi-label F1 differences.

    Uses vectorized numpy for speed (sklearn f1_score is too slow for 10K iterations).
    Returns dict with CIs for Micro F1 and Macro F1 differences.
    """
    rng = np.random.RandomState(seed)
    n_samples = y_true.shape[0]

    # Convert to bool arrays for fast bitwise ops
    y_true_b = y_true.astype(bool)
    y_pred_a_b = y_pred_a.astype(bool)
    y_pred_b_b = y_pred_b.astype(bool)

    micro_diffs = np.empty(n_bootstrap)
    macro_diffs = np.empty(n_bootstrap)

    print(f"\n  Running {n_bootstrap} bootstrap iterations ({model_a_name} vs {model_b_name})...")
    start = time.time()

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)

        y_t = y_true_b[idx]
        y_a = y_pred_a_b[idx]
        y_b = y_pred_b_b[idx]

        micro_diffs[b] = _fast_micro_f1(y_t, y_a) - _fast_micro_f1(y_t, y_b)
        macro_diffs[b] = _fast_macro_f1(y_t, y_a) - _fast_macro_f1(y_t, y_b)

        if (b + 1) % 2000 == 0:
            elapsed_so_far = time.time() - start
            rate = (b + 1) / elapsed_so_far
            eta = (n_bootstrap - b - 1) / rate
            print(f"    {b+1}/{n_bootstrap} done ({rate:.0f} iter/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"  Bootstrap done in {elapsed:.1f}s ({n_bootstrap/elapsed:.0f} iter/s)")

    result = {
        "comparison": f"{model_a_name} vs {model_b_name}",
        "n_bootstrap": n_bootstrap,
        "time_s": round(elapsed, 1),
        "micro_f1": {
            "mean_diff": round(float(micro_diffs.mean()), 4),
            "std_diff": round(float(micro_diffs.std()), 4),
            "ci_95_lower": round(float(np.percentile(micro_diffs, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(micro_diffs, 97.5)), 4),
            "significant": bool(np.percentile(micro_diffs, 2.5) > 0 or np.percentile(micro_diffs, 97.5) < 0),
        },
        "macro_f1": {
            "mean_diff": round(float(macro_diffs.mean()), 4),
            "std_diff": round(float(macro_diffs.std()), 4),
            "ci_95_lower": round(float(np.percentile(macro_diffs, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(macro_diffs, 97.5)), 4),
            "significant": bool(np.percentile(macro_diffs, 2.5) > 0 or np.percentile(macro_diffs, 97.5) < 0),
        },
    }

    return result


def run_significance_tests(df, feature_cols, go_cols, train_idx, test_idx,
                           dnn_model_path, n_bootstrap, output_dir):
    """Run paired bootstrap tests between XGBoost and DNN."""
    print("\n" + "=" * 70)
    print("6e: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)

    y = df[go_cols].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Scale features (same as DNN training)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df[feature_cols].values[train_idx])
    X_test = scaler.transform(df[feature_cols].values[test_idx])

    # Get predictions
    y_pred_xgb = get_xgboost_predictions(X_train, X_test, y_train, go_cols)
    y_pred_dnn = get_dnn_predictions(X_test, go_cols, model_path=dnn_model_path)

    # Verify test metrics match known results
    xgb_micro = f1_score(y_test, y_pred_xgb, average="micro", zero_division=0)
    dnn_micro = f1_score(y_test, y_pred_dnn, average="micro", zero_division=0)
    print(f"\nSanity check — XGBoost Micro F1: {xgb_micro:.4f}, DNN Micro F1: {dnn_micro:.4f}")

    # Paired bootstrap: DNN vs XGBoost
    print("\n" + "-" * 70)
    print("Paired bootstrap: DNN vs XGBoost")
    print("-" * 70)
    result_dnn_xgb = paired_bootstrap_test(
        y_test, y_pred_dnn, y_pred_xgb, "DNN", "XGBoost", n_bootstrap=n_bootstrap)

    # Print results
    all_results = [result_dnn_xgb]

    print("\n" + "=" * 70)
    print("SIGNIFICANCE RESULTS")
    print("=" * 70)

    for r in all_results:
        print(f"\n  {r['comparison']} (B={r['n_bootstrap']}, {r['time_s']}s):")
        mi = r["micro_f1"]
        ma = r["macro_f1"]
        print(f"    Micro F1 diff: {mi['mean_diff']:+.4f} "
              f"[{mi['ci_95_lower']:+.4f}, {mi['ci_95_upper']:+.4f}] "
              f"{'*** SIGNIFICANT' if mi['significant'] else 'not significant'}")
        print(f"    Macro F1 diff: {ma['mean_diff']:+.4f} "
              f"[{ma['ci_95_lower']:+.4f}, {ma['ci_95_upper']:+.4f}] "
              f"{'*** SIGNIFICANT' if ma['significant'] else 'not significant'}")

    # Save
    json_path = os.path.join(output_dir, "significance_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return all_results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-GO category breakdown (6d) and significance testing (6e)")
    parser.add_argument("--data-path", type=str, default="./data/final_df_with_features_expanded.csv")
    parser.add_argument("--per-go-csv", type=str, default="./data/per_go_analysis.csv")
    parser.add_argument("--go-cache", type=str, default="./data/go_categories_cache.json")
    parser.add_argument("--dnn-model", type=str, default="./data/best_dnn_model.pth")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--skip-category", action="store_true", help="Skip 6d category breakdown")
    parser.add_argument("--skip-significance", action="store_true", help="Skip 6e significance tests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # 6d: Category breakdown (from existing CSV, fast)
    if not args.skip_category:
        category_breakdown(args.per_go_csv, args.go_cache, args.output_dir)

    # 6e: Significance testing (requires retraining XGBoost, ~18 min)
    if not args.skip_significance:
        print("\n" + "=" * 70)
        print("Loading data for significance testing")
        print("=" * 70)
        df, feature_cols, go_cols = load_and_filter(args.data_path)
        train_idx, val_idx, test_idx = create_splits(df, feature_cols, go_cols)

        run_significance_tests(
            df, feature_cols, go_cols, train_idx, test_idx,
            args.dnn_model, args.n_bootstrap, args.output_dir)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
