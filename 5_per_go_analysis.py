"""Per-GO-term performance analysis for XGBoost and DNN models.

Addresses reviewer R4.8:
  - Which GO terms are most/least accurately predicted?
  - Biological patterns in errors?
  - Virulence/pathogenesis-related GO term performance?
  - Performance grouped by GO category (MF, BP, CC)?

Uses the same 70/15/15 stratified split as DNN and GNN for fair comparison.

Usage:
  python 5_per_go_analysis.py                           # full analysis
  python 5_per_go_analysis.py --skip-xgboost            # skip XGBoost retraining (DNN only)
  python 5_per_go_analysis.py --output-dir ./results     # custom output dir
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ── GO category and virulence term definitions ───────────────────────────────

# Virulence/pathogenesis-related GO terms commonly found in bacterial genomes.
# Curated from Gene Ontology for Flavobacterium context (aquaculture pathogens).
VIRULENCE_GO_TERMS = {
    # Direct virulence / pathogenesis
    "GO:0009405": "pathogenesis",
    "GO:0044419": "biological process involved in interspecies interaction between organisms",
    "GO:0051701": "biological process involved in interaction with host",
    "GO:0030001": "metal ion transport",
    "GO:0006508": "proteolysis",
    "GO:0006810": "transport",
    "GO:0055085": "transmembrane transport",
    "GO:0006811": "ion transport",
    # Secretion systems (key virulence mechanism)
    "GO:0009306": "protein secretion",
    "GO:0015031": "protein transport",
    # Biofilm and adhesion
    "GO:0022610": "biological adhesion",
    "GO:0007155": "cell adhesion",
    "GO:0042710": "biofilm formation",
    # Cell envelope / outer membrane (virulence factors)
    "GO:0009279": "cell outer membrane",
    "GO:0005886": "plasma membrane",
    "GO:0016021": "integral component of membrane",
    "GO:0005618": "cell wall",
    # Iron acquisition (critical for pathogenesis)
    "GO:0006826": "iron ion transport",
    "GO:0005506": "iron ion binding",
    "GO:0015343": "siderophore transmembrane transporter activity",
    # Gliding motility (Flavobacterium-specific virulence)
    "GO:0001539": "cilium or flagellum-dependent cell motility",
    "GO:0006928": "movement of cell or subcellular component",
    # Stress response / survival in host
    "GO:0006950": "response to stress",
    "GO:0006979": "response to oxidative stress",
    "GO:0009432": "SOS response",
    # LPS and peptidoglycan (immune evasion)
    "GO:0009252": "peptidoglycan biosynthetic process",
    "GO:0009103": "lipopolysaccharide biosynthetic process",
}


def fetch_go_categories(go_terms, cache_path=None):
    """Fetch GO term names and categories (MF/BP/CC) from QuickGO API.

    Falls back to cached results or returns 'unknown' if API unavailable.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading GO categories from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    go_info = {}
    try:
        import requests
        print(f"Fetching GO categories for {len(go_terms)} terms from QuickGO API...")
        # Batch in chunks of 200 (API limit)
        batch_size = 200
        for i in range(0, len(go_terms), batch_size):
            batch = go_terms[i:i + batch_size]
            ids = ",".join(batch)
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids}"
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for result in data.get("results", []):
                    go_id = result.get("id", "")
                    go_info[go_id] = {
                        "name": result.get("name", "unknown"),
                        "aspect": result.get("aspect", "unknown"),
                    }
            if (i // batch_size + 1) % 2 == 0:
                print(f"  Fetched {min(i + batch_size, len(go_terms))}/{len(go_terms)}")

        # Map aspect to standard abbreviations
        aspect_map = {
            "molecular_function": "MF",
            "biological_process": "BP",
            "cellular_component": "CC",
        }
        for go_id in go_info:
            go_info[go_id]["category"] = aspect_map.get(go_info[go_id]["aspect"], "unknown")

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(go_info, f, indent=2)
            print(f"Cached GO categories to {cache_path}")

    except Exception as e:
        print(f"[WARNING] Could not fetch GO categories: {e}")
        print("  Continuing without GO names/categories.")
        for go_id in go_terms:
            go_info[go_id] = {"name": "unknown", "aspect": "unknown", "category": "unknown"}

    # Fill missing terms
    for go_id in go_terms:
        if go_id not in go_info:
            go_info[go_id] = {"name": "unknown", "aspect": "unknown", "category": "unknown"}

    return go_info


# ── Data loading (shared pipeline with DNN and GNN) ─────────────────────────

def load_and_filter(path="./data/final_df_with_features_expanded.csv", go_min_freq=4):
    """Load CSV, filter rare GO terms. Same logic as DNN and GNN scripts."""
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
    """70/15/15 stratified multi-label split (same as DNN and GNN)."""
    X = df[feature_cols].values
    y = df[go_cols].values

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(X, y))

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return X_train, X_test, y_train, y_test


# ── XGBoost training ────────────────────────────────────────────────────────

def train_and_predict_xgboost(X_train, X_test, y_train, go_cols):
    """Train one XGBoost per GO term and return predictions."""
    from xgboost import XGBClassifier
    from tqdm import tqdm

    print(f"\nTraining XGBoost ({len(go_cols)} GO terms)...")
    start = time.time()

    y_pred = np.zeros((X_test.shape[0], len(go_cols)), dtype=int)

    for i, col in enumerate(tqdm(go_cols, desc="XGBoost")):
        pos_count = y_train[:, i].sum()
        if pos_count == 0:
            clf = XGBClassifier(eval_metric="logloss", random_state=42, base_score=1e-5, verbosity=0)
        else:
            clf = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
        clf.fit(X_train, y_train[:, i])
        y_pred[:, i] = clf.predict(X_test)

    elapsed = time.time() - start
    print(f"XGBoost training + prediction done in {elapsed:.1f}s")
    return y_pred


# ── DNN prediction ──────────────────────────────────────────────────────────

def predict_dnn(X_test, go_cols, model_path="./data/best_dnn_model.pth"):
    """Load saved DNN model and predict on test set."""
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


# ── Per-GO-term analysis ────────────────────────────────────────────────────

def compute_per_go_metrics(y_test, y_pred, go_cols, model_name):
    """Compute per-GO-term F1, precision, recall, support."""
    rows = []
    for i, go_id in enumerate(go_cols):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]

        support = int(y_true_i.sum())
        pred_pos = int(y_pred_i.sum())
        tp = int((y_true_i * y_pred_i).sum())
        fp = pred_pos - tp
        fn = support - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        rows.append({
            "go_term": go_id,
            "model": model_name,
            "support": support,
            "predicted_pos": pred_pos,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(rows)


def generate_analysis(df_metrics, go_info, go_cols, y_test, output_dir):
    """Generate and print comprehensive analysis tables."""

    # Add GO info columns
    df_metrics["go_name"] = df_metrics["go_term"].map(
        lambda g: go_info.get(g, {}).get("name", "unknown"))
    df_metrics["category"] = df_metrics["go_term"].map(
        lambda g: go_info.get(g, {}).get("category", "unknown"))
    df_metrics["is_virulence"] = df_metrics["go_term"].isin(VIRULENCE_GO_TERMS)

    models = df_metrics["model"].unique()

    # ── 1. Overall summary per model ──
    print("\n" + "=" * 70)
    print("1. OVERALL SUMMARY PER MODEL")
    print("=" * 70)
    for model in models:
        m = df_metrics[df_metrics["model"] == model]
        print(f"\n  {model}:")
        print(f"    Mean F1:  {m['f1'].mean():.4f} (±{m['f1'].std():.4f})")
        print(f"    Mean P:   {m['precision'].mean():.4f}")
        print(f"    Mean R:   {m['recall'].mean():.4f}")
        print(f"    F1 = 0:   {(m['f1'] == 0).sum()} GO terms")
        print(f"    F1 = 1:   {(m['f1'] == 1.0).sum()} GO terms")
        print(f"    F1 > 0.9: {(m['f1'] > 0.9).sum()} GO terms")
        print(f"    F1 < 0.5: {(m['f1'] < 0.5).sum()} GO terms")

    # ── 2. Top 15 and Bottom 15 GO terms (by DNN or first model) ──
    ref_model = "DNN" if "DNN" in models else models[0]
    ref = df_metrics[df_metrics["model"] == ref_model].sort_values("f1", ascending=False)

    print("\n" + "=" * 70)
    print(f"2. TOP 15 GO TERMS BY F1 ({ref_model})")
    print("=" * 70)
    top = ref.head(15)
    print(f"{'GO Term':<14} {'Name':<45} {'Cat':>4} {'Supp':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 90)
    for _, row in top.iterrows():
        name = row["go_name"][:43] if len(row["go_name"]) > 43 else row["go_name"]
        print(f"{row['go_term']:<14} {name:<45} {row['category']:>4} {row['support']:>5} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    print(f"\n{'=' * 70}")
    print(f"3. BOTTOM 15 GO TERMS BY F1 ({ref_model})")
    print("=" * 70)
    # Exclude GO terms with 0 support in test (meaningless)
    bottom = ref[ref["support"] > 0].tail(15)
    print(f"{'GO Term':<14} {'Name':<45} {'Cat':>4} {'Supp':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 90)
    for _, row in bottom.iterrows():
        name = row["go_name"][:43] if len(row["go_name"]) > 43 else row["go_name"]
        print(f"{row['go_term']:<14} {name:<45} {row['category']:>4} {row['support']:>5} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    # ── 4. Performance by GO category (MF, BP, CC) ──
    print(f"\n{'=' * 70}")
    print("4. PERFORMANCE BY GO CATEGORY")
    print("=" * 70)
    for model in models:
        m = df_metrics[df_metrics["model"] == model]
        print(f"\n  {model}:")
        cat_stats = m.groupby("category").agg(
            count=("f1", "size"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_p=("precision", "mean"),
            mean_r=("recall", "mean"),
            total_support=("support", "sum"),
        ).round(4)
        print(f"    {'Cat':<8} {'Count':>6} {'Mean F1':>8} {'Std F1':>8} {'Mean P':>8} {'Mean R':>8} {'Support':>8}")
        print(f"    {'-'*56}")
        for cat, row in cat_stats.iterrows():
            print(f"    {cat:<8} {int(row['count']):>6} {row['mean_f1']:>8.4f} {row['std_f1']:>8.4f} "
                  f"{row['mean_p']:>8.4f} {row['mean_r']:>8.4f} {int(row['total_support']):>8}")

    # ── 5. Virulence-related GO terms ──
    print(f"\n{'=' * 70}")
    print("5. VIRULENCE / PATHOGENESIS-RELATED GO TERMS")
    print("=" * 70)
    vir = df_metrics[df_metrics["is_virulence"]]
    if len(vir) == 0:
        print("  No virulence-related GO terms found in dataset.")
    else:
        for model in models:
            vm = vir[vir["model"] == model].sort_values("f1", ascending=False)
            non_vir = df_metrics[(df_metrics["model"] == model) & (~df_metrics["is_virulence"])]
            print(f"\n  {model} ({len(vm)} virulence terms found):")
            print(f"    Virulence mean F1:     {vm['f1'].mean():.4f}")
            print(f"    Non-virulence mean F1: {non_vir['f1'].mean():.4f}")
            print(f"\n    {'GO Term':<14} {'Name':<40} {'Supp':>5} {'P':>6} {'R':>6} {'F1':>6}")
            print(f"    {'-'*80}")
            for _, row in vm.iterrows():
                vname = VIRULENCE_GO_TERMS.get(row["go_term"], row["go_name"])[:38]
                print(f"    {row['go_term']:<14} {vname:<40} {row['support']:>5} "
                      f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    # ── 6. Performance by support bucket ──
    print(f"\n{'=' * 70}")
    print("6. PERFORMANCE BY GO TERM FREQUENCY (TEST SUPPORT)")
    print("=" * 70)
    for model in models:
        m = df_metrics[df_metrics["model"] == model].copy()
        bins = [0, 5, 20, 50, 200, float("inf")]
        labels = ["1-5", "6-20", "21-50", "51-200", "200+"]
        m["freq_bin"] = pd.cut(m["support"], bins=bins, labels=labels, right=True)
        print(f"\n  {model}:")
        print(f"    {'Bin':<8} {'Count':>6} {'Mean F1':>8} {'Std F1':>8} {'Mean P':>8} {'Mean R':>8}")
        print(f"    {'-'*48}")
        for label in labels:
            subset = m[m["freq_bin"] == label]
            if len(subset) > 0:
                print(f"    {label:<8} {len(subset):>6} {subset['f1'].mean():>8.4f} {subset['f1'].std():>8.4f} "
                      f"{subset['precision'].mean():>8.4f} {subset['recall'].mean():>8.4f}")

    # ── 7. Cross-model comparison (if multiple models) ──
    if len(models) > 1:
        print(f"\n{'=' * 70}")
        print("7. CROSS-MODEL COMPARISON (PER GO TERM)")
        print("=" * 70)

        pivot = df_metrics.pivot(index="go_term", columns="model", values="f1")
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                diff = pivot[m1] - pivot[m2]
                print(f"\n  {m1} vs {m2}:")
                print(f"    {m1} wins: {(diff > 0).sum()} GO terms")
                print(f"    {m2} wins: {(diff < 0).sum()} GO terms")
                print(f"    Tied:     {(diff == 0).sum()} GO terms")
                print(f"    Mean F1 diff: {diff.mean():+.4f}")

                # GO terms where models differ most
                top_diff = diff.abs().nlargest(5)
                print(f"\n    Top 5 largest disagreements:")
                for go_id, d in top_diff.items():
                    actual = diff[go_id]
                    winner = m1 if actual > 0 else m2
                    name = go_info.get(go_id, {}).get("name", "unknown")[:35]
                    print(f"      {go_id} ({name}): {m1}={pivot[m1][go_id]:.3f}, "
                          f"{m2}={pivot[m2][go_id]:.3f}  [{winner} +{abs(actual):.3f}]")

    # ── Save full results to CSV ──
    csv_path = os.path.join(output_dir, "per_go_analysis.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n\nFull results saved to {csv_path}")

    # ── Save summary JSON ──
    summary = {}
    for model in models:
        m = df_metrics[df_metrics["model"] == model]
        summary[model] = {
            "mean_f1": round(m["f1"].mean(), 4),
            "std_f1": round(m["f1"].std(), 4),
            "mean_precision": round(m["precision"].mean(), 4),
            "mean_recall": round(m["recall"].mean(), 4),
            "f1_zero_count": int((m["f1"] == 0).sum()),
            "f1_above_90_count": int((m["f1"] > 0.9).sum()),
            "f1_below_50_count": int((m["f1"] < 0.5).sum()),
        }
        # Category breakdown
        for cat in ["MF", "BP", "CC"]:
            cat_m = m[m["category"] == cat]
            if len(cat_m) > 0:
                summary[model][f"{cat}_mean_f1"] = round(cat_m["f1"].mean(), 4)
                summary[model][f"{cat}_count"] = len(cat_m)

        # Virulence
        vir_m = m[m["is_virulence"]]
        if len(vir_m) > 0:
            summary[model]["virulence_mean_f1"] = round(vir_m["f1"].mean(), 4)
            summary[model]["virulence_count"] = len(vir_m)

    json_path = os.path.join(output_dir, "per_go_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")

    return df_metrics


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-GO-term performance analysis [R4.8]")
    parser.add_argument("--data-path", type=str, default="./data/final_df_with_features_expanded.csv")
    parser.add_argument("--dnn-model", type=str, default="./data/best_dnn_model.pth")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost (DNN only)")
    parser.add_argument("--go-cache", type=str, default="./data/go_categories_cache.json",
                        help="Cache file for GO term categories")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # 1. Load data
    print("=" * 70)
    print("Step 1: Loading and filtering data")
    print("=" * 70)
    df, feature_cols, go_cols = load_and_filter(args.data_path)

    # 2. Create splits (same as DNN/GNN)
    print("\n" + "=" * 70)
    print("Step 2: Creating 70/15/15 stratified split")
    print("=" * 70)
    X_train, X_test, y_train, y_test = create_splits(df, feature_cols, go_cols)

    # 3. Fetch GO categories
    print("\n" + "=" * 70)
    print("Step 3: Fetching GO term categories")
    print("=" * 70)
    go_info = fetch_go_categories(go_cols, cache_path=args.go_cache)
    found = sum(1 for g in go_cols if go_info.get(g, {}).get("category", "unknown") != "unknown")
    print(f"  Categories resolved: {found}/{len(go_cols)}")

    # 4. Train/predict models
    all_metrics = []

    if not args.skip_xgboost:
        print("\n" + "=" * 70)
        print("Step 4a: XGBoost")
        print("=" * 70)
        y_pred_xgb = train_and_predict_xgboost(X_train, X_test, y_train, go_cols)
        xgb_metrics = compute_per_go_metrics(y_test, y_pred_xgb, go_cols, "XGBoost")
        all_metrics.append(xgb_metrics)

    if os.path.exists(args.dnn_model):
        print("\n" + "=" * 70)
        print("Step 4b: DNN")
        print("=" * 70)
        y_pred_dnn = predict_dnn(X_test, go_cols, model_path=args.dnn_model)
        dnn_metrics = compute_per_go_metrics(y_test, y_pred_dnn, go_cols, "DNN")
        all_metrics.append(dnn_metrics)
    else:
        print(f"\n[WARNING] DNN model not found at {args.dnn_model}. Skipping DNN.")

    if not all_metrics:
        print("No models to analyze. Exiting.")
        return

    # 5. Analysis
    df_metrics = pd.concat(all_metrics, ignore_index=True)

    print("\n" + "=" * 70)
    print("Step 5: Per-GO-term analysis")
    print("=" * 70)
    generate_analysis(df_metrics, go_info, go_cols, y_test, args.output_dir)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
