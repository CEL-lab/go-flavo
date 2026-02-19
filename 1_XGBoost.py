import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

# ── Data loading and filtering ───────────────────────────────────────────────

df = pd.read_csv("./data/final_df_with_features_expanded.csv")

GO_MIN_FREQ = 4
go_cols = [col for col in df.columns if col.startswith("GO:")]
go_counts = df[go_cols].sum()
keep_go = go_counts[go_counts >= GO_MIN_FREQ].index.tolist()
drop_go = go_counts[go_counts < GO_MIN_FREQ].index.tolist()

print(f"Total GO terms: {len(go_cols)}")
print(f"Keeping (freq >= {GO_MIN_FREQ}): {len(keep_go)}")
print(f"Dropping (freq < {GO_MIN_FREQ}): {len(drop_go)}")

df = df.drop(columns=drop_go)

remaining_go = [col for col in df.columns if col.startswith("GO:")]
mask = df[remaining_go].sum(axis=1) > 0
print(f"Rows before: {len(df)}, after removing zero-annotation rows: {mask.sum()}")
df = df[mask].reset_index(drop=True)

# ── Feature and target columns ───────────────────────────────────────────────

features = [col for col in df.columns if col not in ["sequence"] + [col for col in df.columns if col.startswith("GO:")]]
go_terms = [col for col in df.columns if col.startswith("GO:")]

X = df[features].values
y = df[go_terms].values

# ── 70/15/15 stratified multi-label split (same as DNN, GNN, per-GO, ablation) ──

msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(msss1.split(X, y))

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], y[temp_idx]))
val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

rare_counts = y_train.sum(axis=0)
print(f"GO terms with 0 positives in train: {(rare_counts == 0).sum()}")
print(f"GO terms with 1-3 positives in train: {((rare_counts >= 1) & (rare_counts <= 3)).sum()}")
print(f"GO terms with 4+ positives in train: {(rare_counts >= 4).sum()}")
print(f"Total GO terms: {len(go_terms)}")

# ── Train one XGBoost per GO term ────────────────────────────────────────────

start_time = time.time()
estimators = []
for i, col in enumerate(tqdm(go_terms, desc="Training XGBoost per GO term")):
    pos_count = y_train[:, i].sum()
    if pos_count == 0:
        clf = XGBClassifier(eval_metric="logloss", random_state=42, base_score=1e-5, verbosity=0)
    else:
        clf = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    clf.fit(X_train, y_train[:, i])
    estimators.append(clf)

train_time = time.time() - start_time

# ── Predict ──────────────────────────────────────────────────────────────────

y_pred = np.column_stack([est.predict(X_test) for est in tqdm(estimators, desc="Predicting")])

# ── Evaluate ─────────────────────────────────────────────────────────────────

print("\nClassification Report (per GO term):")
report_text = classification_report(y_test, y_pred, target_names=go_terms, zero_division=0)
print(report_text)

report_dict = classification_report(y_test, y_pred, target_names=go_terms, zero_division=0, output_dict=True)

results = {
    "model": "XGBoost",
    "split": "70/15/15",
    "random_state": 42,
    "train_size": len(train_idx),
    "val_size": len(val_idx),
    "test_size": len(test_idx),
    "num_go_terms": len(go_terms),
    "num_features": len(features),
    "train_time_s": round(train_time, 1),
    "micro_precision": round(report_dict["micro avg"]["precision"], 4),
    "micro_recall": round(report_dict["micro avg"]["recall"], 4),
    "micro_f1": round(report_dict["micro avg"]["f1-score"], 4),
    "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    "samples_f1": round(report_dict["samples avg"]["f1-score"], 4),
}

print("\nSummary:")
for k, v in results.items():
    print(f"  {k}: {v}")

with open("./data/xgboost_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to ./data/xgboost_results.json")
