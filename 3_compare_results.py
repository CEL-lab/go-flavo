#!/usr/bin/env python3
"""Evaluate external GO predictors on a reproducible v2 evaluation set.

This script supports:
- Internal-comparable evaluation on the same 70/15/15 test split used in v2.
- Optional random-sample evaluation (legacy-style) for manuscript back-compat.
- Multiple external model formats (ProteInfer, DeepGO-SE, NetGO, TransFew).
- Export of FASTA for the chosen evaluation set using sequence_<index> IDs.

Example:
  python3 3_compare_results.py \
      --split-mode test_split \
      --export-fasta ./data/external_test_split.fasta \
      --proteinfer ./data/external/proteinfer.tsv \
      --deepgose-bp ./data/external/deep_go_se_bp.tsv \
      --deepgose-cc ./data/external/deep_go_se_cc.tsv \
      --deepgose-mf ./data/external/deep_go_se_mf.tsv \
      --netgo ./data/external/netgo_1.txt ./data/external/netgo_2.txt
"""

from __future__ import annotations

import argparse
import json
import textwrap
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings(
    "ignore",
    message=r".*unknown class.*",
    category=UserWarning,
    module=r"sklearn\.preprocessing\._label",
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate external GO predictors on v2 data with reproducible splits."
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATA_DIR / "final_df_with_features_expanded.csv"),
        help="Path to v2 feature-expanded dataset CSV.",
    )
    parser.add_argument(
        "--go-min-freq",
        type=int,
        default=4,
        help="Drop GO terms with frequency lower than this value (default: 4).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for the internal 70/15/15 stratified split.",
    )

    parser.add_argument(
        "--split-mode",
        choices=["test_split", "random_sample", "all"],
        default="test_split",
        help=(
            "Evaluation set selection: internal test split (default), random sample, or all rows."
        ),
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.2,
        help="Fraction to sample when --split-mode=random_sample.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=1,
        help="Random state for --split-mode=random_sample.",
    )

    parser.add_argument(
        "--export-fasta",
        default=None,
        help="Optional path to write FASTA for the selected evaluation set.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export the evaluation FASTA (if requested) and exit.",
    )

    parser.add_argument(
        "--proteinfer",
        default=str(DEFAULT_DATA_DIR / "ver1" / "proteinfer.tsv"),
        help="Path to ProteInfer TSV output.",
    )
    parser.add_argument(
        "--proteinfer-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for ProteInfer GO predictions.",
    )

    parser.add_argument(
        "--deepgose-bp",
        default=str(DEFAULT_DATA_DIR / "ver1" / "deep_go_se_bp.tsv"),
        help="Path to DeepGO-SE BP predictions TSV.",
    )
    parser.add_argument(
        "--deepgose-cc",
        default=str(DEFAULT_DATA_DIR / "ver1" / "deep_go_se_cc.tsv"),
        help="Path to DeepGO-SE CC predictions TSV.",
    )
    parser.add_argument(
        "--deepgose-mf",
        default=str(DEFAULT_DATA_DIR / "ver1" / "deep_go_se_mf.tsv"),
        help="Path to DeepGO-SE MF predictions TSV.",
    )
    parser.add_argument(
        "--deepgose-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for DeepGO-SE predictions.",
    )

    parser.add_argument(
        "--netgo",
        nargs="*",
        default=[
            str(DEFAULT_DATA_DIR / "ver1" / "netG0_1.txt"),
            str(DEFAULT_DATA_DIR / "ver1" / "netG0_2.txt"),
        ],
        help="One or more NetGO prediction files.",
    )
    parser.add_argument(
        "--netgo-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for NetGO predictions.",
    )

    parser.add_argument(
        "--transfew",
        nargs="*",
        default=[],
        help="Optional one or more TransFew prediction TSV files.",
    )
    parser.add_argument(
        "--transfew-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for TransFew predictions.",
    )

    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_DATA_DIR / "external_eval_metrics.csv"),
        help="Output CSV path for summary metrics.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_DATA_DIR / "external_eval_metrics.json"),
        help="Output JSON path for summary metrics and metadata.",
    )

    return parser.parse_args()


def load_and_filter_dataset(path: str, go_min_freq: int) -> tuple[pd.DataFrame, list[str]]:
    """Load v2 data and apply the same GO filtering as internal models."""
    df = pd.read_csv(path)

    go_cols = [c for c in df.columns if c.startswith("GO:")]
    if not go_cols:
        raise ValueError("No GO columns found in dataset.")

    go_counts = df[go_cols].sum()
    keep_go = go_counts[go_counts >= go_min_freq].index.tolist()
    drop_go = go_counts[go_counts < go_min_freq].index.tolist()

    print(f"Total GO terms: {len(go_cols)}")
    print(f"Keeping GO terms (freq >= {go_min_freq}): {len(keep_go)}")
    print(f"Dropping GO terms (freq < {go_min_freq}): {len(drop_go)}")

    df = df.drop(columns=drop_go)
    go_cols = [c for c in df.columns if c.startswith("GO:")]

    mask = df[go_cols].sum(axis=1) > 0
    print(f"Rows before filtering zero-annotation proteins: {len(df)}")
    print(f"Rows after filtering: {int(mask.sum())}")

    df = df[mask].reset_index(drop=True)
    return df, go_cols


def create_internal_split_indices(
    df: pd.DataFrame,
    go_cols: list[str],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the same 70/15/15 split logic used by internal models."""
    feature_cols = [
        c for c in df.columns if c != "sequence" and not c.startswith("GO:")
    ]

    X = df[feature_cols].values
    y = df[go_cols].values

    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=random_state
    )
    train_idx, temp_idx = next(msss1.split(X, y))

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=random_state
    )
    val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], y[temp_idx]))

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    print(
        f"Internal split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def select_eval_indices(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if args.split_mode == "test_split":
        _, _, test_idx = create_internal_split_indices(
            df,
            go_cols=[c for c in df.columns if c.startswith("GO:")],
            random_state=args.random_state,
        )
        return np.array(sorted(test_idx))

    if args.split_mode == "random_sample":
        sample = df.sample(frac=args.sample_frac, random_state=args.sample_random_state)
        print(
            f"Random sample mode: frac={args.sample_frac}, random_state={args.sample_random_state}, n={len(sample)}"
        )
        return np.array(sorted(sample.index.values))

    print(f"All-rows mode: n={len(df)}")
    return np.array(sorted(df.index.values))


def normalize_sequence_name(value: object) -> str:
    seq = str(value).strip()
    if seq.startswith("sequence_"):
        seq = seq[len("sequence_") :]
    return seq


def clean_predictions(
    df: pd.DataFrame,
    sequence_col: str,
    label_col: str,
    confidence_col: str | None,
    threshold: float,
) -> pd.DataFrame:
    out = df[[sequence_col, label_col] + ([confidence_col] if confidence_col else [])].copy()
    out = out.rename(columns={sequence_col: "sequence_name", label_col: "predicted_label"})

    out["sequence_name"] = out["sequence_name"].map(normalize_sequence_name)
    out["predicted_label"] = out["predicted_label"].astype(str).str.strip()
    out = out[out["predicted_label"].str.startswith("GO:")]

    if confidence_col:
        out[confidence_col] = pd.to_numeric(out[confidence_col], errors="coerce")
        out = out[out[confidence_col] >= threshold]

    out = out[["sequence_name", "predicted_label"]].dropna()
    out = out.drop_duplicates()
    return out


def load_proteinfer(path: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"sequence_name", "predicted_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Proteinfer file missing columns: {required}")
    confidence_col = "confidence" if "confidence" in df.columns else None
    return clean_predictions(
        df,
        sequence_col="sequence_name",
        label_col="predicted_label",
        confidence_col=confidence_col,
        threshold=threshold,
    )


def load_deepgose(bp_path: str, cc_path: str, mf_path: str, threshold: float) -> pd.DataFrame:
    cols = ["sequence_name", "predicted_label", "confidence"]
    bp = pd.read_csv(bp_path, sep="\t", header=None, names=cols)
    cc = pd.read_csv(cc_path, sep="\t", header=None, names=cols)
    mf = pd.read_csv(mf_path, sep="\t", header=None, names=cols)
    all_df = pd.concat([bp, cc, mf], axis=0, ignore_index=True)
    return clean_predictions(
        all_df,
        sequence_col="sequence_name",
        label_col="predicted_label",
        confidence_col="confidence",
        threshold=threshold,
    )


def load_netgo(paths: Iterable[str], threshold: float) -> pd.DataFrame:
    cols = ["sequence_name", "predicted_label", "confidence", "aspect", "description"]
    frames = [pd.read_csv(path, sep="\t", header=None, names=cols) for path in paths]
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    return clean_predictions(
        all_df,
        sequence_col="sequence_name",
        label_col="predicted_label",
        confidence_col="confidence",
        threshold=threshold,
    )


def load_transfew(paths: Iterable[str], threshold: float) -> pd.DataFrame:
    cols = ["sequence_name", "predicted_label", "confidence"]
    frames = [pd.read_csv(path, sep="\t", header=None, names=cols) for path in paths]
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    return clean_predictions(
        all_df,
        sequence_col="sequence_name",
        label_col="predicted_label",
        confidence_col="confidence",
        threshold=threshold,
    )


def build_ground_truth(df: pd.DataFrame, indices: np.ndarray, go_cols: list[str]) -> pd.DataFrame:
    subset = df.loc[indices, ["sequence"] + go_cols].copy()
    subset["sequence_name"] = subset.index.astype(str)
    subset["true_go_terms"] = subset[go_cols].apply(
        lambda row: [go for go in go_cols if row[go] == 1], axis=1
    )
    return subset[["sequence_name", "true_go_terms", "sequence"]]


def evaluate_model(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    go_cols: list[str],
    method: str,
    model_name: str,
) -> dict:
    pred_grouped = (
        predictions.groupby("sequence_name")["predicted_label"]
        .apply(lambda vals: sorted(set(vals)))
        .reset_index(name="predicted_go_terms")
    )

    merged = ground_truth.merge(pred_grouped, on="sequence_name", how="left")
    merged["predicted_go_terms"] = merged["predicted_go_terms"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    if method == "gt_only":
        classes = go_cols
    elif method == "union":
        pred_terms = {
            term for labels in merged["predicted_go_terms"] for term in labels
        }
        classes = sorted(set(go_cols) | pred_terms)
    else:
        raise ValueError(f"Unknown method: {method}")

    mlb = MultiLabelBinarizer(classes=classes)
    y_true = mlb.fit_transform(merged["true_go_terms"])
    y_pred = mlb.transform(merged["predicted_go_terms"])

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    def get_metric(block: str, metric: str) -> float:
        return float(report.get(block, {}).get(metric, 0.0))

    gt_ids = set(ground_truth["sequence_name"].tolist())
    pred_ids = set(predictions["sequence_name"].tolist())
    matched_ids = gt_ids & pred_ids

    return {
        "model": model_name,
        "method": method,
        "micro_precision": round(get_metric("micro avg", "precision"), 4),
        "micro_recall": round(get_metric("micro avg", "recall"), 4),
        "micro_f1": round(get_metric("micro avg", "f1-score"), 4),
        "macro_precision": round(get_metric("macro avg", "precision"), 4),
        "macro_recall": round(get_metric("macro avg", "recall"), 4),
        "macro_f1": round(get_metric("macro avg", "f1-score"), 4),
        "weighted_precision": round(get_metric("weighted avg", "precision"), 4),
        "weighted_recall": round(get_metric("weighted avg", "recall"), 4),
        "weighted_f1": round(get_metric("weighted avg", "f1-score"), 4),
        "samples_precision": round(get_metric("samples avg", "precision"), 4),
        "samples_recall": round(get_metric("samples avg", "recall"), 4),
        "samples_f1": round(get_metric("samples avg", "f1-score"), 4),
        "support": int(report.get("micro avg", {}).get("support", 0)),
        "gt_sequences": len(gt_ids),
        "pred_sequences": len(pred_ids),
        "matched_sequences": len(matched_ids),
        "coverage_pct": round(100.0 * len(matched_ids) / max(1, len(gt_ids)), 2),
        "prediction_rows": int(len(predictions)),
        "unique_predicted_go_terms": int(predictions["predicted_label"].nunique()),
    }


def write_fasta(df: pd.DataFrame, indices: np.ndarray, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx in indices:
            seq = str(df.at[idx, "sequence"]).strip()
            if not seq:
                continue
            f.write(f">sequence_{idx}\n")
            for chunk in textwrap.wrap(seq, width=80):
                f.write(chunk + "\n")

    print(f"Wrote FASTA for {len(indices)} proteins to: {out_path}")


def path_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def main() -> None:
    args = parse_args()

    df, go_cols = load_and_filter_dataset(args.dataset, args.go_min_freq)
    eval_indices = select_eval_indices(df, args)

    if args.export_fasta:
        write_fasta(df, eval_indices, args.export_fasta)

    if args.export_only:
        print("Export-only mode complete.")
        return

    ground_truth = build_ground_truth(df, eval_indices, go_cols)
    print(f"Evaluation proteins: {len(ground_truth)}")

    model_predictions: list[tuple[str, pd.DataFrame, dict]] = []

    if path_exists(args.proteinfer):
        pred = load_proteinfer(args.proteinfer, args.proteinfer_threshold)
        model_predictions.append(
            (
                "ProteInfer",
                pred,
                {
                    "path": args.proteinfer,
                    "threshold": args.proteinfer_threshold,
                },
            )
        )
    else:
        print(f"[SKIP] ProteInfer file not found: {args.proteinfer}")

    if path_exists(args.deepgose_bp) and path_exists(args.deepgose_cc) and path_exists(args.deepgose_mf):
        pred = load_deepgose(
            args.deepgose_bp,
            args.deepgose_cc,
            args.deepgose_mf,
            args.deepgose_threshold,
        )
        model_predictions.append(
            (
                "DeepGO-SE",
                pred,
                {
                    "bp_path": args.deepgose_bp,
                    "cc_path": args.deepgose_cc,
                    "mf_path": args.deepgose_mf,
                    "threshold": args.deepgose_threshold,
                },
            )
        )
    else:
        print("[SKIP] DeepGO-SE file(s) missing.")

    netgo_paths = [p for p in args.netgo if path_exists(p)]
    if netgo_paths:
        pred = load_netgo(netgo_paths, args.netgo_threshold)
        model_predictions.append(
            (
                "NetGO",
                pred,
                {
                    "paths": netgo_paths,
                    "threshold": args.netgo_threshold,
                },
            )
        )
    else:
        print("[SKIP] NetGO file(s) missing.")

    transfew_paths = [p for p in args.transfew if path_exists(p)]
    if transfew_paths:
        pred = load_transfew(transfew_paths, args.transfew_threshold)
        model_predictions.append(
            (
                "TransFew",
                pred,
                {
                    "paths": transfew_paths,
                    "threshold": args.transfew_threshold,
                },
            )
        )

    if not model_predictions:
        raise RuntimeError(
            "No external prediction files were found. Provide paths via CLI arguments."
        )

    rows: list[dict] = []
    metadata_models: dict[str, dict] = {}

    for model_name, pred_df, meta in model_predictions:
        metadata_models[model_name] = {
            **meta,
            "prediction_rows": int(len(pred_df)),
            "pred_sequences": int(pred_df["sequence_name"].nunique()),
            "pred_go_terms": int(pred_df["predicted_label"].nunique()),
        }
        print(
            f"[{model_name}] rows={len(pred_df)}, sequences={pred_df['sequence_name'].nunique()}, terms={pred_df['predicted_label'].nunique()}"
        )

        for method in ("union", "gt_only"):
            result = evaluate_model(
                ground_truth=ground_truth,
                predictions=pred_df,
                go_cols=go_cols,
                method=method,
                model_name=model_name,
            )
            rows.append(result)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(by=["model", "method"]).reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(output_csv, index=False)

    metadata = {
        "config": {
            "dataset": args.dataset,
            "go_min_freq": args.go_min_freq,
            "split_mode": args.split_mode,
            "random_state": args.random_state,
            "sample_frac": args.sample_frac,
            "sample_random_state": args.sample_random_state,
            "eval_size": int(len(eval_indices)),
            "go_terms": int(len(go_cols)),
            "export_fasta": args.export_fasta,
        },
        "models": metadata_models,
        "results": rows,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSummary metrics:")
    print(
        metrics_df[
            [
                "model",
                "method",
                "micro_f1",
                "macro_f1",
                "weighted_f1",
                "samples_f1",
                "coverage_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV:  {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
