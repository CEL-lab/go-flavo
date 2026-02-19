#!/usr/bin/env python3
"""
Modular script to evaluate GO term predictions using multiple methods and datasets.
Prints six classification reports (union and gt_only) for Proteinfer, DeepGOse, and NetGO.
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*unknown class.*",
    category=UserWarning,
    module=r"sklearn\.preprocessing\._label"
)
def evaluate_go_predictions(
    df: pd.DataFrame,
    pdf: pd.DataFrame,
    method: str = 'gt_only'
) -> None:
    """
    Evaluate GO term predictions and print a classification report.

    Parameters
    ----------
    df : pd.DataFrame
        Ground truth DataFrame with one-hot encoded GO term columns and sequence identifiers as the index.
    pdf : pd.DataFrame
        Predictions DataFrame with 'sequence_name' and 'predicted_label' columns.
    method : str
        'gt_only' to use only ground-truth GO terms, 'union' to include predicted-only labels.
    """
    # Identify GO columns
    go_columns = [col for col in df.columns if col.startswith('GO:')]
    if not go_columns:
        raise ValueError(
            "No GO-term columns found in the ground truth DataFrame."
        )

    # Build true labels list
    df_copy = df.copy()
    df_copy['true_go_terms'] = (
        df_copy[go_columns]
        .apply(lambda row: list(row.index[row == 1]), axis=1)
    )
    df_copy.index = df_copy.index.astype(str)

    # Merge and group
    merged = pdf.merge(
        df_copy[['true_go_terms']],
        left_on='sequence_name',
        right_index=True,
        how='left'
    )
    grouped = merged.groupby('sequence_name').agg({
        'predicted_label': lambda labels: list(labels),
        'true_go_terms' : 'first'
    }).reset_index()

    # Determine classes
    if method == 'gt_only':
        classes = go_columns
    elif method == 'union':
        pred_terms = {term for labels in grouped['predicted_label'] for term in labels}
        classes = sorted(set(go_columns) | pred_terms)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'gt_only' or 'union'.")

    mlb = MultiLabelBinarizer(classes=classes)
    y_true = mlb.fit_transform(grouped['true_go_terms'])
    y_pred = mlb.transform(grouped['predicted_label'])
    
    # get full report as dict
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )
    
    # define the rows we want, in the order we want them
    summary_keys = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
    
    # print header
    print(f"{'':15s} {'precision':>8s} {'recall':>8s} {'f1-score':>8s} {'support':>8s}")
    # print each summary line
    for key in summary_keys:
        row = report_dict.get(key, {})
        p = row.get('precision', 0.0)
        r = row.get('recall',    0.0)
        f = row.get('f1-score',  0.0)
        s = int(row.get('support', 0))
        print(f"{key:15s} {p:8.2f} {r:8.2f} {f:8.2f} {s:8d}")


def load_ground_truth(path: str, sample_frac: float = 0.2, random_state: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def load_proteinfer(path: str) -> pd.DataFrame:
    pdf = pd.read_csv(path, sep='\t')
    pdf = pdf[pdf['predicted_label'].str.startswith('GO:')].copy()
    pdf['sequence_name'] = pdf['sequence_name'].str.replace('sequence_', '', regex=False)
    return pdf[['sequence_name', 'predicted_label']]


def load_deepgose(bp_path: str, cc_path: str, mf_path: str, threshold: float = 0.5) -> pd.DataFrame:
    cols = ['sequence_name', 'predicted_label', 'confidence']
    go_bp = pd.read_csv(bp_path, sep='\t', names=cols)
    go_cc = pd.read_csv(cc_path, sep='\t', names=cols)
    go_mf = pd.read_csv(mf_path, sep='\t', names=cols)
    go_all = pd.concat([go_bp, go_cc, go_mf], axis=0)
    df = go_all[go_all['confidence'] > threshold].copy()
    df['sequence_name'] = df['sequence_name'].str.replace('sequence_', '', regex=False)
    return df[['sequence_name', 'predicted_label']]


def load_netgo(paths: list, threshold: float = 0.5) -> pd.DataFrame:
    cols = ['sequence_name', 'predicted_label', 'confidence', 'aspect', 'description']
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t', header=None, names=cols)
        dfs.append(df)
    go_all = pd.concat(dfs, axis=0)
    df = go_all[go_all['confidence'] > threshold].copy()
    df['sequence_name'] = df['sequence_name'].str.replace('sequence_', '', regex=False)
    return df[['sequence_name', 'predicted_label']]


def main():
    # Load and sample ground truth
    ground_truth = load_ground_truth('./data/final_dataset.csv')

    # Proteinfer
    prot = load_proteinfer('./data/proteinfer.tsv')
    print("##### Proteinfer: union #####")
    evaluate_go_predictions(ground_truth, prot, method='union')
    print("##### Proteinfer: gt_only #####")
    evaluate_go_predictions(ground_truth, prot, method='gt_only')

    # DeepGOse
    deep = load_deepgose(
        './data/deep_go_se_bp.tsv',
        './data/deep_go_se_cc.tsv',
        './data/deep_go_se_mf.tsv'
    )
    print("##### DeepGOse: union #####")
    evaluate_go_predictions(ground_truth, deep, method='union')
    print("##### DeepGOse: gt_only #####")
    evaluate_go_predictions(ground_truth, deep, method='gt_only')

    # NetGO
    net = load_netgo([
        './data/netG0_1.txt',
        './data/netG0_2.txt'
    ])
    print("##### NetGO: union #####")
    evaluate_go_predictions(ground_truth, net, method='union')
    print("##### NetGO: gt_only #####")
    evaluate_go_predictions(ground_truth, net, method='gt_only')


if __name__ == '__main__':
    main()
