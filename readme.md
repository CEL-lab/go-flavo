# GO-Flavo: Multi-Label Gene Ontology Term Prediction for Flavobacterium Proteins

This repository contains the implementation and experiments for the research paper: **"A Comparative Study of Boosting, Deep Learning, and Graph Neural Networks for Multi-Label Gene Ontology Term Prediction in Flavobacterium Proteins"**, submitted to Engineering Applications of Artificial Intelligence (EAAI-25-16789).

Authors: Yusuf Akbulut, M Mishkatur Rahman, and Harun Pirim.

## Overview

This project develops and compares taxon-specific machine learning models for predicting Gene Ontology (GO) terms in Flavobacterium proteins. Five models are trained and evaluated:

- **XGBoost** -- gradient-boosted decision trees (traditional ML baseline)
- **DNN** -- fully connected deep neural network
- **GAT** -- Graph Attention Network
- **GCN** -- Graph Convolutional Network
- **GraphSAGE** -- inductive graph learning via neighbor sampling

Models are benchmarked against three general-purpose GO predictors (ProteInfer, DeepGO-SE, NetGO 4.0) to quantify the advantage of taxon-specific training.

## Dataset

- **279 UniProt reference/representative Flavobacterium proteomes**
- **35,554 protein sequences** after quality filtering (annotation score >= 3, no ambiguous residues, length <= 1,022)
- **762 GO terms** (minimum frequency 4) across Molecular Function, Biological Process, and Cellular Component
- **70/15/15** stratified multi-label train/validation/test split

### Features

| Feature set | Dimensions | Source |
|---|---|---|
| BioPython | 29 | Amino acid composition, molecular weight, pI, aromaticity, instability, GRAVY, secondary structure fractions |
| ESM-2 embeddings | 320 | `facebook/esm2_t6_8M_UR50D` mean-pooled last hidden state |
| **Total** | **349** | |

### Graph construction

A protein similarity graph is constructed from pairwise cosine similarity of ESM-2 embeddings at threshold 0.99, yielding 2.8M edges with label homophily 0.92.

## Key Results

| Model | Micro F1 | Macro F1 | Weighted F1 |
|---|---|---|---|
| **DNN** | **0.93** | **0.89** | **0.93** |
| XGBoost | 0.90 | 0.80 | 0.89 |
| GraphSAGE | 0.81 | 0.81 | 0.85 |
| GAT | 0.78 | 0.78 | 0.84 |
| GCN | 0.77 | 0.79 | 0.84 |
| ProteInfer | 0.49 | 0.52 | 0.65 |
| NetGO 4.0 | 0.35 | 0.47 | 0.48 |
| DeepGO-SE | 0.19 | 0.07 | 0.13 |

DNN advantage over XGBoost confirmed by paired bootstrap test (B=10,000): Micro F1 +0.034 [95% CI: +0.030, +0.038].

## Repository Structure

```
./
├── 0_data_prep.ipynb           # Proteome download, filtering, feature extraction
├── 1_XGBoost.py                # XGBoost multi-label training and evaluation
├── 2_DNN.py                    # DNN training and evaluation
├── 3_compare_results.py        # Comparison with general-purpose baselines
├── 5_per_go_analysis.py        # Per-GO-term and per-category F1 analysis
├── 6_ablation.py               # Feature ablation (All / ESM-2 / BioPython)
├── 7_statistical_analysis.py   # Paired bootstrap significance testing
├── 8_heatmap.py                # F1 deviation heatmap visualization
├── gnn/
│   ├── build_graph.py          # ESM-2 cosine similarity graph construction
│   ├── data_utils.py           # PyG data object creation and splitting
│   └── train_gnn.py            # GAT / GCN / GraphSAGE training
└── readme.md                   # This file
```

## Requirements

- Python 3.10+
- PyTorch, PyTorch Geometric
- Transformers (HuggingFace) for ESM-2
- XGBoost, scikit-learn, BioPython
- pandas, numpy, matplotlib, seaborn

## Citation

```bibtex
@article{go_flavo_2026,
  title={A Comparative Study of Boosting, Deep Learning, and Graph Neural Networks
         for Multi-Label Gene Ontology Term Prediction in Flavobacterium Proteins},
  author={Akbulut, Yusuf and Rahman, M Mishkatur and Pirim, Harun},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  note={Under review}
}
```

## Acknowledgments

This research is supported by USDA NIFA grant 2021-67015-34532.
