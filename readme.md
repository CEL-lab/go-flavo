# GO-Flavo: Multi-Label Gene Ontology Term Prediction for Flavobacterium Proteins

This repository contains the implementation and experiments for the research paper: **"A Comparative Study of Boosting, Deep Learning, and Graph Neural Networks for Multi-Label GO Term Prediction in Flavobacterium Proteins"** by M Mishkatur Rahman, Yusuf Akbulut, and Harun Pirim.

## Overview

This project implements and compares three different machine learning approaches for predicting Gene Ontology (GO) terms in Flavobacterium proteins:
- **XGBoost** (Gradient Boosting)
- **Deep Neural Networks (DNN)**
- **Graph Neural Networks (GNN)**

The study focuses on multi-label classification where each protein can be associated with multiple GO terms across three categories: Biological Process (BP), Cellular Component (CC), and Molecular Function (MF).

## Dataset

The dataset consists of **9,947 Flavobacterium protein sequences** with their corresponding GO term annotations. The proteins are annotated with GO terms that have a minimum frequency of 4 occurrences in the dataset.

### Data Files

- `final_dataset.csv` - Complete dataset with protein sequences and GO annotations
- `final_dataset_frequency_4.csv` - Filtered dataset with GO terms having frequency ≥ 4
- `sequence_graph.graphml` - Protein similarity graph for GNN approach
- `deep_go_se_*.tsv` - DeepGOse baseline predictions for BP, CC, and MF
- `proteinfer.tsv` - Proteinfer baseline predictions
- `netG0_*.txt` - NetGO baseline predictions

## Methods

### 1. XGBoost Approach
- Uses ESM (Evolutionary Scale Modeling) protein embeddings
- Gradient boosting classifier for multi-label prediction
- Optimized hyperparameters through grid search

### 2. Deep Neural Network (DNN)
- Multi-layer feedforward neural network
- Input: ESM embeddings and additional protein features
- Multi-label output with sigmoid activation
- Dropout regularization and early stopping

### 3. Graph Neural Network (GNN)
- Constructs protein similarity graph based on sequence homology
- Uses Graph Convolutional Networks (GCN) for node classification
- Incorporates both node features (ESM embeddings) and graph structure
- Message passing for learning protein relationships

## File Structure

```
├── GNN_Flavo_Final.ipynb          # Graph Neural Network implementation
├── XGBoost_DNN_flavo.ipynb        # XGBoost and DNN implementations  
├── ablation.ipynb                 # Ablation study comparing feature sets
├── compare_results.py             # Evaluation script for different methods
├── data/                          # Dataset and baseline predictions
│   ├── final_dataset.csv          # Complete protein dataset
│   ├── final_dataset_frequency_4.csv  # Filtered dataset
│   ├── sequence_graph.graphml     # Protein similarity graph
│   ├── deep_go_se_*.tsv          # DeepGOse baseline predictions
│   ├── proteinfer.tsv            # Proteinfer baseline predictions
│   └── netG0_*.txt               # NetGO baseline predictions
└── readme.md                     # This file
```

## Baseline Comparisons

The study compares against established protein function prediction tools:
- **Proteinfer**: Sequence-based GO term prediction
- **DeepGOse**: Deep learning approach for GO prediction
- **NetGO**: Network-based GO term prediction

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{go_flavo_2025,
  title={A Comparative Study of Boosting, Deep Learning, and Graph Neural Networks for Multi-Label GO Term Prediction in Flavobacterium Proteins},
  author={Rahman, M Mishkatur and Akbulut, Yusuf and Pirim, Harun},
  journal={[Not yet published]},
  year={2025}
}
```

## Acknowledgments

- ESM embeddings from Facebook AI Research
- GO annotations from Gene Ontology Consortium
- Baseline methods from respective authors