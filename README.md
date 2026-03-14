# IMDB Sentiment Analysis - RNN / LSTM / GRU with Ensemble Voting

Binary sentiment classification (positive/negative) on 50,000 IMDB movie reviews
using PyTorch recurrent models and ensemble strategies.

## Overview

### Phase 1 - Model Training & Improvement
| Model | Problem Identified | Fix Applied |
|-------|--------------------|-------------|
| RNN   | Underfitting       | Bidirectional layers + attention mechanism |
| LSTM  | Overfitting        | Aggressive dropout + reduced capacity + layer norm |
| GRU   | Overfitting        | Same strategy as LSTM |

Techniques used across all improved models:
- Multi-position dropout (embedding, recurrent, output)
- Gradient clipping (`clip_grad_norm_`)
- Learning rate scheduling (`ReduceLROnPlateau`)
- Early stopping with model checkpointing

### Phase 2 - Ensemble System
Three voting strategies were evaluated on the test set:
1. **Hard Voting** - majority class wins
2. **Soft Voting** - average of softmax probabilities
3. **Weighted Voting** - models weighted by validation accuracy

## Visualizations
- Word count distribution histogram
- Class balance bar chart
- Positive / Negative word clouds
- Original vs Improved validation loss curves
- Confusion matrices for all 6 models/ensembles

## Dataset
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) - place as `IMDB-Dataset.csv` in your working directory.

## Requirements
torch  pandas  scikit-learn  matplotlib  seaborn  wordcloud  numpy



## Usage
Open `fixed_bxldry3z.ipynb` in Jupyter / Google Colab and run all cells.
GPU recommended (CUDA auto-detected).
