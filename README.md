# FX Direction Classification Draft

This repository contains Derrick's academic machine learning draft project for predicting meaningful next-day EUR/USD direction from daily OHLC price data. The current draft uses leakage-safe feature engineering, a chronological train/test split, and two classification models: Logistic Regression and Random Forest.

The goal of this submission is to demonstrate a working prototype for instructor feedback before final optimization. It is a decision-support notebook, not a live trading system.

## Dataset

- Source file: `data/raw/EURUSD_daily.csv`
- Original dataset source: Yahoo Finance — EUR/USD historical data (`EURUSD=X`)
- Original dataset source link: https://finance.yahoo.com/quote/EURUSD=X/history/
- Dataset type: daily EUR/USD OHLC price history
- Rows: 5,000
- Columns: `timestamp`, `open`, `high`, `low`, `close`
- Date range: 2007-01-16 to 2026-03-17
- Notes:
  - No volume column is present
  - Raw data contains no missing values
  - The raw CSV is stored newest-to-oldest, so the notebook sorts it chronologically before any feature engineering

## Repository Structure

- `final_project_draft.ipynb`: main submission notebook
- `data/raw/EURUSD_daily.csv`: dataset used in the draft
- `README.md`: project overview, dataset details, run steps, and current results

## How To Run

1. Open this repository in VS Code or Jupyter.
2. Make sure Python 3 with the following libraries is installed:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
3. Run the notebook `final_project_draft.ipynb` from top to bottom.
4. The notebook generates and displays the project charts during execution.

## Current Draft Scope

The notebook currently includes:

- data loading and exploratory analysis
- target construction using a volatility-adjusted threshold
- five leakage-safe engineered features, with three emphasized for the draft requirement
- preprocessing with chronological split and train-only scaling for Logistic Regression
- two complete models: Logistic Regression and Random Forest
- side-by-side comparison table
- confusion matrix and training/validation visualization
- written analysis, interpretation, and next-step reflection

## Current Results Summary

Draft modeling is currently based on 2,012 labeled observations after removing neutral days from the ATR-based target rule.

Preliminary test-set results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.489 | 0.500 | 0.500 | 0.500 |
| Random Forest | 0.494 | 0.504 | 0.636 | 0.562 |

Current takeaway:

- Random Forest is the stronger draft model because it captures more positive cases and achieves the best F1 score.
- Logistic Regression remains useful as an interpretable baseline.
- Further feature engineering and threshold tuning are likely needed before the final submission.
