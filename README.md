# FX Direction Classification — Final Submission

**Student:** Derrick Chua
**Project type:** Academic machine learning project
**Dataset:** EUR/USD daily OHLC, 5,000 observations (2007–2026)

---

## Project Description

This project builds an end-to-end machine learning pipeline to predict the next-day directional movement of the EUR/USD currency pair using engineered technical features derived from historical OHLC data. The goal is not to build a profitable trading system — it is to demonstrate a rigorous, leakage-safe ML workflow on real financial time-series data, compare multiple modeling approaches honestly, and communicate findings with the analytical clarity expected of a professional data scientist.

---

## Setup Instructions

1. Clone the repository
2. Install dependencies:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```
3. Open `final_project_draft.ipynb` in Jupyter
4. Restart the kernel and run all cells from top to bottom

The raw data file is included at `data/raw/EURUSD_daily.csv`. No external data download is required.

---

## Features Engineered

| Feature | Description |
|---|---|
| `return_1d` | Prior day percentage return |
| `range_pct` | Prior day high-low range scaled by close |
| `momentum_5` | 5-day price momentum |
| `sma_ratio_5_10` | Ratio of 5-day to 10-day simple moving average |
| `atr_pct_14` | 14-day average true range scaled by prior close |
| `rsi_14` | 14-day RSI-style momentum strength indicator |
| `momentum_20` | 20-day price momentum |
| `instability_flag` | Binary flag for known macro stress periods (GFC 2008–09, COVID 2020–21) |

All features use only data available at prediction time. No lookahead. No same-day data.

---

## Models Compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Baseline (Always Up) | 0.510 | 0.510 | 1.000 | 0.675 |
| Logistic Regression | 0.458 | 0.458 | 0.346 | 0.394 |
| Random Forest | 0.505 | 0.512 | 0.624 | 0.563 |
| Feedforward NN (MLP) | 0.507 | 0.518 | 0.502 | 0.510 |

Train/test split is chronological (80/20). Test period: 2022–2026.

---

## Key Findings

No trained model beats the naive baseline on accuracy. All three models score at or below 0.5 AUC, indicating no meaningful rank-ordering ability. Longer prediction horizons (5-day, 10-day) do not improve performance.

The primary finding is that daily EUR/USD OHLC features, as engineered here, do not contain sufficient signal to reliably predict next-day direction. This is consistent across all model architectures tested. The contribution of this project is the methodology — leakage-safe feature engineering, chronological evaluation, and honest multi-model comparison against a naive floor — not the accuracy numbers.

Random Forest is the recommended model for further development because it provides feature importance rankings (top features: RSI-14, 1-day return, ATR). Those rankings inform what signals to expand in a next iteration, even when overall accuracy is weak.

---

## Limitations

- OHLC features alone do not provide sufficient directional signal at daily granularity
- No volume data available in source dataset
- Single market pair, single holdout window — results may not generalize
- `instability_flag` contributes near-zero importance (0.011) — regime flagging requires a more continuous approach to be useful

---

## Repository Structure

```
├── data/raw/EURUSD_daily.csv        # Source data — do not modify
├── final_project_submission.ipynb   # Full submission notebook
├── outputs/figures/                 # All saved charts
└── README.md
```
