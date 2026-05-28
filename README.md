# FX Direction Classification

**Author:** Derrick Chua  
**Dataset:** EUR/USD daily OHLC — 5,000 observations (Jan 2007 – Mar 2026)  
**Models:** Logistic Regression · Random Forest · Feedforward Neural Network

---

## What This Project Does

This project builds an end-to-end machine learning pipeline to classify next-day directional movement of the EUR/USD currency pair using engineered features derived from historical OHLC price data.

The goal is not to build a profitable trading system. The goal is to demonstrate a rigorous, leakage-safe ML workflow on real financial time-series data — compare multiple modeling approaches honestly, benchmark every result against a naive baseline, and communicate findings with analytical clarity.

---

## Methodology

**Target design.** The binary label uses a volatility-adjusted threshold: only days where next-day return exceeds ±0.5× ATR-14 are labeled. Neutral days are dropped. This prevents training on noise and produces a near-balanced class distribution (1,007 down / 1,005 up from 5,000 raw rows).

**Leakage prevention.** Every feature is shifted to use only information available at prediction time. Rolling calculations use `.shift(1)` before window computation. No same-day data is used anywhere in the feature set.

**Train/test split.** Strict chronological 80/20 split. No shuffling. No stratification. Test period covers 2022–2026 — a structurally different regime from the training window.

**Baseline.** A naive "always predict up" classifier is evaluated first. Every trained model must beat this floor to justify its complexity. Most do not.

---

## Features

| Feature | Description |
|---|---|
| `return_1d` | Prior day close-to-close return — immediate momentum direction |
| `range_pct` | Prior day high-low range scaled by close — intraday volatility proxy |
| `momentum_5` | 5-day price change ratio — short-term directional persistence |
| `sma_ratio_5_10` | 5-day vs 10-day SMA ratio — trend structure above/below baseline |
| `atr_pct_14` | 14-day ATR scaled by close — current volatility regime |
| `rsi_14` | 14-day RSI — overbought/oversold momentum state |
| `momentum_20` | 20-day price change ratio — medium-term directional trend |
| `instability_flag` | Binary flag for GFC (2008–09) and COVID (2020–21) macro stress periods |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naive Baseline (Always Up) | 0.510 | 0.510 | 1.000 | 0.675 | — |
| Logistic Regression | 0.458 | 0.458 | 0.346 | 0.394 | 0.451 |
| Random Forest | 0.505 | 0.512 | 0.624 | 0.563 | 0.484 |
| Feedforward NN (MLP) | 0.507 | 0.518 | 0.502 | 0.510 | 0.505 |

No trained model beats the naive baseline on accuracy. All three AUC scores are at or below 0.5 — no model achieves meaningful rank-ordering ability over random chance.

**Multi-horizon experiment.** Rerunning the full pipeline with 5-day and 10-day forward targets does not improve signal. All models remain below their respective naive floors at both horizons.

---

## Key Findings

Three meaningfully different architectures — linear, tree-based, and neural network — converge on the same result. That convergence is the finding: the ceiling is in the data and feature design, not the model choice.

**What this analysis does produce:**
- Random Forest feature importances identify RSI-14 (0.180), 1-day return (0.161), and ATR-14 (0.152) as the dominant signals — a direct input to next-iteration feature design.
- The `instability_flag` contributes near-zero importance (0.011). Binary regime flags over multi-year windows are too coarse to be useful splitting features.
- All five incorrect RF predictions in the example set carry confidence within 0.471–0.533, averaging 0.506. The model is not confidently wrong — it has no signal to act on.

**What would improve performance:** Volume data, macroeconomic calendar events, cross-pair correlation features, or alternative target definitions (weekly direction, magnitude-based labels). The pipeline is transferable — the bottleneck is signal richness, not methodology.

---

## Ethical Considerations

This model trained primarily on calm-market conditions will degrade during the next structural break. The instability flag was intended to address this but contributes almost no discriminative power. Any real-world use of a system like this would require multi-regime out-of-sample validation, ongoing performance monitoring, and human review of every signal before action. Those conditions are not met here. This is a research prototype and workflow demonstration — not a decision input in any live context.

---

## Setup

```bash
git clone https://github.com/derrickhabibii/my-final-project-derrickfx.git
cd my-final-project-derrickfx
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook final_project_submission.ipynb
```

The raw data file is included at `data/raw/EURUSD_daily.csv`. No external downloads required.

---

## Repository Structure

```
├── data/
│   └── raw/
│       └── EURUSD_daily.csv          # Source data — do not modify
├── outputs/
│   └── figures/                      # All saved charts (generated on run)
├── final_project_submission.ipynb    # Full submission notebook
└── README.md
```
