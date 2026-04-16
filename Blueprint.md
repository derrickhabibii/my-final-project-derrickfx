# FX Signal Prototype — Blueprint

**Derrick Chua | Machine Learning Portfolio Project**

---

## 1. Problem Statement

Short term FX moves are noisy and hard to forecast. Traders, analysts, and treasury teams still need structured ways to read direction, volatility, and market bias, but simple rules don't cut it.

This project tackles one manageable slice of that: predicting whether a currency pair like EUR/USD closes up or down in the next period, using historical price data and engineered technical features.

ML fits here because market behavior is nonlinear and driven by multiple interacting patterns, not a single clean formula. Descriptive stats can summarize the past. They can't test whether combinations of lagged returns, moving averages, volatility, and momentum have repeatable predictive value. That's what classification modeling is for.

The target user isn't a trading bot. It's a researcher or analyst who wants additional signal context before making a judgment call. Success means reproducible performance, interpretable features, a clear baseline comparison, and a workflow professional enough to hold up as a portfolio case study.

---

## 2. Dataset

**Starting point:** Daily EUR/USD historical data (Yahoo Finance, ticker: `EURUSD=X`)

One pair for the class submission. Then test transferability on GBP/USD or USD/JPY once the base pipeline is solid.

| Field | Detail |
|---|---|
| Primary source | Yahoo Finance EUR/USD historical data |
| Observation unit | One trading day per row |
| Raw columns | Date, Open, High, Low, Close (Adj Close / Volume if available) |
| Engineered features | Daily return, rolling volatility, short MA, long MA, MA gap, momentum, RSI-style strength, lagged returns, high-low range, volatility regime flag |
| Target variable | Binary: 1 if next period closes higher, 0 otherwise |
| Quality notes | Sort chronologically, handle missing trading days, drop NaN rows from rolling windows, avoid time leakage on splits |

**Sources:** [Yahoo Finance EUR/USD](https://finance.yahoo.com/quote/EURUSD%3DX/history/) | [Alpha Vantage FX docs](https://www.alphavantage.co/documentation/)

**Requirement check:**
- Observations: well above 1,000 (multi-year daily history)
- Features: 10+ after feature engineering
- Target: clean binary label for next-period direction
- Business relevance: FX directional support has real use in trading and treasury settings

---

## 3. ML Approaches

Two meaningfully different models, plus a naive baseline.

| Model | Why it fits | What it adds | Main risk |
|---|---|---|---|
| Random Forest classifier | Handles tabular data and nonlinear relationships well | Strong interpretable baseline, feature importance view | Can overfit noisy market data if the split isn't time-aware |
| Feedforward neural network | More flexible nonlinear architecture for the same binary target | Tests whether added capacity improves directional prediction | Needs scaling and tuning; may not beat tree models on limited signal |
| Naive baseline | Always-up or previous-direction continuation | Shows whether ML is actually adding value over simple guessing | May expose that markets are hard to beat in any stable way |

Evaluation: accuracy, precision, recall, F1, confusion matrix behavior, and stability on a time-aware split. No single headline metric.

---

## 4. Portfolio Framing

The portfolio examples from the course all point to the same thing: the best ML projects aren't memorable because of the algorithm. They're memorable because the author framed a real problem, explained why the modeling choice fits, documented the workflow clearly, and translated technical output into decisions that matter.

That shaped how this project is positioned. The pitch isn't "a model that beats forex." It's a full end-to-end ML case study on directional forecasting under uncertainty, including feature engineering, baseline comparison, interpretability, and honest discussion of limitations. That's more credible to professors, employers, and grad programs because it shows judgment, not just code.

What this project demonstrates: working with messy real-world time-series data, engineering domain-informed features, comparing multiple ML approaches, evaluating model quality without overselling, and writing about deployment risk honestly.

---

## 5. Capstone Expansion Plan

The next version of this moves from a single-pair offline classifier toward a broader FX decision-support system.

**Expansion 1 — Multi-pair testing:** Run the same pipeline on EUR/USD, GBP/USD, and USD/JPY. Check if the feature logic generalizes or if each pair needs a more specialized model.

**Expansion 2 — Market condition filter:** Flag high-uncertainty periods (abnormal volatility, major news windows) where the safer recommendation is to stay out of the market.

**Longer term:** Richer data sources, macroeconomic indicators, event calendars, sentiment features, sequence models, regime detection. The goal isn't a "buy or sell" engine. It's a system that knows when signal quality is weak, communicates risk clearly, and supports human decision making rather than pretending to eliminate uncertainty.

---

## Appendix: Data Acquisition Phases

| Phase | Task |
|---|---|
| 1 | Pull historical data for EUR/USD |
| 2 | Clean data and engineer technical features |
| 3 | Create the binary target label for next-period direction |
| 4 | Split data chronologically into train / validation / test |
| 5 | Train Random Forest, Neural Network, and naive baseline |
| 6 | Compare metrics and inspect overfitting |
| 7 | Rerun the same pipeline on GBP/USD or USD/JPY for out-of-pair comparison |

A model that looks good on one pair may just be overfitting that pair's structure. Testing other pairs answers the more important question: is the workflow learning something general, or is it memorizing local behavior?