# LATAM Inflation Forecasting

An end-to-end time series forecasting pipeline for monthly CPI (Consumer Price Index) across five Latin American economies — **Mexico, Brazil, Chile, Colombia, and Argentina** — using macroeconomic signals and two complementary models: Meta's Prophet and a stacked LSTM.

Built as a portfolio project targeting data science and analytics roles.

---

## Project Overview

| Layer | Tool | Purpose |
|---|---|---|
| Data sourcing | Python + OECD SDMX + FRED API | Pull CPI, FX rates, policy rates, commodity prices |
| Feature engineering | pandas | Lags, MoM/YoY deltas, rate changes, commodity assignment |
| Baseline model | Prophet | Country-specific models with macro regressors |
| Deep learning model | PyTorch LSTM | Stacked 2-layer LSTM, pooled across all 5 countries |
| Evaluation | scikit-learn | RMSE and MAPE on held-out 12-month test period |
| Dashboard | Plotly Dash | Interactive forecast explorer |

---

## Countries Covered

| Country | Currency | Key Commodity | Notes |
|---|---|---|---|
| Mexico | MXN | Oil (WTI) | Significant remittance and oil export exposure |
| Brazil | BRL | Soybean | Largest LATAM economy; history of high inflation |
| Chile | CLP | Copper | Most commodity-concentrated economy in the set |
| Colombia | COP | Oil (WTI) | Dollarised oil revenues affect domestic prices |
| Argentina | ARS | Soybean | Structural hyperinflation; highest forecast difficulty |

---

## Key Findings

### Model Comparison (test period = last 12 months)

| Country | Prophet MAPE | LSTM MAPE | Better model |
|---|---|---|---|
| Mexico | 11.0% | 12.0% | Prophet |
| Brazil | 12.9% | 5.0% | **LSTM** |
| Chile | 7.0% | 6.7% | Tie |
| Colombia | 12.2% | 21.3% | Prophet |
| Argentina | 192.6% | 36.2% | **LSTM** |

**Argentina** had the worst forecast accuracy in both models, consistent with its history of hyperinflation and structural economic instability. The 2025 test period coincided with a rapid disinflation (inflation fell from ~150% → 31%) that had no precedent in the training window — a structural break that neither model could fully anticipate.

**Brazil** and **Argentina** were best served by the LSTM, which benefited from cross-country transfer learning (pooled model). The LSTM likely borrowed Brazil's 1990s disinflation dynamics to partially predict Argentina's 2025 stabilization.

**Mexico** and **Colombia** were better handled by Prophet, which captures explicit seasonality and regressor effects cleanly for economies with more stable inflation regimes.

**Chile** was a near-tie — the most macro-stable economy in the set and the easiest to forecast with either approach.

---

## Data Sources

- **OECD SDMX API** (`sdmx.oecd.org`) — monthly CPI index for MEX, BRA, CHL, COL; YoY% for ARG (no API key required)
- **FRED** (Federal Reserve Bank of St. Louis) — monthly FX rates, policy rates, commodity prices (free API key required)

*See `docs/NOTES.md` for full source substitution rationale and data decisions.*

---

## Repo Structure

```
latam-inflation-forecasting/
├── data/
│   ├── raw/          # Downloaded source files (gitignored)
│   └── processed/    # Cleaned, merged modeling dataset (gitignored)
├── models/           # Saved model weights, forecasts, and metrics
├── scripts/
│   ├── pull_cpi_data.py          # Phase 1: CPI pull (OECD)
│   ├── pull_macro_indicators.py  # Phase 1: FX, rates, commodities (FRED)
│   ├── validate_data.py          # Phase 1: coverage and range checks
│   ├── build_features.py         # Phase 2: merge + feature engineering
│   ├── train_prophet.py          # Phase 3: Prophet models
│   ├── train_lstm.py             # Phase 4: stacked LSTM
│   └── compare_models.py         # Phase 5: model comparison table
├── dashboard/
│   └── app.py        # Phase 6: Plotly Dash app
├── docs/
│   └── NOTES.md      # Technical decisions and interview notes
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- FRED API key (free): <https://fred.stlouisfed.org/docs/api/api_key.html>

### Install

```bash
git clone https://github.com/johnesposito17/latam-inflation-forecasting.git
cd latam-inflation-forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your FRED API key
```

### Run the pipeline

```bash
python3 scripts/pull_cpi_data.py          # pulls CPI data (no key needed)
python3 scripts/pull_macro_indicators.py  # pulls FX, rates, commodities (FRED key needed)
python3 scripts/validate_data.py          # checks coverage and value ranges
python3 scripts/build_features.py         # merges and engineers features
python3 scripts/train_prophet.py          # fits Prophet models (~30 seconds)
python3 scripts/train_lstm.py             # trains LSTM (~1 minute)
python3 scripts/compare_models.py         # prints comparison table
```

### Launch dashboard

```bash
python3 dashboard/app.py
# Open http://localhost:8050
```

---

## Dashboard

The dashboard lets you:
- Select a country (dropdown)
- Select a forecast horizon (3 / 6 / 12 months)
- Toggle between Prophet, LSTM, or both on the same chart
- View historical CPI YoY% + test-period forecasts with Prophet confidence intervals
- See RMSE and MAPE metrics with the better model highlighted in green
