# LATAM Inflation Forecasting

An end-to-end time series forecasting pipeline for monthly CPI (Consumer Price Index) across five Latin American economies — **Mexico, Brazil, Chile, Colombia, and Argentina** — using macroeconomic signals and two complementary models: Meta's Prophet and a stacked LSTM.

Built as a portfolio project targeting data science and analytics roles.

---

## Project Overview

| Layer | Tool | Purpose |
|---|---|---|
| Data sourcing | Python + World Bank API | Pull CPI, interest rates, FX rates, commodity prices |
| Feature engineering | pandas | Lags, MoM/YoY deltas, rate changes |
| Baseline model | Prophet | Country-specific models with macro regressors |
| Deep learning model | PyTorch LSTM | Stacked 2-layer LSTM on engineered sequences |
| Dashboard | Plotly Dash | Interactive forecast explorer with country/horizon toggles |

---

## Countries Covered

| Country | Currency | Key Commodity | Notes |
|---|---|---|---|
| Mexico | MXN | Oil (WTI/Brent) | Significant remittance and oil export exposure |
| Brazil | BRL | Soybean | Largest LATAM economy; history of high inflation |
| Chile | CLP | Copper | Most commodity-concentrated economy in the set |
| Colombia | COP | Oil (Brent) | Dollarized oil revenues affect domestic prices |
| Argentina | ARS | Soybean | Structural hyperinflation; highest forecast difficulty |

---

## Data Sources

- **World Bank API** — CPI index, interest rates, FX rates (wbgapi)
- **FRED (Federal Reserve Bank of St. Louis)** — commodity prices where World Bank lacks monthly frequency
- **IMF International Financial Statistics (IFS)** — monthly CPI fallback where World Bank only has annual data

*See `docs/NOTES.md` for substitutions made and why.*

---

## Repo Structure

```
latam-inflation-forecasting/
├── data/
│   ├── raw/          # Downloaded source files (gitignored)
│   └── processed/    # Cleaned, merged modeling dataset (gitignored)
├── models/           # Saved model weights and forecast outputs
├── scripts/          # Data pull, cleaning, modeling, evaluation scripts
├── notebooks/        # Exploratory and validation notebooks
├── dashboard/        # Plotly Dash app
├── docs/             # Technical notes and model comparison
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.12
- FRED API key (free): <https://fred.stlouisfed.org/docs/api/api_key.html>

### Install

```bash
git clone https://github.com/johnesposito17/latam-inflation-forecasting.git
cd latam-inflation-forecasting
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your FRED_API_KEY
```

### Run the pipeline

```bash
python3 scripts/pull_cpi_data.py
python3 scripts/pull_macro_indicators.py
python3 scripts/validate_data.py
python3 scripts/build_features.py
python3 scripts/train_prophet.py
python3 scripts/train_lstm.py
python3 scripts/compare_models.py
```

### Launch dashboard

```bash
python3 dashboard/app.py
# Open http://localhost:8050
```

---

## Key Findings

> _To be completed after modeling (Phase 5)_

---

## Dashboard

> _Screenshot / GIF to be added after Phase 6_

**Run locally:** `python3 dashboard/app.py` → <http://localhost:8050>
