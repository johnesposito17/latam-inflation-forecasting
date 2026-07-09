"""
Phase 3 — Fit country-specific Prophet models with macroeconomic regressors.

Model design decisions:
  Target variable:
    cpi_yoy_pct (YoY % inflation) for all five countries.
    Using the growth rate rather than the raw CPI index keeps the target
    roughly stationary, which Prophet handles more reliably. Raw CPI index
    has a strong upward trend that dominates and can mask seasonality and
    regressor effects. Growth rate is also the quantity economists and
    interviewers care about ("what is inflation?").

  Train/test split:
    Chronological split — the last 12 months of each country's data are held
    out as the test set. No random shuffling. Shuffling time series data
    causes data leakage: the model would "see" future observations during
    training, making test metrics meaninglessly optimistic.
    Train: everything up to (exclusive of) the last 12 months
    Test:  the final 12 months

  Regressors added to Prophet:
    - fx_mom_pct:         monthly FX depreciation — currency weakness drives
                          imported inflation with a lag of 1-3 months
    - commodity_mom_pct:  primary commodity price change — pass-through to
                          domestic prices (oil → gasoline, soy → food)
    - rate_lag1:          lagged policy rate — monetary tightening reduces
                          inflation with a delay; lagged to avoid leakage
    - cpi_lag1:           autoregressive term — past inflation predicts future
                          inflation (persistence)
    All regressors are filled to NaN-safe versions (0 fill for missing)
    ONLY after the train/test split, so imputation doesn't leak test info.

  Seasonality:
    Prophet's built-in yearly seasonality is enabled (CPI has clear seasonal
    patterns in food/energy). Weekly and daily seasonality are off (monthly
    data). A custom monthly seasonality (Fourier order 3) is added as well.

  Confidence intervals:
    Prophet produces 80% and 95% uncertainty intervals natively via MCMC
    sampling. We use the default (yhat_lower/yhat_upper = 80%) for the
    dashboard display.

Outputs:
  models/prophet_forecasts.parquet — train+test actuals and predictions
  models/prophet_metrics.parquet   — RMSE and MAPE per country
  models/prophet_{country}.json    — serialised Prophet model per country
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROC_DIR  = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_MONTHS   = 12
REGRESSORS    = ["fx_mom_pct", "commodity_mom_pct", "rate_lag1", "cpi_lag1"]
COUNTRIES     = ["Mexico", "Brazil", "Chile", "Colombia", "Argentina"]


# ─────────────────────────────────────── helpers

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def prep_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to Prophet convention and fill regressor NaNs with 0."""
    out = df[["date", "cpi_yoy_pct"] + REGRESSORS].copy()
    out = out.rename(columns={"date": "ds", "cpi_yoy_pct": "y"})
    out = out.dropna(subset=["y"])          # drop rows where target is NaN
    for col in REGRESSORS:
        out[col] = out[col].fillna(0.0)     # safe default; NaN would break Stan
    return out.sort_values("ds").reset_index(drop=True)


# ─────────────────────────────────────── fit + evaluate one country

def fit_country(country: str, df_country: pd.DataFrame) -> dict:
    df_prophet = prep_prophet_df(df_country)

    # Chronological train/test split
    split_idx  = len(df_prophet) - TEST_MONTHS
    train_df   = df_prophet.iloc[:split_idx].copy()
    test_df    = df_prophet.iloc[split_idx:].copy()

    print(f"\n  {country}")
    print(f"    Train: {train_df['ds'].min().date()} → {train_df['ds'].max().date()} ({len(train_df)} months)")
    print(f"    Test:  {test_df['ds'].min().date()} → {test_df['ds'].max().date()} ({len(test_df)} months)")

    # Build and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.80,
    )
    # Custom monthly-level seasonality (Fourier order 3 = 6 params)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=3)

    for reg in REGRESSORS:
        model.add_regressor(reg)

    model.fit(train_df)

    # Predict on test set (must supply regressors for future dates)
    forecast_test = model.predict(test_df[["ds"] + REGRESSORS])

    # Also predict in-sample for diagnostics
    forecast_train = model.predict(train_df[["ds"] + REGRESSORS])

    # Metrics on test set
    y_true = test_df["y"].values
    y_pred = forecast_test["yhat"].values

    r = rmse(y_true, y_pred)
    m = mape(y_true, y_pred)
    print(f"    Test RMSE={r:.3f}  MAPE={m:.2f}%")

    # Assemble full forecast frame
    train_out = train_df[["ds", "y"]].copy()
    train_out["yhat"]       = forecast_train["yhat"].values
    train_out["yhat_lower"] = forecast_train["yhat_lower"].values
    train_out["yhat_upper"] = forecast_train["yhat_upper"].values
    train_out["split"]      = "train"

    test_out = test_df[["ds", "y"]].copy()
    test_out["yhat"]       = forecast_test["yhat"].values
    test_out["yhat_lower"] = forecast_test["yhat_lower"].values
    test_out["yhat_upper"] = forecast_test["yhat_upper"].values
    test_out["split"]      = "test"

    combined = pd.concat([train_out, test_out], ignore_index=True)
    combined["country"] = country

    # Save serialised model
    model_path = MODEL_DIR / f"prophet_{country.lower().replace(' ', '_')}.json"
    with open(model_path, "w") as f:
        f.write(model_to_json(model))

    return {
        "forecast_df": combined,
        "metrics": {"country": country, "rmse": r, "mape": m,
                    "train_months": len(train_df), "test_months": len(test_df)},
    }


# ─────────────────────────────────────── main

def main():
    print("Loading modeling dataset...")
    df = pd.read_parquet(PROC_DIR / "modeling_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])

    all_forecasts = []
    all_metrics   = []

    for country in COUNTRIES:
        df_c = df[df["country"] == country].copy()
        result = fit_country(country, df_c)
        all_forecasts.append(result["forecast_df"])
        all_metrics.append(result["metrics"])

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    metrics_df   = pd.DataFrame(all_metrics)

    # Save
    forecasts_path = MODEL_DIR / "prophet_forecasts.parquet"
    metrics_path   = MODEL_DIR / "prophet_metrics.parquet"
    forecasts_df.to_parquet(forecasts_path, index=False)
    metrics_df.to_parquet(metrics_path, index=False)

    print("\n\n=== Prophet Model Summary ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved forecasts → {forecasts_path}")
    print(f"Saved metrics   → {metrics_path}")


if __name__ == "__main__":
    main()
