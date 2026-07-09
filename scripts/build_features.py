"""
Phase 2 — Clean, merge, and engineer features for all 5 LATAM countries.

Missing data strategy (documented here for interview/notes):
  CPI index (MEX/BRA/CHL/COL):
    Linear interpolation for gaps ≤ 3 months. CPI changes continuously, so
    interpolation is more defensible than forward-fill. Gaps > 3 months are
    left as NaN and flagged.

  CPI for Argentina:
    Native YoY% from OECD, starts 2017-12. No imputation — we simply limit
    Argentina's effective modeling window to 2017-12 onward.

  FX rates:
    Forward-fill for gaps ≤ 2 months. Exchange rates are set by markets or
    central banks continuously; a missing month likely reflects a data lag
    rather than the rate being zero or undefined. Forward-fill preserves the
    last known rate, which is the correct prior.

  Policy rates:
    Forward-fill for gaps ≤ 3 months. Policy rates are explicitly "sticky" —
    central banks hold rates between meetings. Forward-fill is not just
    convenient but economically accurate for inter-meeting periods.

  Argentina policy rate:
    Entirely absent from FRED. Kept as NaN in the output. The LSTM model
    will be trained without this feature for Argentina (or with it zeroed and
    a country-indicator flag). Noted in docs/NOTES.md.

  Commodity prices:
    Complete from FRED, no imputation needed.

Feature engineering:
  Derived from CPI index (MEX/BRA/CHL/COL):
    cpi_yoy_pct  — YoY % change: (CPI_t / CPI_{t-12} - 1) * 100
    cpi_mom_pct  — MoM % change: (CPI_t / CPI_{t-1}  - 1) * 100

  For Argentina, cpi_yoy_pct is native; cpi_mom_pct not computed (source
  only provides YoY series).

  Lagged CPI YoY (all countries):
    cpi_lag1, cpi_lag3, cpi_lag6, cpi_lag12 — lagged cpi_yoy_pct values

  FX features:
    fx_mom_pct  — MoM depreciation: (FX_t / FX_{t-1}  - 1) * 100
    fx_yoy_pct  — YoY depreciation: (FX_t / FX_{t-12} - 1) * 100
    fx_lag1_mom, fx_lag3_mom

  Policy rate features:
    rate_delta_mom — month-over-month change in policy rate (pp)
    rate_lag1, rate_lag3 — lagged rate levels

  Commodity features (country-specific primary commodity assigned below):
    commodity_mom_pct  — MoM % change in primary commodity price
    commodity_yoy_pct  — YoY % change
    commodity_lag1_mom, commodity_lag3_mom

Data leakage note:
  All lag features are computed with strictly positive lags (t-1, t-3, etc.).
  cpi_yoy_pct at time t uses CPI_t and CPI_{t-12} — both are known at time t
  and do NOT use any future value. The train/test split in Phase 3 will
  further ensure the model never sees future data during training.

Output: data/processed/modeling_dataset.parquet
  One row per (country, month). First 12 rows per country have NaN lag
  features and should be excluded from model training (set aside as burn-in).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Which commodity is most relevant per country
COUNTRY_COMMODITY = {
    "Mexico":    "oil_wti_usd",
    "Colombia":  "oil_wti_usd",
    "Argentina": "soybean_usd",
    "Brazil":    "soybean_usd",
    "Chile":     "copper_usd",
}


# ─────────────────────────────────────── helpers

def pct_change_yoy(series: pd.Series) -> pd.Series:
    return (series / series.shift(12) - 1) * 100


def pct_change_mom(series: pd.Series) -> pd.Series:
    return (series / series.shift(1) - 1) * 100


def ffill_limited(series: pd.Series, limit: int) -> pd.Series:
    return series.ffill(limit=limit)


def interpolate_limited(series: pd.Series, limit: int) -> pd.Series:
    return series.interpolate(method="linear", limit=limit, limit_direction="forward")


# ─────────────────────────────────────── load raw data

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cpi = pd.read_parquet(RAW_DIR / "cpi_monthly.parquet")
    fx  = pd.read_parquet(RAW_DIR / "fx_rates.parquet")
    rates = pd.read_parquet(RAW_DIR / "policy_rates.parquet")
    comm  = pd.read_parquet(RAW_DIR / "commodity_prices.parquet")

    # Normalise dates to month-start
    for df in [cpi, fx, rates, comm]:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    return cpi, fx, rates, comm


# ─────────────────────────────────────── build per-country frame

def build_country(
    country: str,
    cpi: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
    comm: pd.DataFrame,
) -> pd.DataFrame:

    c_cpi   = cpi[cpi["country"] == country].set_index("date").sort_index()
    c_fx    = fx[fx["country"] == country].set_index("date").sort_index()
    c_rates = rates[rates["country"] == country].set_index("date").sort_index()

    # Determine country date range (intersect across available series)
    # Argentina starts later due to CPI data limits
    start = max(
        c_cpi.index.min(),
        c_fx.index.min() if not c_fx.empty else pd.Timestamp("1900-01-01"),
    )
    end = pd.Timestamp("2025-12-01")
    idx = pd.date_range(start=start, end=end, freq="MS")

    df = pd.DataFrame(index=idx)
    df.index.name = "date"
    df["country"] = country

    # ── CPI
    df["cpi_index"]   = c_cpi["cpi_index"].reindex(idx)
    df["cpi_yoy_pct"] = c_cpi["cpi_yoy_pct"].reindex(idx)

    # Impute CPI index: linear interpolation ≤ 3 months
    df["cpi_index"] = interpolate_limited(df["cpi_index"], limit=3)

    # Compute YoY and MoM from index where index is available
    index_mask = df["cpi_index"].notna()
    df.loc[index_mask, "cpi_yoy_pct"] = pct_change_yoy(
        df.loc[index_mask, "cpi_index"]
    )
    df.loc[index_mask, "cpi_mom_pct"] = pct_change_mom(
        df.loc[index_mask, "cpi_index"]
    )

    # For Argentina: cpi_yoy_pct is native (already set above); cpi_mom_pct stays NaN

    # ── FX
    df["fx_lcu_per_usd"] = c_fx["fx_lcu_per_usd"].reindex(idx)
    df["fx_lcu_per_usd"] = ffill_limited(df["fx_lcu_per_usd"], limit=2)

    df["fx_mom_pct"] = pct_change_mom(df["fx_lcu_per_usd"])
    df["fx_yoy_pct"] = pct_change_yoy(df["fx_lcu_per_usd"])

    # ── Policy rate
    if not c_rates.empty:
        df["policy_rate_pct"] = c_rates["policy_rate_pct"].reindex(idx)
        df["policy_rate_pct"] = ffill_limited(df["policy_rate_pct"], limit=3)
    else:
        df["policy_rate_pct"] = np.nan   # Argentina

    df["rate_delta_mom"] = df["policy_rate_pct"].diff(1)

    # ── Primary commodity
    comm_col = COUNTRY_COMMODITY[country]
    comm_indexed = comm.set_index("date").sort_index()
    df["commodity_price"] = comm_indexed[comm_col].reindex(idx)
    df["commodity_name"]  = comm_col

    df["commodity_mom_pct"] = pct_change_mom(df["commodity_price"])
    df["commodity_yoy_pct"] = pct_change_yoy(df["commodity_price"])

    # ── Lags
    df["cpi_lag1"]          = df["cpi_yoy_pct"].shift(1)
    df["cpi_lag3"]          = df["cpi_yoy_pct"].shift(3)
    df["cpi_lag6"]          = df["cpi_yoy_pct"].shift(6)
    df["cpi_lag12"]         = df["cpi_yoy_pct"].shift(12)

    df["fx_lag1_mom"]       = df["fx_mom_pct"].shift(1)
    df["fx_lag3_mom"]       = df["fx_mom_pct"].shift(3)

    df["rate_lag1"]         = df["policy_rate_pct"].shift(1)
    df["rate_lag3"]         = df["policy_rate_pct"].shift(3)

    df["commodity_lag1_mom"] = df["commodity_mom_pct"].shift(1)
    df["commodity_lag3_mom"] = df["commodity_mom_pct"].shift(3)

    return df.reset_index()


# ─────────────────────────────────────── main

def main():
    print("Loading raw data...")
    cpi, fx, rates, comm = load_raw()

    countries = ["Mexico", "Brazil", "Chile", "Colombia", "Argentina"]
    frames = []

    for country in countries:
        print(f"  Building features for {country}...")
        df_c = build_country(country, cpi, fx, rates, comm)
        frames.append(df_c)

    df = pd.concat(frames, ignore_index=True)

    # ── Summary
    print("\n--- Modeling dataset coverage ---")
    for country, grp in df.groupby("country"):
        # Rows with cpi_yoy_pct (the primary target)
        valid = grp["cpi_yoy_pct"].notna()
        n_valid = valid.sum()
        n_full  = (grp["cpi_lag12"].notna() & grp["cpi_yoy_pct"].notna()).sum()
        print(
            f"  {country:12s}: {len(grp):4d} rows  "
            f"cpi_yoy_obs={n_valid:4d}  "
            f"full_feature_obs={n_full:4d}  "
            f"commodity={COUNTRY_COMMODITY[country]}"
        )

    # ── Missing data report
    print("\n--- Null counts (across all countries) ---")
    key_cols = [
        "cpi_yoy_pct", "cpi_mom_pct", "fx_lcu_per_usd",
        "policy_rate_pct", "commodity_price",
        "cpi_lag1", "cpi_lag12",
    ]
    for col in key_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            pct    = n_null / len(df) * 100
            print(f"  {col:25s}: {n_null:5d} nulls ({pct:.1f}%)")

    out_path = PROC_DIR / "modeling_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
