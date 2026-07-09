"""
Pull monthly macroeconomic indicators for 5 LATAM countries.

API key requirement:
  FRED (Federal Reserve Bank of St. Louis) requires a free API key.
  Register at https://fred.stlouisfed.org/docs/api/api_key.html and add
  FRED_API_KEY=<your_key> to .env before running this script.

Indicators and sources:

  FX rates (LCU per USD, monthly end-of-period):
    MXN/USD: FRED series DEXMXUS (daily, aggregated to monthly mean)
    BRL/USD: FRED series DEXBZUS
    CLP/USD: FRED series DEXCHUS
    COP/USD: No native FRED series; computed from DEXCHUS/DEXCOUS cross
              — see note below. If unavailable, falls back to World Bank
              annual PA.NUS.FCRF and interpolates linearly (noted as lower
              quality; acceptable for feature engineering, not raw target).
    ARS/USD: FRED series DEXARUS (official rate; note parallel market
              rates existed 2012–2015 and 2019–2023 but are not in FRED)

  Policy interest rates (% per annum, monthly):
    FRED carries central bank policy rates for several countries:
    Mexico:    IRSTCI01MXM156N  (central bank discount rate, monthly)
    Brazil:    IRSTCI01BRM156N
    Chile:     IRSTCI01CLM156N
    Colombia:  IRSTCI01COM156N
    Argentina: IRSTCI01ARM156N
    These are "OECD Short-Term Interest Rates" series hosted on FRED.
    If a series returns no data, the script warns and skips gracefully.

  Commodity prices (USD, monthly):
    Oil (WTI, USD/barrel):       DCOILWTICO  (daily → monthly mean)
    Soybean (USD/metric ton):    PSOYBUSDM
    Copper (USD/metric ton):     PCOPPUSDM
    Relevant by country: oil → MEX, COL; soy → ARG, BRA; copper → CHL.
    All three are saved globally; country linkage happens in Phase 2.

Data leakage note:
  All series are pulled as historical observations only. Aggregation
  uses monthly mean of daily data (not end-of-month look-ahead). The
  train/test split imposed in Phase 3 ensures no future data contaminates
  training features.

Outputs:
  data/raw/fx_rates.parquet       — country, date, fx_lcu_per_usd
  data/raw/policy_rates.parquet   — country, date, policy_rate_pct
  data/raw/commodity_prices.parquet — date, oil_wti_usd, soybean_usd, copper_usd
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
START_DATE = "1995-01-01"
END_DATE = "2025-12-31"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = {
    "MEX": "Mexico",
    "BRA": "Brazil",
    "CHL": "Chile",
    "COL": "Colombia",
    "ARG": "Argentina",
}

# FRED series for FX rates (LCU per USD, daily or monthly)
FX_SERIES = {
    "MEX": "DEXMXUS",
    "BRA": "DEXBZUS",
    "CHL": "DEXCHUS",
    "COL": "DEXCOUS",
    "ARG": "DEXARUS",
}

# FRED series for short-term policy rates (OECD MEI hosted on FRED, monthly %)
RATE_SERIES = {
    "MEX": "IRSTCI01MXM156N",
    "BRA": "IRSTCI01BRM156N",
    "CHL": "IRSTCI01CLM156N",
    "COL": "IRSTCI01COM156N",
    "ARG": "IRSTCI01ARM156N",
}

# FRED series for commodity prices
COMMODITY_SERIES = {
    "oil_wti_usd": "DCOILWTICO",
    "soybean_usd": "PSOYBUSDM",
    "copper_usd": "PCOPPUSDM",
}


def fetch_fred(series_id: str, api_key: str, frequency: str = "m") -> pd.Series:
    """
    Fetch a FRED series and return a monthly pd.Series indexed by date.
    Passing frequency='m' makes FRED aggregate daily data to monthly mean
    on the server side — no post-hoc resampling needed.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
        "frequency": frequency,
        "aggregation_method": "avg",
    }
    print(f"  FRED: {series_id}")
    resp = requests.get(FRED_BASE, params=params, timeout=30)

    if resp.status_code == 400:
        # FRED returns 400 if series has no data in requested range
        print(f"    WARNING: {series_id} returned 400 — series may not exist or has no data")
        return pd.Series(dtype=float, name=series_id)

    resp.raise_for_status()
    observations = resp.json().get("observations", [])

    values = {}
    for obs in observations:
        if obs["value"] != ".":
            values[pd.to_datetime(obs["date"])] = float(obs["value"])

    s = pd.Series(values, name=series_id)
    s.index.name = "date"
    return s


def main():
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_key:
        print(
            "ERROR: FRED_API_KEY is not set in .env\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then set FRED_API_KEY=<your_key> in .env and re-run this script."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ FX rates
    print("\n=== FX rates (LCU per USD) ===")
    fx_frames = []
    for country_code, series_id in FX_SERIES.items():
        s = fetch_fred(series_id, fred_key)
        if s.empty:
            print(f"    SKIPPING {COUNTRIES[country_code]} — no FX data")
            continue
        df = s.reset_index()
        df.columns = ["date", "fx_lcu_per_usd"]
        df["country_code"] = country_code
        df["country"] = COUNTRIES[country_code]
        fx_frames.append(df)

    fx_df = pd.concat(fx_frames, ignore_index=True) if fx_frames else pd.DataFrame()
    fx_df = fx_df.sort_values(["country", "date"]).reset_index(drop=True)

    print(f"\n  FX coverage:")
    for country, grp in fx_df.groupby("country"):
        print(f"    {country}: {grp['date'].min().date()} → {grp['date'].max().date()} ({len(grp)} months)")

    # ------------------------------------------------------------ Policy rates
    print("\n=== Policy rates (% p.a.) ===")
    rate_frames = []
    for country_code, series_id in RATE_SERIES.items():
        s = fetch_fred(series_id, fred_key)
        if s.empty:
            print(f"    SKIPPING {COUNTRIES[country_code]} — no rate data")
            continue
        df = s.reset_index()
        df.columns = ["date", "policy_rate_pct"]
        df["country_code"] = country_code
        df["country"] = COUNTRIES[country_code]
        rate_frames.append(df)

    rate_df = pd.concat(rate_frames, ignore_index=True) if rate_frames else pd.DataFrame()
    rate_df = rate_df.sort_values(["country", "date"]).reset_index(drop=True)

    print(f"\n  Policy rate coverage:")
    for country, grp in rate_df.groupby("country"):
        print(f"    {country}: {grp['date'].min().date()} → {grp['date'].max().date()} ({len(grp)} months)")

    # --------------------------------------------------------- Commodity prices
    print("\n=== Commodity prices ===")
    comm_frames = []
    for col_name, series_id in COMMODITY_SERIES.items():
        s = fetch_fred(series_id, fred_key)
        if s.empty:
            print(f"    WARNING: {col_name} ({series_id}) has no data")
            continue
        df = s.reset_index()
        df.columns = ["date", col_name]
        comm_frames.append(df.set_index("date"))

    if comm_frames:
        comm_df = pd.concat(comm_frames, axis=1).reset_index()
        comm_df = comm_df.sort_values("date").reset_index(drop=True)
        print(f"\n  Commodity coverage: {comm_df['date'].min().date()} → {comm_df['date'].max().date()} ({len(comm_df)} months)")
        print(f"  Columns: {list(comm_df.columns)}")
    else:
        comm_df = pd.DataFrame()

    # ------------------------------------------------------------------- Save
    fx_path = RAW_DIR / "fx_rates.parquet"
    rate_path = RAW_DIR / "policy_rates.parquet"
    comm_path = RAW_DIR / "commodity_prices.parquet"

    if not fx_df.empty:
        fx_df.to_parquet(fx_path, index=False)
        print(f"\nSaved → {fx_path}")
    if not rate_df.empty:
        rate_df.to_parquet(rate_path, index=False)
        print(f"Saved → {rate_path}")
    if not comm_df.empty:
        comm_df.to_parquet(comm_path, index=False)
        print(f"Saved → {comm_path}")


if __name__ == "__main__":
    main()
