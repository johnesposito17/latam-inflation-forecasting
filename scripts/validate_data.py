"""
Validate raw data files pulled by pull_cpi_data.py and pull_macro_indicators.py.

Checks per file:
  - File exists
  - Required columns are present
  - Date coverage and gap detection (gaps > 45 days in monthly data)
  - Value range sanity checks
  - Null counts

Run after both pull scripts complete. Reports warnings but does not stop
execution, so you can see all issues at once before deciding how to handle
them in Phase 2 feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
COUNTRIES = ["Argentina", "Brazil", "Chile", "Colombia", "Mexico"]

PASS = "OK"
WARN = "WARN"
MISS = "MISSING"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def check_gaps(dates: pd.Series, label: str) -> list[str]:
    """Return list of gap descriptions for gaps > 45 days."""
    sorted_dates = dates.sort_values().reset_index(drop=True)
    diffs = sorted_dates.diff().dropna()
    big_gaps = diffs[diffs > pd.Timedelta(days=45)]
    issues = []
    for idx in big_gaps.index:
        gap_start = sorted_dates.iloc[idx - 1].date()
        gap_end = sorted_dates.iloc[idx].date()
        issues.append(f"{gap_start} → {gap_end}")
    return issues


def validate_country_series(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    expected_range: tuple[float, float] | None = None,
):
    print(f"\n  [{label}]")
    for country, grp in df.groupby("country"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        nulls = grp[value_col].isna().sum()
        d_min = grp["date"].min().date()
        d_max = grp["date"].max().date()

        flags = []
        if nulls > 0:
            flags.append(f"{nulls} nulls")
        if expected_range:
            lo, hi = expected_range
            out = ((grp[value_col] < lo) | (grp[value_col] > hi)).sum()
            if out > 0:
                flags.append(f"{out} outside [{lo},{hi}]")
        gaps = check_gaps(grp["date"], country)
        if gaps:
            flags.append(f"gaps: {gaps[:2]}")

        status = WARN if flags else PASS
        flag_str = " | ".join(flags) if flags else ""
        print(
            f"    [{status:4s}] {country:12s}  {d_min} → {d_max}  n={n:4d}"
            + (f"  ⚠ {flag_str}" if flag_str else "")
        )


def main():
    section("FILE EXISTENCE")
    files = {
        "cpi_monthly.parquet": True,        # required
        "fx_rates.parquet": False,           # optional (needs FRED key)
        "policy_rates.parquet": False,
        "commodity_prices.parquet": False,
    }
    all_required_present = True
    for fname, required in files.items():
        path = RAW_DIR / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [FOUND]   {fname}  ({size_kb:.1f} KB)")
        else:
            tag = MISS if required else "OPTIONAL"
            print(f"  [{tag}] {fname}")
            if required:
                all_required_present = False

    if not all_required_present:
        print("\nRun pull_cpi_data.py first.")
        return

    # ----------------------------------------------------------------- CPI
    section("CPI DATA  (cpi_monthly.parquet)")
    cpi = pd.read_parquet(RAW_DIR / "cpi_monthly.parquet")
    print(f"  Shape: {cpi.shape}")
    print(f"  Columns: {list(cpi.columns)}")
    print(f"  Country-level breakdown:")

    # CPI index for MEX, BRA, CHL, COL
    cpi_index = cpi[cpi["cpi_index"].notna()].copy()
    if not cpi_index.empty:
        validate_country_series(
            cpi_index,
            "cpi_index",
            "CPI index (2015=100)",
            expected_range=(10.0, 10_000.0),  # wide range; ARG could be very high
        )

    # CPI YoY% for ARG
    cpi_yoy = cpi[cpi["cpi_yoy_pct"].notna()].copy()
    if not cpi_yoy.empty:
        validate_country_series(
            cpi_yoy,
            "cpi_yoy_pct",
            "CPI YoY % (ARG only)",
            expected_range=(0.0, 1000.0),   # ARG has seen 200%+ inflation
        )

    print("\n  Sample rows:")
    print(cpi.groupby("country").tail(2).to_string(index=False))

    # ----------------------------------------------------------------- FX
    fx_path = RAW_DIR / "fx_rates.parquet"
    if fx_path.exists():
        section("FX RATES  (fx_rates.parquet)")
        fx = pd.read_parquet(fx_path)
        print(f"  Shape: {fx.shape}")
        validate_country_series(
            fx,
            "fx_lcu_per_usd",
            "FX rate (LCU per USD)",
            expected_range=(0.0, 10_000.0),  # COP runs 3k-5k/USD; ARS 1k+ by 2024
        )
        # Special note for Argentina — official rate can diverge from reality
        arg_fx = fx[fx["country"] == "Argentina"]
        if not arg_fx.empty:
            print(
                f"\n  Note: Argentina FX range {arg_fx['fx_lcu_per_usd'].min():.1f}"
                f" – {arg_fx['fx_lcu_per_usd'].max():.1f} ARS/USD"
                " (official rate; parallel rate diverged significantly 2012–2015, 2019–2023)"
            )

    # -------------------------------------------------------------- Rates
    rate_path = RAW_DIR / "policy_rates.parquet"
    if rate_path.exists():
        section("POLICY RATES  (policy_rates.parquet)")
        rates = pd.read_parquet(rate_path)
        print(f"  Shape: {rates.shape}")
        validate_country_series(
            rates,
            "policy_rate_pct",
            "Policy rate (% p.a.)",
            expected_range=(0.0, 150.0),    # ARG has exceeded 100%
        )
        arg_rate = rates[rates["country"] == "Argentina"]
        if not arg_rate.empty:
            print(
                f"\n  Note: Argentina rate range "
                f"{arg_rate['policy_rate_pct'].min():.1f}%"
                f" – {arg_rate['policy_rate_pct'].max():.1f}%"
            )

    # ---------------------------------------------------------- Commodities
    comm_path = RAW_DIR / "commodity_prices.parquet"
    if comm_path.exists():
        section("COMMODITY PRICES  (commodity_prices.parquet)")
        comm = pd.read_parquet(comm_path)
        print(f"  Shape: {comm.shape}")
        print(f"  Date range: {comm['date'].min().date()} → {comm['date'].max().date()}")

        checks = {
            "oil_wti_usd": (5.0, 200.0),
            "soybean_usd": (100.0, 3000.0),
            "copper_usd": (500.0, 15_000.0),
        }
        print(f"\n  Column stats:")
        for col, (lo, hi) in checks.items():
            if col not in comm.columns:
                print(f"    [MISSING] {col}")
                continue
            nulls = comm[col].isna().sum()
            out = ((comm[col] < lo) | (comm[col] > hi)).sum()
            status = WARN if (nulls > 0 or out > 0) else PASS
            print(
                f"    [{status:4s}] {col}: "
                f"min={comm[col].min():.1f}  max={comm[col].max():.1f}  "
                f"nulls={nulls}  out_of_range({lo},{hi})={out}"
            )

    section("VALIDATION COMPLETE")
    print("  Review any WARN entries above before Phase 2 feature engineering.")


if __name__ == "__main__":
    main()
