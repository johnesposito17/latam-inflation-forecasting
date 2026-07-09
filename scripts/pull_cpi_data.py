"""
Pull monthly CPI data for 5 LATAM countries from the OECD SDMX API.

Source decisions:
  World Bank (FP.CPI.TOTL) only provides annual CPI — not usable for
  monthly modeling. IMF IFS SDMX API is unreachable from this environment.
  OECD SDMX (sdmx.oecd.org) is public and requires no authentication.

Dataset mapping per country:
  MEX, BRA → DSD_G20_PRICES@DF_G20_PRICES
    These are G20 members with complete OECD series (1995–2025, 372 months).
    CPI index (2015 = 100), national methodology.

  CHL → DSD_PRICES@DF_PRICES_ALL spliced with
         DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL
    Chile released CPI under COICOP 1999 through 2023-12, then switched to
    COICOP 2018 from 2024-01. Both series use base year 2015=100 and produce
    identical values in the overlap window, so they splice cleanly.

  COL → DSD_PRICES@DF_PRICES_ALL
    Colombia joined OECD in 2020 but OECD carries data from 1995.
    Full coverage 1995–2025.

  ARG → DSD_G20_PRICES@DF_G20_PRICES  (YoY % change only, 2017+)
    OECD excludes Argentina from full CPI index datasets. INDEC (Argentina's
    stats agency) falsified CPI data 2007–2016; IMF censured Argentina in
    2013. OECD carries only post-2017 data, and only as YoY % change in the
    G20 dataset. Argentina's training window therefore starts in 2017, and
    we forecast YoY % change for all countries (derived from index for the
    other four during Phase 2 feature engineering).

Important: batch requests to the OECD SDMX API silently truncate data when
  the response is large. Each country is fetched in a separate request to
  guarantee complete coverage.

Output: data/raw/cpi_monthly.parquet
  Columns: country, country_code, date, cpi_index, cpi_yoy_pct
  cpi_index is NaN for ARG; cpi_yoy_pct is NaN for non-ARG until Phase 2.
"""

import requests
import pandas as pd
from pathlib import Path

OECD_BASE = "https://sdmx.oecd.org/public/rest/data"
START = "1995-01"
END = "2025-12"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = {
    "MEX": "Mexico",
    "BRA": "Brazil",
    "CHL": "Chile",
    "COL": "Colombia",
    "ARG": "Argentina",
}


def fetch_oecd(dataflow: str, key: str, start: str = START, end: str = END) -> dict:
    url = (
        f"{OECD_BASE}/{dataflow}/{key}"
        f"?startPeriod={start}&endPeriod={end}&format=jsondata"
    )
    print(f"  GET {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]


def parse_series(data: dict, country_code: str) -> list[dict]:
    """
    Flatten a single-country SDMX-JSON 2.0 response to a list of
    {country_code, date, value} dicts.
    """
    struct = data["structures"][0]
    time_values = [v["id"] for v in struct["dimensions"]["observation"][0]["values"]]
    area_values = [v["id"] for v in struct["dimensions"]["series"][0]["values"]]

    rows = []
    for series_key, series_data in data["dataSets"][0]["series"].items():
        dim_indices = [int(x) for x in series_key.split(":")]
        code = area_values[dim_indices[0]]
        if code != country_code:
            continue
        for obs_key, obs_vals in series_data["observations"].items():
            rows.append(
                {
                    "country_code": country_code,
                    "date": pd.to_datetime(time_values[int(obs_key)]),
                    "value": float(obs_vals[0]),
                }
            )
    return rows


def fetch_index(country_code: str, dataflow: str) -> pd.DataFrame:
    key = f"{country_code}.M.N.CPI.IX._T.N._Z"
    data = fetch_oecd(dataflow, key)
    rows = parse_series(data, country_code)
    df = pd.DataFrame(rows).rename(columns={"value": "cpi_index"})
    df["cpi_yoy_pct"] = float("nan")
    return df


def main():
    frames = []

    # --- Mexico and Brazil: G20 dataset ---
    for code in ["MEX", "BRA"]:
        print(f"Pulling {COUNTRIES[code]} CPI index from G20 dataset...")
        df = fetch_index(code, "OECD.SDD.TPS,DSD_G20_PRICES@DF_G20_PRICES")
        frames.append(df)

    # --- Chile: splice COICOP 1999 (up to 2023-12) + COICOP 2018 (2024+) ---
    print("Pulling Chile CPI index from DF_PRICES_ALL (COICOP 1999)...")
    df_chl_1999 = fetch_index("CHL", "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL")

    print("Pulling Chile CPI index from DF_PRICES_C2018_ALL (COICOP 2018, extension)...")
    data_chl_2018 = fetch_oecd(
        "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL",
        "CHL.M.N.CPI.IX._T.N._Z",
        start="2024-01",
    )
    rows_chl_2018 = parse_series(data_chl_2018, "CHL")
    df_chl_2018 = pd.DataFrame(rows_chl_2018).rename(columns={"value": "cpi_index"})
    df_chl_2018["cpi_yoy_pct"] = float("nan")

    # Splice: keep COICOP 1999 through 2023-12, COICOP 2018 from 2024-01 onward
    df_chl = pd.concat(
        [
            df_chl_1999[df_chl_1999["date"] < "2024-01-01"],
            df_chl_2018[df_chl_2018["date"] >= "2024-01-01"],
        ],
        ignore_index=True,
    )
    frames.append(df_chl)

    # --- Colombia: DF_PRICES_ALL ---
    print("Pulling Colombia CPI index from DF_PRICES_ALL...")
    df_col = fetch_index("COL", "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL")
    frames.append(df_col)

    # --- Argentina: YoY % change from G20 dataset ---
    print("Pulling Argentina CPI YoY% from G20 dataset (2017+)...")
    data_arg = fetch_oecd(
        "OECD.SDD.TPS,DSD_G20_PRICES@DF_G20_PRICES",
        "ARG.M.N.CPI.PA._T.N.GY",
        start="2017-01",
    )
    rows_arg = parse_series(data_arg, "ARG")
    df_arg = pd.DataFrame(rows_arg).rename(columns={"value": "cpi_yoy_pct"})
    df_arg["cpi_index"] = float("nan")
    frames.append(df_arg)

    # --- Combine and save ---
    df = pd.concat(frames, ignore_index=True)
    df["country"] = df["country_code"].map(COUNTRIES)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    print("\n--- CPI Coverage ---")
    for country, grp in df.groupby("country"):
        n_index = grp["cpi_index"].notna().sum()
        n_yoy = grp["cpi_yoy_pct"].notna().sum()
        d_min = grp["date"].min().date()
        d_max = grp["date"].max().date()
        print(
            f"  {country:12s}: {d_min} → {d_max}  "
            f"cpi_index={n_index}  cpi_yoy_pct={n_yoy}"
        )

    out_path = RAW_DIR / "cpi_monthly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
