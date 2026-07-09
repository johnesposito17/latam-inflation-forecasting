"""
Phase 5 — Compare Prophet vs LSTM performance and print a summary table.
Outputs models/comparison.parquet for use by the dashboard.
"""

import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    prophet = pd.read_parquet(MODEL_DIR / "prophet_metrics.parquet")
    lstm    = pd.read_parquet(MODEL_DIR / "lstm_metrics.parquet")

    prophet = prophet.rename(columns={"rmse": "prophet_rmse", "mape": "prophet_mape"})
    lstm    = lstm.rename(columns={"rmse": "lstm_rmse", "mape": "lstm_mape"})

    comp = prophet[["country", "prophet_rmse", "prophet_mape"]].merge(
        lstm[["country", "lstm_rmse", "lstm_mape"]], on="country"
    )
    comp["rmse_winner"] = comp.apply(
        lambda r: "LSTM" if r["lstm_rmse"] < r["prophet_rmse"] else "Prophet", axis=1
    )
    comp["mape_winner"] = comp.apply(
        lambda r: "LSTM" if r["lstm_mape"] < r["prophet_mape"] else "Prophet", axis=1
    )

    print("=" * 78)
    print(f"{'Country':<12} {'P-RMSE':>8} {'L-RMSE':>8} {'P-MAPE%':>9} {'L-MAPE%':>9} {'RMSE-Win':>10} {'MAPE-Win':>10}")
    print("-" * 78)
    for _, row in comp.iterrows():
        print(
            f"{row['country']:<12} {row['prophet_rmse']:>8.3f} {row['lstm_rmse']:>8.3f}"
            f" {row['prophet_mape']:>9.2f} {row['lstm_mape']:>9.2f}"
            f" {row['rmse_winner']:>10} {row['mape_winner']:>10}"
        )
    print("=" * 78)

    # Also load per-month forecasts for the dashboard
    pf = pd.read_parquet(MODEL_DIR / "prophet_forecasts.parquet")
    lf = pd.read_parquet(MODEL_DIR / "lstm_forecasts.parquet")
    lf = lf.rename(columns={"yhat": "lstm_yhat"})

    pf_test = pf[pf["split"] == "test"][["country", "ds", "y", "yhat", "yhat_lower", "yhat_upper"]]
    pf_test = pf_test.rename(columns={"ds": "date", "yhat": "prophet_yhat"})
    lf_test = lf[["country", "date", "lstm_yhat"]]
    lf_test["date"] = pd.to_datetime(lf_test["date"])
    pf_test["date"] = pd.to_datetime(pf_test["date"])

    combined = pf_test.merge(lf_test, on=["country", "date"], how="left")

    out = MODEL_DIR / "comparison.parquet"
    combined.to_parquet(out, index=False)
    comp.to_parquet(MODEL_DIR / "comparison_metrics.parquet", index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
