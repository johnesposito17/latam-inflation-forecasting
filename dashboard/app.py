"""
LATAM Inflation Forecasting — Plotly Dash dashboard.

Run locally:
    python3 dashboard/app.py
    Open http://localhost:8050

Features:
  - Country dropdown (Mexico, Brazil, Chile, Colombia, Argentina)
  - Forecast horizon slider (3, 6, 12 months)
  - Model toggle (Prophet / LSTM / Both)
  - Main chart: historical CPI YoY% + test-period forecast with
    Prophet confidence interval shading
  - Metrics card: test-period RMSE and MAPE for selected country/model
"""

import sys
from pathlib import Path

# Allow running from any working directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# ─────────────────────────────────────── load data

MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

modeling   = pd.read_parquet(PROC_DIR / "modeling_dataset.parquet")
comparison = pd.read_parquet(MODEL_DIR / "comparison.parquet")
metrics    = pd.read_parquet(MODEL_DIR / "comparison_metrics.parquet")

modeling["date"]   = pd.to_datetime(modeling["date"])
comparison["date"] = pd.to_datetime(comparison["date"])

COUNTRIES = ["Mexico", "Brazil", "Chile", "Colombia", "Argentina"]
COLORS = {
    "historical": "#4A90D9",
    "prophet":    "#E67E22",
    "lstm":       "#27AE60",
    "ci":         "rgba(230, 126, 34, 0.15)",
}

# ─────────────────────────────────────── app layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="LATAM Inflation Forecasting",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("LATAM Inflation Forecasting", className="mb-0"),
                    html.P(
                        "Prophet + LSTM forecasts of monthly CPI YoY% for 5 LATAM economies",
                        className="text-muted",
                    ),
                ]),
                className="py-3",
            )
        ),
        dbc.Row(
            [
                # ── Controls
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Label("Country", className="fw-bold"),
                            dcc.Dropdown(
                                id="country-dropdown",
                                options=[{"label": c, "value": c} for c in COUNTRIES],
                                value="Mexico",
                                clearable=False,
                                className="mb-3",
                            ),
                            html.Label("Forecast horizon (months)", className="fw-bold"),
                            dcc.Slider(
                                id="horizon-slider",
                                min=3, max=12, step=3, value=12,
                                marks={3: "3", 6: "6", 12: "12"},
                                className="mb-3",
                            ),
                            html.Label("Model", className="fw-bold"),
                            dcc.RadioItems(
                                id="model-radio",
                                options=[
                                    {"label": "Prophet",  "value": "prophet"},
                                    {"label": "LSTM",     "value": "lstm"},
                                    {"label": "Both",     "value": "both"},
                                ],
                                value="both",
                                labelStyle={"display": "block", "marginBottom": "4px"},
                                className="mb-3",
                            ),
                            html.Hr(),
                            html.Div(id="metrics-card"),
                        ])
                    ),
                    width=3,
                ),
                # ── Chart
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="forecast-chart", style={"height": "500px"})
                        )
                    ),
                    width=9,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            dbc.Col(
                html.Small(
                    "Sources: OECD SDMX (CPI), FRED (FX/rates/commodities). "
                    "Test period = last 12 months of each country's available data. "
                    "Confidence intervals from Prophet (80%). "
                    "LSTM: pooled multi-country model with 12-month lookback window.",
                    className="text-muted",
                )
            )
        ),
    ],
    fluid=True,
    className="px-4",
)


# ─────────────────────────────────────── callbacks

@app.callback(
    Output("forecast-chart", "figure"),
    Output("metrics-card", "children"),
    Input("country-dropdown", "value"),
    Input("horizon-slider", "value"),
    Input("model-radio", "value"),
)
def update_chart(country: str, horizon: int, model_choice: str):
    # Historical series
    hist = (
        modeling[modeling["country"] == country]
        .sort_values("date")
        .dropna(subset=["cpi_yoy_pct"])
    )

    # Test-period forecasts (last 12 months available; slice to horizon)
    fc = comparison[comparison["country"] == country].sort_values("date")
    fc = fc.tail(horizon)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["cpi_yoy_pct"],
        name="Historical CPI YoY%",
        line=dict(color=COLORS["historical"], width=2),
    ))

    # Prophet confidence interval (shaded band)
    if model_choice in ("prophet", "both") and "yhat_lower" in fc.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=COLORS["ci"],
            line=dict(color="rgba(0,0,0,0)"),
            name="Prophet 80% CI",
            showlegend=True,
        ))

    # Prophet forecast line
    if model_choice in ("prophet", "both"):
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["prophet_yhat"],
            name="Prophet forecast",
            line=dict(color=COLORS["prophet"], width=2, dash="dash"),
            mode="lines+markers",
        ))

    # LSTM forecast line
    if model_choice in ("lstm", "both") and "lstm_yhat" in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["lstm_yhat"],
            name="LSTM forecast",
            line=dict(color=COLORS["lstm"], width=2, dash="dot"),
            mode="lines+markers",
        ))

    # Actual test values (dots)
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["y"],
        name="Actual (test)",
        mode="markers",
        marker=dict(color="black", size=7, symbol="circle-open"),
    ))

    # Vertical line at train/test boundary
    split_date = fc["date"].min()
    fig.add_vline(
        x=split_date,
        line_dash="dot",
        line_color="grey",
        annotation_text="Test start",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"{country} — CPI YoY Inflation (%) · Last {horizon}-month forecast",
        xaxis_title="Date",
        yaxis_title="CPI YoY %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ── Metrics card
    row = metrics[metrics["country"] == country]
    if row.empty:
        metrics_content = html.P("No metrics available.")
    else:
        row = row.iloc[0]

        def metric_row(label, p_val, l_val, lower_is_better=True):
            p_better = (p_val < l_val) if lower_is_better else (p_val > l_val)
            p_style  = {"color": "#27AE60", "fontWeight": "bold"} if p_better else {}
            l_style  = {"color": "#27AE60", "fontWeight": "bold"} if not p_better else {}
            return html.Tr([
                html.Td(label),
                html.Td(f"{p_val:.3f}", style=p_style),
                html.Td(f"{l_val:.3f}", style=l_style),
            ])

        metrics_content = html.Div([
            html.P(f"Test period metrics — {country}", className="fw-bold mb-1"),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th(""), html.Th("Prophet"), html.Th("LSTM")])),
                    html.Tbody([
                        metric_row("RMSE", row["prophet_rmse"], row["lstm_rmse"]),
                        metric_row("MAPE %", row["prophet_mape"], row["lstm_mape"]),
                    ]),
                ],
                bordered=True, size="sm", className="mb-0",
            ),
            html.Small("Green = better model for that metric", className="text-muted"),
        ])

    return fig, metrics_content


# ─────────────────────────────────────── run

if __name__ == "__main__":
    app.run(debug=True, port=8050)
