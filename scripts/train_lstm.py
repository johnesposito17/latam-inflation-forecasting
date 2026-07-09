"""
Phase 4 — Stacked LSTM model for LATAM CPI YoY% forecasting.

Architecture decisions:
  Stacked LSTM (2 layers):
    Layer 1: 64 hidden units, returns sequences → Layer 2: 32 hidden units
    Dropout (0.2) between layers to reduce overfitting on smaller series.
    Final linear layer maps hidden state → 1 output (next-month YoY%).
    Two LSTM layers allow the model to learn both short-run momentum and
    medium-run macro dynamics simultaneously.

  Multi-country vs per-country:
    Single model trained on all 5 countries jointly, with country as a
    one-hot embedding in the input features. Rationale: LATAM economies
    share structural similarities (commodity exposure, dollar pass-through,
    rate sensitivity), so pooling data improves generalisation — especially
    for Argentina which has only 85 training months. Per-country models
    would underfit Argentina severely. The pooled model effectively learns
    a shared macro dynamic and uses the country embedding to learn offsets.

  Window / lookback:
    12 months. Chosen because: (1) YoY% inflation has a natural 12-month
    periodicity — using exactly 12 lags lets the model see the same month
    in the prior year, capturing seasonality without a separate component;
    (2) policy rate changes typically pass through to CPI within 6-12 months;
    (3) shorter windows (3-6) miss medium-run dynamics; longer (24+) create
    sparse gradients with limited data.

  Input features per timestep:
    cpi_yoy_pct, fx_mom_pct, commodity_mom_pct, rate_lag1 + country one-hot
    → 4 numeric + 5 binary = 9-dimensional input at each of 12 timesteps.

  Normalisation:
    StandardScaler fit ONLY on the training set, then applied to test.
    Fitting on the full dataset would leak test-set statistics into
    training (a subtle but real form of data leakage).

  Train/test split:
    Same chronological cutoff as Prophet (last 12 months = test).
    Cross-country: each country contributes its own split; the model
    sees all countries simultaneously during training.

  Training:
    Optimizer: Adam (lr=1e-3). Loss: MSE.
    Early stopping on validation loss (patience=20 epochs).
    Max 200 epochs. Batch size 32.

  Evaluation:
    Same metrics as Prophet (RMSE, MAPE on test set) for direct comparison.

Outputs:
  models/lstm_weights.pt       — saved model state dict
  models/lstm_forecasts.parquet
  models/lstm_metrics.parquet
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

PROC_DIR  = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

WINDOW      = 12
TEST_MONTHS = 12
HIDDEN1     = 64
HIDDEN2     = 32
DROPOUT     = 0.2
LR          = 1e-3
MAX_EPOCHS  = 200
PATIENCE    = 20
BATCH_SIZE  = 32

COUNTRIES   = ["Mexico", "Brazil", "Chile", "Colombia", "Argentina"]
FEATURES    = ["cpi_yoy_pct", "fx_mom_pct", "commodity_mom_pct", "rate_lag1"]


# ─────────────────────────────────────── model

class StackedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop    = nn.Dropout(dropout)
        self.lstm2   = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc      = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out     = self.drop(out)
        out, _ = self.lstm2(out)
        return self.fc(out[:, -1, :]).squeeze(-1)   # last timestep


# ─────────────────────────────────────── helpers

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def make_sequences(
    data: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: X[i] = data[i:i+window], y[i] = target at i+window."""
    X, y = [], []
    target_col = 0   # cpi_yoy_pct is always the first feature column
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────── data prep

def prepare_data(df: pd.DataFrame):
    """
    Returns scaled train/test tensors and scalers, plus metadata for
    reconstructing per-country results.
    """
    country_one_hot = pd.get_dummies(df["country"], dtype=float).reindex(
        columns=COUNTRIES, fill_value=0.0
    )
    feature_cols = FEATURES + COUNTRIES

    # Fill NaN safely (Argentina policy rate → 0)
    feat_df = df[FEATURES].fillna(0.0)
    full_df = pd.concat([feat_df.reset_index(drop=True),
                         country_one_hot.reset_index(drop=True)], axis=1)
    full_df["date"]    = df["date"].values
    full_df["country"] = df["country"].values

    # Per-country chronological split → gather train and test rows
    train_rows, test_rows = [], []
    country_meta = {}   # country → (train_start_idx, test_start_idx in full_df)

    for country in COUNTRIES:
        mask = full_df["country"] == country
        c_df = full_df[mask].sort_values("date").reset_index(drop=True)
        split = len(c_df) - TEST_MONTHS
        train_rows.append(c_df.iloc[:split])
        test_rows.append(c_df.iloc[split:])
        country_meta[country] = {
            "train_len": split,
            "test_dates": c_df.iloc[split:]["date"].values,
            "test_actual": c_df.iloc[split:]["cpi_yoy_pct"].values,
        }

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df  = pd.concat(test_rows, ignore_index=True)

    # Fit scaler on training numeric features only (no leakage)
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[feature_cols].values)
    test_num  = scaler.transform(test_df[feature_cols].values)

    return train_num, test_num, scaler, feature_cols, country_meta, train_df, test_df


# ─────────────────────────────────────── training loop

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    best_val  = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Shuffle training batches
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}  (best val MSE={best_val:.4f})")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}: train MSE={epoch_loss/n_batches:.4f}  val MSE={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────── main

def main():
    print("Loading modeling dataset...")
    df = pd.read_parquet(PROC_DIR / "modeling_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    print("Preparing sequences...")
    train_num, test_num, scaler, feature_cols, country_meta, train_df, test_df = prepare_data(df)

    # Build sequences per country, then pool
    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    X_te_meta = []   # (country, date) tuples for reconstruction

    for country in COUNTRIES:
        # Training country sequences
        c_train_mask = (train_df["country"] == country).values
        c_train_data = train_num[c_train_mask]
        if len(c_train_data) > WINDOW:
            Xc, yc = make_sequences(c_train_data, WINDOW)
            X_tr_list.append(Xc)
            y_tr_list.append(yc)

        # Test: we need WINDOW months of train context + test months
        # Pull last WINDOW training rows + all test rows for this country
        c_test_mask  = (test_df["country"] == country).values
        c_test_data  = test_num[c_test_mask]
        c_train_tail = c_train_data[-WINDOW:]
        c_context    = np.concatenate([c_train_tail, c_test_data], axis=0)
        Xt, yt = make_sequences(c_context, WINDOW)
        X_te_list.append(Xt)
        y_te_list.append(yt)
        for d in country_meta[country]["test_dates"]:
            X_te_meta.append((country, d))

    X_train = torch.tensor(np.concatenate(X_tr_list), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate(y_tr_list), dtype=torch.float32)
    X_test  = torch.tensor(np.concatenate(X_te_list), dtype=torch.float32)
    y_test  = torch.tensor(np.concatenate(y_te_list), dtype=torch.float32)

    print(f"  Train sequences: {X_train.shape}  Test sequences: {X_test.shape}")

    # Use 10% of train as internal validation for early stopping
    val_split  = int(0.9 * len(X_train))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_tr  = X_train[:val_split]
    y_tr  = y_train[:val_split]

    model = StackedLSTM(
        input_size=len(feature_cols),
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        dropout=DROPOUT,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training...")

    model = train_model(model, X_tr, y_tr, X_val, y_val)

    # ── Evaluate per country
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).numpy()

    # Inverse-scale predictions (only the target column = index 0)
    # We reconstruct a full-feature row to inverse-transform properly
    dummy = np.zeros((len(preds_scaled), len(feature_cols)), dtype=np.float32)
    dummy[:, 0] = preds_scaled
    preds_unscaled = scaler.inverse_transform(dummy)[:, 0]

    # Assemble per-country results
    all_forecasts = []
    all_metrics   = []
    pos = 0
    for country in COUNTRIES:
        meta     = country_meta[country]
        n_test   = TEST_MONTHS
        y_pred_c = preds_unscaled[pos: pos + n_test]
        y_true_c = meta["test_actual"]
        dates_c  = meta["test_dates"]

        r = rmse(y_true_c, y_pred_c)
        m = mape(y_true_c, y_pred_c)
        print(f"  {country:12s}  RMSE={r:.3f}  MAPE={m:.2f}%")

        all_forecasts.append(pd.DataFrame({
            "country": country,
            "date":    dates_c,
            "y":       y_true_c,
            "yhat":    y_pred_c,
            "split":   "test",
        }))
        all_metrics.append({"country": country, "rmse": r, "mape": m,
                             "test_months": n_test})
        pos += n_test

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    metrics_df   = pd.DataFrame(all_metrics)

    torch.save(model.state_dict(), MODEL_DIR / "lstm_weights.pt")
    forecasts_df.to_parquet(MODEL_DIR / "lstm_forecasts.parquet", index=False)
    metrics_df.to_parquet(MODEL_DIR / "lstm_metrics.parquet", index=False)

    print("\n=== LSTM Model Summary ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
