"""
=================================================================================
Artificial Intelligence and Neural Networks - COM7019 - [3921]
Task 1: Stock Market Prediction using LSTM and GRU
=================================================================================
Student      : Farid Negahbani
Student ID   : 24154844
Email        : 24154844@ardenuniversity.ac.uk
Module       : Artificial Intelligence and Neural Networks [COM7019]
Professor    : Ali Vaisifard
Dataset      : Stock_Price_Data.csv  (daily OHLCV time-series data)

Description
-----------
This script implements an end-to-end deep learning pipeline that designs,
trains, and critically compares two recurrent neural network architectures -
Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) - for univariate
stock closing-price prediction. The pipeline covers data loading, exploratory
data analysis, pre-processing (scaling and supervised-sequence windowing),
model design, training with dropout and early stopping for generalisation,
a controlled hyperparameter experiment, quantitative evaluation of
(MAE, RMSE, MAPE, R^2), and the generation of all figures referenced in the
accompanying report.

All generated figures and tables are written to the ./outputs folder using the
filenames referenced throughout the written report, so they can be located and
inserted at the correct point in the document.
=================================================================================
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS AND GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
import os
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_PATH = "dataset/Stock_Price_Data.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modelling configuration
WINDOW_SIZE = 60          # number of past trading days used to predict the next day
TRAIN_SPLIT = 0.80        # chronological 80/20 train-test split
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 6               # early-stopping patience

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def save_fig(filename):
    """Helper to save the current matplotlib figure into the outputs folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"[Saved figure] {path}")


# ---------------------------------------------------------------------------
# 2. DATA LOADING AND EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: LOADING AND EXPLORING THE DATASET")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Dataset shape           : {df.shape}")
print(f"Date range              : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Missing values per col  :\n{df.isna().sum()}")
print(df.describe())

# Fig 1: Full closing-price history
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1)
plt.title("Historical Closing Price (Full Series)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(alpha=0.3)
save_fig("fig1_closing_price_trend.png")

# Fig 2: Trading volume over time
plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["Volume"], color="#d62728", linewidth=0.8)
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(alpha=0.3)
save_fig("fig2_trading_volume_trend.png")

# Fig 3: Correlation heatmap between OHLCV features
plt.figure(figsize=(6, 5))
corr = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].corr()
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Matrix of OHLCV Features")
save_fig("fig3_feature_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING: SCALING AND SEQUENCE WINDOWING
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: PRE-PROCESSING (SCALING + SUPERVISED SEQUENCE WINDOWING)")
print("=" * 80)

close_prices = df[["Close"]].values

# Chronological split BEFORE scaling to avoid data leakage from future values
split_idx = int(len(close_prices) * TRAIN_SPLIT)
train_raw = close_prices[:split_idx]
test_raw = close_prices[split_idx - WINDOW_SIZE:]  # include lookback context

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)


def create_sequences(data, window_size):
    """Transform a 1-D scaled price series into supervised (X, y) sequences."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training sequences : {X_train.shape}")
print(f"Testing sequences  : {X_test.shape}")

# Dates aligned with the test predictions (for plotting)
test_dates = df["Date"].iloc[split_idx:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. MODEL DEFINITIONS (LSTM AND GRU)
# ---------------------------------------------------------------------------
def build_lstm_model(window_size, units=64, dropout_rate=0.2, lr=0.001):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation="relu"),
        Dense(1)
    ], name="LSTM_Model")
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def build_gru_model(window_size, units=64, dropout_rate=0.2, lr=0.001):
    model = Sequential([
        Input(shape=(window_size, 1)),
        GRU(units, return_sequences=True),
        Dropout(dropout_rate),
        GRU(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation="relu"),
        Dense(1)
    ], name="GRU_Model")
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


early_stop = EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
)

# ---------------------------------------------------------------------------
# 5. TRAIN THE LSTM MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 3: TRAINING THE LSTM MODEL")
print("=" * 80)

lstm_model = build_lstm_model(WINDOW_SIZE)
lstm_model.summary()

t0 = time.time()
lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2
)
lstm_train_time = time.time() - t0
print(f"LSTM training time: {lstm_train_time:.1f}s "
      f"(stopped at epoch {len(lstm_history.history['loss'])} of {EPOCHS})")

# Fig 4: LSTM training/validation loss curve
plt.figure(figsize=(8, 5))
plt.plot(lstm_history.history["loss"], label="Training Loss")
plt.plot(lstm_history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Model: Training vs Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
save_fig("fig4_lstm_loss_curve.png")


# ---------------------------------------------------------------------------
# 6. TRAIN THE GRU MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4: TRAINING THE GRU MODEL")
print("=" * 80)

gru_model = build_gru_model(WINDOW_SIZE)
gru_model.summary()

t0 = time.time()
gru_history = gru_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2
)
gru_train_time = time.time() - t0
print(f"GRU training time: {gru_train_time:.1f}s "
      f"(stopped at epoch {len(gru_history.history['loss'])} of {EPOCHS})")

# Fig 5: GRU training/validation loss curve
plt.figure(figsize=(8, 5))
plt.plot(gru_history.history["loss"], label="Training Loss")
plt.plot(gru_history.history["val_loss"], label="Validation Loss")
plt.title("GRU Model: Training vs Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
save_fig("fig5_gru_loss_curve.png")


# ---------------------------------------------------------------------------
# 7. PREDICTIONS AND INVERSE SCALING
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 5: GENERATING PREDICTIONS")
print("=" * 80)

lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
gru_pred_scaled = gru_model.predict(X_test, verbose=0)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
gru_pred = scaler.inverse_transform(gru_pred_scaled)

plot_dates = test_dates.iloc[-len(y_test_actual):].reset_index(drop=True)

# Fig 6: LSTM predicted vs actual closing price
plt.figure(figsize=(12, 5))
plt.plot(plot_dates, y_test_actual, label="Actual Close Price", color="black", linewidth=1.2)
plt.plot(plot_dates, lstm_pred, label="LSTM Predicted Price", color="#1f77b4", linewidth=1.2)
plt.title("LSTM: Actual vs Predicted Closing Price (Test Set)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
save_fig("fig6_lstm_actual_vs_predicted.png")

# Fig 7: GRU predicted vs actual closing price
plt.figure(figsize=(12, 5))
plt.plot(plot_dates, y_test_actual, label="Actual Close Price", color="black", linewidth=1.2)
plt.plot(plot_dates, gru_pred, label="GRU Predicted Price", color="#2ca02c", linewidth=1.2)
plt.title("GRU: Actual vs Predicted Closing Price (Test Set)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
save_fig("fig7_gru_actual_vs_predicted.png")

# Fig 8: Combined LSTM vs GRU comparison against actual price
plt.figure(figsize=(12, 5))
plt.plot(plot_dates, y_test_actual, label="Actual Close Price", color="black", linewidth=1.4)
plt.plot(plot_dates, lstm_pred, label="LSTM Predicted", color="#1f77b4", linewidth=1.0, alpha=0.8)
plt.plot(plot_dates, gru_pred, label="GRU Predicted", color="#2ca02c", linewidth=1.0, alpha=0.8)
plt.title("LSTM vs GRU: Predicted Closing Price Comparison")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
save_fig("fig8_lstm_vs_gru_comparison.png")


# ---------------------------------------------------------------------------
# 8. QUANTITATIVE EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 6: EVALUATION METRICS")
print("=" * 80)


def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name:10s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | MAPE: {mape:6.2f}% | R2: {r2:6.4f}")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


lstm_metrics = evaluate(y_test_actual, lstm_pred, "LSTM")
gru_metrics = evaluate(y_test_actual, gru_pred, "GRU")

results_df = pd.DataFrame([lstm_metrics, gru_metrics])
results_df["Training Time (s)"] = [round(lstm_train_time, 1), round(gru_train_time, 1)]
results_df["Epochs Run"] = [len(lstm_history.history["loss"]), len(gru_history.history["loss"])]
results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n[Saved table] {results_path}")
print(results_df.to_string(index=False))

# Fig 9: Bar chart comparing MAE / RMSE / MAPE between LSTM and GRU
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metrics_to_plot = ["MAE", "RMSE", "MAPE (%)"]
colors = ["#1f77b4", "#2ca02c"]
for ax, metric in zip(axes, metrics_to_plot):
    ax.bar(results_df["Model"], results_df[metric], color=colors)
    ax.set_title(metric)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("LSTM vs GRU: Error Metric Comparison (Lower is Better)")
save_fig("fig9_model_comparison_metrics.png")


# ---------------------------------------------------------------------------
# 9. HYPERPARAMETER / CONFIGURATION EXPERIMENT
#    (demonstrates exploration of architectural choices, as required by the
#     assignment brief, using a smaller controlled run for comparison)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 7: HYPERPARAMETER EXPERIMENT (DROPOUT RATE AND UNIT SIZE)")
print("=" * 80)

experiment_configs = [
    {"name": "LSTM (units=32, dropout=0.1)", "units": 32, "dropout": 0.1},
    {"name": "LSTM (units=64, dropout=0.2)", "units": 64, "dropout": 0.2},
    {"name": "LSTM (units=128, dropout=0.4)", "units": 128, "dropout": 0.4},
]

experiment_results = []
for cfg in experiment_configs:
    model = build_lstm_model(WINDOW_SIZE, units=cfg["units"], dropout_rate=cfg["dropout"])
    hist = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=0
    )
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
    mae = mean_absolute_error(y_test_actual, pred)
    experiment_results.append({"Configuration": cfg["name"], "RMSE": rmse, "MAE": mae,
                                "Val Loss (best)": min(hist.history["val_loss"])})
    print(f"{cfg['name']:32s} | RMSE: {rmse:8.4f} | MAE: {mae:8.4f}")

experiment_df = pd.DataFrame(experiment_results)
experiment_path = os.path.join(OUTPUT_DIR, "hyperparameter_experiment_results.csv")
experiment_df.to_csv(experiment_path, index=False)
print(f"[Saved table] {experiment_path}")

# Fig 10: Hyperparameter experiment RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(experiment_df["Configuration"], experiment_df["RMSE"], color="#ff7f0e")
plt.title("Effect of Units and Dropout Rate on LSTM RMSE")
plt.ylabel("RMSE (USD)")
plt.xticks(rotation=15, ha="right")
plt.grid(alpha=0.3, axis="y")
save_fig("fig10_hyperparameter_experiment.png")


# ---------------------------------------------------------------------------
# 10. SUMMARY OUTPUT
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 8: FINAL SUMMARY")
print("=" * 80)

better_model = "LSTM" if lstm_metrics["RMSE"] < gru_metrics["RMSE"] else "GRU"
summary_lines = [
    "STOCK MARKET PREDICTION - LSTM vs GRU - SUMMARY REPORT",
    "=" * 60,
    f"Dataset size            : {df.shape[0]} rows ({df['Date'].min().date()} to {df['Date'].max().date()})",
    f"Window size (lookback)  : {WINDOW_SIZE} trading days",
    f"Train / test split      : {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%",
    "",
    "LSTM Results:",
    f"  MAE  : {lstm_metrics['MAE']:.4f}",
    f"  RMSE : {lstm_metrics['RMSE']:.4f}",
    f"  MAPE : {lstm_metrics['MAPE (%)']:.2f}%",
    f"  R2   : {lstm_metrics['R2']:.4f}",
    f"  Training time : {lstm_train_time:.1f}s over {len(lstm_history.history['loss'])} epochs",
    "",
    "GRU Results:",
    f"  MAE  : {gru_metrics['MAE']:.4f}",
    f"  RMSE : {gru_metrics['RMSE']:.4f}",
    f"  MAPE : {gru_metrics['MAPE (%)']:.2f}%",
    f"  R2   : {gru_metrics['R2']:.4f}",
    f"  Training time : {gru_train_time:.1f}s over {len(gru_history.history['loss'])} epochs",
    "",
    f"Best performing model (lower RMSE): {better_model}",
]
summary_text = "\n".join(summary_lines)
print(summary_text)

with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
    f.write(summary_text)

print(f"\nAll figures and tables saved to: {os.path.abspath(OUTPUT_DIR)}")
print("Script completed successfully.")
