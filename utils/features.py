"""
Centralized feature computation for stock prediction models.

Used by: preprocess_data.py, train_technical.py, train_hybrid.py,
         retrain_models.py, backtest.py, app.py

All training and prediction code imports feature lists from here
to guarantee consistency between training and inference.
"""
import numpy as np
import pandas as pd


# Canonical feature lists — training and app.py must use these exact lists.
TECHNICAL_FEATURES = [
    "rsi_14",
    "sma_50",
    "volume_obv",
    "macd_line",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "atr_14",
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_10d",
    "volume_ratio_20d",
]

HYBRID_FEATURES = TECHNICAL_FEATURES + ["sentiment_score"]


def compute_features(df):
    """
    Compute all technical features on a single-ticker OHLCV DataFrame.

    Expects columns: Date, Close, Volume.
    Optional columns: Open, High, Low (used for ATR; skipped if missing).

    Returns df with feature columns added. NaN rows from warmup
    are NOT dropped here — the caller decides when to dropna.
    """
    df = df.sort_values("Date").copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    has_hl = "High" in df.columns and "Low" in df.columns
    if has_hl:
        h = df["High"].astype(float)
        l = df["Low"].astype(float)

    # --- RSI-14 ---
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- SMA-50 ---
    df["sma_50"] = c.rolling(50).mean()

    # --- OBV (On-Balance Volume) ---
    df["volume_obv"] = (np.sign(c.diff()) * v).cumsum()

    # --- MACD (12, 26, 9) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20-day, 2 std) ---
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # --- ATR-14 (Average True Range) ---
    if has_hl:
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
    else:
        # Fallback: use daily range from close prices
        df["atr_14"] = c.diff().abs().rolling(14).mean()

    # --- Lagged Returns ---
    df["return_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)

    # --- Realized Volatility (10-day rolling std of daily returns) ---
    df["volatility_10d"] = c.pct_change(1).rolling(10).std()

    # --- Volume ratio (today's volume / 20-day average volume) ---
    vol_avg = v.rolling(20).mean()
    df["volume_ratio_20d"] = v / vol_avg.replace(0, np.nan)

    return df
