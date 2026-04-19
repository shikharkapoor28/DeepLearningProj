import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns like ('Close','SPY').
    Flatten to single level (e.g. 'Close') to simplify downstream code.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal technical feature set without external dependencies:
    - return_1p, volatility(20), RSI(14), MACD(12,26,9), turbulence (vol z-score)
    Expects columns: Open, High, Low, Close, Volume
    """
    out = df.copy()

    out["return_1p"] = out["Close"].pct_change()
    out["volatility"] = out["return_1p"].rolling(window=20).std()

    # RSI(14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12,26,9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    vol_roll = out["volatility"].rolling(100)
    out["turbulence"] = (out["volatility"] - vol_roll.mean()) / (vol_roll.std() + 1e-12)

    out.dropna(inplace=True)
    return out


def robust_tanh_norm(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    mad = mad if mad != 0 else 1e-12
    z = (s - med) / (mad * 1.4826)
    return np.tanh(z)


def build_feature_matrix(feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    cols = ["return_1p", "volatility", "rsi_14", "macd", "turbulence", "Volume"]
    if feature_df.empty:
        raise ValueError("feature_df is empty (not enough history after indicator windows).")
    for c in cols:
        if c not in feature_df.columns:
            raise ValueError(f"Missing required column {c!r} for features.")

    mat_df = feature_df.copy()
    for c in cols:
        mat_df[f"{c}_norm"] = robust_tanh_norm(mat_df[c].astype(float))

    norm_cols = [f"{c}_norm" for c in cols]
    feature_matrix = mat_df[norm_cols].to_numpy(dtype=np.float32)
    close_prices = mat_df["Close"].to_numpy(dtype=np.float64)
    if len(close_prices) == 0:
        raise ValueError("No rows left after normalization; need more history.")

    latest_raw = {
        "price": float(mat_df["Close"].iloc[-1]),
        "volume_1h": float(mat_df["Volume"].iloc[-1]),
        "rsi": float(mat_df["rsi_14"].iloc[-1]),
        "macd": float(mat_df["macd"].iloc[-1]),
        "turbulence": float(mat_df["turbulence"].iloc[-1]),
    }
    return feature_matrix, close_prices, latest_raw


def json_safe(x):
    """
    Convert values to JSON-safe primitives.
    In particular: NaN/Infinity are not valid JSON and will crash browser JSON.parse.
    """
    if x is None:
        return None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (int, str, bool)):
        return x
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    try:
        if hasattr(x, "item"):
            return json_safe(x.item())
    except Exception:
        pass
    return str(x)

