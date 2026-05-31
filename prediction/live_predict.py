"""
prediction/live_predict.py
===========================
Fetch recent OHLCV data via yfinance, run the feature pipeline on the fresh
technical snapshot, and produce per-stock live predictions from the saved
XGBoost model.

Behaviour
---------
* Downloads ~400 days per stock so that 252-day 52w-high, 60-day z-scores
    and 20-day rolling features all have enough warm-up.
* Temporarily replaces technical.csv and merged_final.csv while the feature
    pipeline runs, then restores the originals from backup.
* Returns {stock: {direction, prob_up, confidence, edge}}.
"""

import json
import logging
import os
import shutil
import time

import pandas as pd
import yfinance as yf

from config.settings import MERGED_CSV, STOCKS_NS, TECHNICAL_CSV
from features.build_features import run as build_features_run
from models.xgboost.predict import load_xgb, predict_label

log = logging.getLogger(__name__)

# Largest look-back used in feature engineering is 252 (52-week high). We add
# a buffer so that warm-up of 20-day stds, 60-day z-scores and EWM 26/9 macros
# all settle before the most recent prediction date.
_FETCH_PERIOD = "400d"


def _fetch_one(symbol: str) -> pd.DataFrame:
    """Download enough days of OHLCV + technical indicators for one symbol."""
    try:
        raw = yf.download(symbol, period=_FETCH_PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in raw.columns for c in required):
        return pd.DataFrame()

    df = raw[required].copy().dropna(subset=["Close"])
    if len(df) < 20:
        return pd.DataFrame()

    try:
        import ta
        df["EMA_20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd = ta.trend.MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["ATR"] = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"]).average_true_range()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["Close"], volume=df["Volume"]).on_balance_volume()
    except Exception:
        # Minimal fallback without the `ta` library
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["RSI"] = 50.0  # neutral placeholder
        df["MACD"] = (df["Close"].ewm(span=12, adjust=False).mean()
                       - df["Close"].ewm(span=26, adjust=False).mean())
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean().fillna(0)
        df["OBV"] = (df["Volume"] * df["Close"].pct_change().fillna(0).gt(0).astype(int)).cumsum()

    df["Return_1d"] = df["Close"].pct_change()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Stock"] = symbol.replace(".NS", "")

    keep_cols = ["Date", "Stock", "Open", "High", "Low", "Close", "Volume",
                 "EMA_20", "RSI", "MACD", "MACD_signal", "ATR", "OBV", "Return_1d"]
    return df[[c for c in keep_cols if c in df.columns]]


def live_predict(save_json: str = None) -> dict:
    """
    Fetch live data, rebuild features, and run XGBoost inference.

    Parameters
    ----------
    save_json : optional path to write results as JSON.

    Returns
    -------
    dict mapping stock -> {direction, prob_up, confidence, edge}
    """
    tech_backup = None
    merged_backup = None

    # Back up existing files so the pipeline can write fresh ones
    # Use a timestamped backup name so a prior crashed run's recovery file
    # is never overwritten by the current run's backup-of-truncated-data.
    import time as _time_mod
    _stamp = _time_mod.strftime("%Y%m%d_%H%M%S")
    if os.path.exists(TECHNICAL_CSV):
        tech_backup = f"{TECHNICAL_CSV}.{_stamp}.bak"
        shutil.copy2(TECHNICAL_CSV, tech_backup)
    if os.path.exists(MERGED_CSV):
        merged_backup = f"{MERGED_CSV}.{_stamp}.bak"
        shutil.copy2(MERGED_CSV, merged_backup)

    try:
        parts = []
        for sym in STOCKS_NS:
            print(f"Fetching {sym}...", end=" ", flush=True)
            part = _fetch_one(sym)
            if part is None or part.empty:
                print("[skip]")
                continue
            parts.append(part)
            print(f"ok ({len(part)} rows)")
            time.sleep(0.2)   # gentle rate limiting

        if not parts:
            raise RuntimeError("No technical data downloaded.")

        all_tech = pd.concat(parts, ignore_index=True)
        os.makedirs(os.path.dirname(TECHNICAL_CSV), exist_ok=True)
        all_tech.to_csv(TECHNICAL_CSV, index=False)

        merged = build_features_run()

        latest = merged.sort_values("Date").groupby("Stock", as_index=False).last()
        payload = load_xgb()
        labels, proba = predict_label(latest, payload)

        results = {}
        for i, stock in enumerate(latest["Stock"].tolist()):
            p_up = float(proba[i, 1])
            results[stock] = {
                "direction": int(labels[i]),
                "prob_up": p_up,
                "confidence": float(proba[i].max()),
                "edge": p_up - 0.5,
            }

        if save_json:
            with open(save_json, "w") as fh:
                json.dump(results, fh, indent=2)

        return results

    finally:
        # Always restore backups, even if an exception occurred
        try:
            if tech_backup and os.path.exists(tech_backup):
                shutil.move(tech_backup, TECHNICAL_CSV)
            if merged_backup and os.path.exists(merged_backup):
                shutil.move(merged_backup, MERGED_CSV)
        except Exception as exc:
            log.exception("Failed to restore backups in live_predict: %s", exc)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="  [%(levelname)s] %(message)s")
    results = live_predict()
    print(json.dumps(results, indent=2))
