"""
prediction/adaptive_gate.py
============================
Regime-adaptive confidence gating for watchlist generation.

Confidence = max(P_down, P_up) in [0.5, 1.0] because probabilities sum to 1.
The meaningful base threshold is WATCHLIST_MIN_CONFIDENCE (0.60), which
filters approximately 18% of rows.

Threshold adjustments
---------------------
  Bull  (score > +0.20)  base - REGIME_BULL_ADJ  e.g. 0.60 -> 0.55
  Neutral                base                     0.60
  Bear  (score < -0.20)  base + REGIME_BEAR_ADJ  e.g. 0.60 -> 0.68
  Hard clamps: [REGIME_MIN_THRESH=0.52, REGIME_MAX_THRESH=0.75]

Regime score is derived from regime columns already in merged_final.csv:
  bullish_regime, bearish_regime, nifty_trend_20d,
  rolling_volatility_20d, market_return_20d
"""

import logging
import os

import numpy as np
import pandas as pd

from config.settings import (
    MERGED_CSV,
    REGIME_BEAR_ADJ, REGIME_BULL_ADJ,
    REGIME_MAX_THRESH, REGIME_MIN_THRESH,
    WATCHLIST_MIN_CONFIDENCE,
)

log = logging.getLogger(__name__)

BASE_THRESHOLD = WATCHLIST_MIN_CONFIDENCE
_BULL_CUTOFF = +0.20
_BEAR_CUTOFF = -0.20
_W_REGIME_FLAGS = 0.50
_W_TREND = 0.30
_W_VOL = 0.20

_REGIME_FEATURE_COLS = [
    "bullish_regime", "bearish_regime",
    "nifty_trend_20d", "rolling_volatility_20d", "market_return_20d",
]


def compute_regime_score(row: pd.Series) -> float:
    """Weighted regime score in [-1, +1]. Positive = bullish."""
    score = 0.0

    bullish = float(row.get("bullish_regime", 0) or 0)
    bearish = float(row.get("bearish_regime", 0) or 0)
    score += _W_REGIME_FLAGS * (bullish - bearish)

    trend = row.get("nifty_trend_20d", np.nan)
    if trend is not None and np.isfinite(float(trend)):
        score += _W_TREND * float(np.clip(trend, -1, 1))

    vol = row.get("rolling_volatility_20d", np.nan)
    mret = row.get("market_return_20d", np.nan)
    if (vol is not None and mret is not None
            and np.isfinite(float(vol)) and np.isfinite(float(mret))
            and float(vol) > 0):
        vol_signal = np.sign(float(mret)) * min(1.0, 0.015 / (float(vol) + 1e-8))
        score += _W_VOL * float(np.clip(vol_signal, -1, 1))

    return float(np.clip(score, -1.0, 1.0))


def adaptive_threshold(regime_score: float) -> float:
    """Map regime score to a confidence threshold.

    Bull (score > 0): lower threshold (more aggressive, wider net).
    Bear (score < 0): raise threshold (more conservative).
    """
    if regime_score > 0:
        # Bull: subtract proportional adjustment → threshold drops.
        adjustment = -REGIME_BULL_ADJ * regime_score
    else:
        # Bear: add proportional adjustment → threshold rises.
        adjustment = REGIME_BEAR_ADJ * abs(regime_score)
    return float(np.clip(BASE_THRESHOLD + adjustment, REGIME_MIN_THRESH, REGIME_MAX_THRESH))


def _regime_label(regime_score: float) -> str:
    if regime_score > _BULL_CUTOFF:
        return "BULL"
    if regime_score < _BEAR_CUTOFF:
        return "BEAR"
    return "NEUTRAL"


def get_current_threshold(df: pd.DataFrame = None) -> tuple[float, float, str]:
    """
    Compute adaptive threshold from the latest date in merged data.

    Returns
    -------
    (threshold, regime_score, regime_label)
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        if not os.path.exists(MERGED_CSV):
            return BASE_THRESHOLD, 0.0, "NEUTRAL"
        df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])

    if df.empty:
        return BASE_THRESHOLD, 0.0, "NEUTRAL"

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    latest_rows = df[df["Date"] == df["Date"].max()]
    present_cols = [c for c in _REGIME_FEATURE_COLS if c in latest_rows.columns]

    if not present_cols:
        log.warning("No regime columns found in merged data. Using default threshold.")
        return BASE_THRESHOLD, 0.0, "NEUTRAL"

    avg_row = latest_rows[present_cols].mean()
    regime_score = compute_regime_score(avg_row)
    threshold = adaptive_threshold(regime_score)
    label = _regime_label(regime_score)

    log.info("Adaptive gate | date=%s  regime=%s (score=%+.3f)  threshold=%.3f",
             df["Date"].max().date(), label, regime_score, threshold)
    return threshold, regime_score, label


def explain_threshold(threshold: float, regime_score: float, regime_label: str) -> str:
    """Human-readable explanation of the current threshold adjustment."""
    delta = threshold - BASE_THRESHOLD
    direction = "lowered" if delta < 0 else ("raised" if delta > 0 else "unchanged")
    lines = [
        f"Market Regime   : {regime_label}  (score={regime_score:+.2f})",
        f"Confidence Gate : {direction} from "
        f"{BASE_THRESHOLD:.2f} to {threshold:.2f}  (delta={delta:+.2f})",
    ]
    if regime_label == "BULL":
        lines.append("-> Low volatility bull market. Threshold lowered to widen net.")
    elif regime_label == "BEAR":
        lines.append("-> High volatility bear market. Threshold raised for high-conviction only.")
    else:
        lines.append("-> Neutral regime. Using base confidence threshold.")
    return "\n".join(lines)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="  [%(levelname)s] %(message)s")
    threshold, score, label = get_current_threshold()
    print("\n" + "=" * 50)
    print("  ADAPTIVE CONFIDENCE GATE")
    print("=" * 50)
    print(explain_threshold(threshold, score, label))
    print("=" * 50 + "\n")
