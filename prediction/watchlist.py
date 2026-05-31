"""
prediction/watchlist.py
=======================
Generates the daily top-N watchlist using XGBoost predictions.

Binary contract:
  proba[:, 0] = P(DOWN)
  proba[:, 1] = P(UP)
"""

import logging
import os

import numpy as np
import pandas as pd

from config.settings import (
    MERGED_CSV, STOCKS, WATCHLIST_MIN_CONFIDENCE, WATCHLIST_OUTPUT_PATH,
)
from models.xgboost.predict import load_xgb, predict_label as _xgb_predict_label, predict_proba as xgb_proba
from prediction.adaptive_gate import explain_threshold, get_current_threshold
from xai.shap_explain import build_explanation_bullets

log = logging.getLogger(__name__)

_INT_TO_DIR = {0: "DOWN", 1: "UP"}


def confidence_label(conf: float) -> str:
    if conf >= 0.60:
        return "HIGH"
    if conf >= 0.45:
        return "MEDIUM"
    return "LOW"


def recommendation(direction: str, conf: float) -> str:
    if direction == "UP":
        return "BUY" if conf >= 0.45 else "WATCH"
    if direction == "DOWN":
        return "AVOID" if conf >= 0.45 else "WATCH"
    return "WATCH"


def expected_movement(row: pd.Series, direction: str) -> str:
    """Heuristic expected movement string based on probability edge."""
    mag = abs(float(row.get("prob_up", 0.5)) - float(row.get("prob_down", 0.5)))
    pct = 0.5 + 5.0 * mag
    return f"+{pct:.2f}%" if direction == "UP" else f"-{pct:.2f}%"


def generate_watchlist(df: pd.DataFrame = None,
                        top_n: int = 10,
                        min_confidence: float = None,
                        direction: str = "up") -> pd.DataFrame:
    """
    Generate ranked watchlist for the latest available date.

    Parameters
    ----------
    df             : pre-loaded merged DataFrame (loaded from disk if None).
    top_n          : number of top candidates to return.
    min_confidence : override adaptive regime threshold when provided.
    direction      : 'up' (default, long-only), 'down', or 'both'.
                     'up' filters to UP-direction predictions only; the CLI
                     default preserves the original long-only behaviour.

    Returns
    -------
    DataFrame with one row per stock, sorted by conviction descending.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        if not os.path.exists(MERGED_CSV):
            log.warning("merged_final.csv not found.")
            return pd.DataFrame()
        df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    latest = (df.sort_values("Date")
                .groupby("Stock", as_index=False)
                .last())

    universe = set(STOCKS)
    latest = latest[latest["Stock"].isin(universe)].reset_index(drop=True)
    if latest.empty:
        log.warning("No matching stocks found in merged data.")
        return pd.DataFrame()

    try:
        payload = load_xgb()
    except Exception as exc:
        log.exception("Cannot load XGBoost model: %s", exc)
        return pd.DataFrame()

    try:
        proba = xgb_proba(latest, payload)   # (N, 2): [P(DOWN), P(UP)]
    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        return pd.DataFrame()

    ext_labels, _ = _xgb_predict_label(latest, payload)
    int_labels = np.where(ext_labels == 1, 1, 0)
    latest["prob_down"] = proba[:, 0]
    latest["prob_up"] = proba[:, 1]
    latest["Direction"] = np.vectorize(_INT_TO_DIR.get)(int_labels)
    latest["conviction"] = (latest["prob_up"] - 0.5).abs()
    latest["Confidence_raw"] = proba.max(axis=1)

    # Adaptive confidence gate
    if min_confidence is None:
        try:
            threshold, regime_score, regime_label = get_current_threshold(df)
            log.info(explain_threshold(threshold, regime_score, regime_label).replace("\n", " | "))
        except Exception as exc:
            log.warning("Adaptive gate failed (%s); using fixed threshold.", exc)
            threshold = WATCHLIST_MIN_CONFIDENCE
    else:
        threshold = float(min_confidence)

    gated = latest[latest["Confidence_raw"] >= threshold].copy()

    # Filter by requested direction; fall back gracefully when pool is empty
    _dir = direction.lower()
    if _dir == "up":
        dir_mask = gated["Direction"] == "UP"
    elif _dir == "down":
        dir_mask = gated["Direction"] == "DOWN"
    else:  # both
        dir_mask = pd.Series(True, index=gated.index)
    top = gated[dir_mask].nlargest(top_n, "conviction").reset_index(drop=True)
    if top.empty:
        log.warning(
            "No candidates passed direction=%r filter at threshold %.3f; "
            "falling back to ALL gated directions.",
            _dir, threshold,
        )
        top = gated.nlargest(top_n, "conviction").reset_index(drop=True)
    if top.empty:
        log.warning(
            "No candidates passed confidence threshold %.3f; "
            "falling back to highest-conviction predictions regardless of gate.",
            threshold,
        )
        top = latest.nlargest(top_n, "conviction").reset_index(drop=True)

    rows = []
    for _, row in top.iterrows():
        pred_dir = str(row["Direction"])
        conf = float(row["Confidence_raw"])
        try:
            bullets = build_explanation_bullets(row, payload, max_bullets=5)
        except Exception:
            bullets = []
        rows.append({
            "Stock": row["Stock"],
            "Prediction": pred_dir,
            "Expected_Movement": expected_movement(row, pred_dir),
            "Confidence": f"{conf * 100:.1f}%",
            "Confidence_Level": confidence_label(conf),
            "Recommendation": recommendation(pred_dir, conf),
            "Last_Date": str(pd.to_datetime(row.get("Date")).date()),
            "Last_Close": f"Rs.{float(row.get('Close', 0)):.2f}",
            "XAI_Factors": " | ".join(bullets),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        try:
            os.makedirs(os.path.dirname(WATCHLIST_OUTPUT_PATH), exist_ok=True)
            out.to_csv(WATCHLIST_OUTPUT_PATH, index=False)
            log.info("Watchlist saved -> %s", WATCHLIST_OUTPUT_PATH)
            xlsx_path = WATCHLIST_OUTPUT_PATH.replace(".csv", ".xlsx")
            out.to_excel(xlsx_path, index=False)
            log.info("Watchlist XLSX saved -> %s", xlsx_path)
        except Exception as exc:
            log.warning("Could not save watchlist: %s", exc)

    return out


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="  [%(levelname)s] %(message)s")
    wl = generate_watchlist()
    print(wl.to_string(index=False) if not wl.empty else "No watchlist produced.")
