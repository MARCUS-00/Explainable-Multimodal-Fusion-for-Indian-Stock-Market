"""
prediction/single_stock.py
==========================
Single-stock prediction entry point.

Returns UP / DOWN with confidence, expected movement, recommendation, and
plain-English XAI factor bullets.
"""

import logging

import pandas as pd

from config.settings import MERGED_CSV
from models.xgboost.predict import load_xgb, predict_label as _xgb_predict_label, predict_proba
from prediction.watchlist import confidence_label, expected_movement, recommendation
from xai.shap_explain import build_explanation_bullets

log = logging.getLogger(__name__)


def _format_output(result: dict) -> str:
    bullets = result.get("XAI_Factors", []) or []
    lines = [
        f"{result.get('Stock', '')} - {result.get('Prediction', '')}",
        f"Expected Movement : {result.get('Expected_Movement', '')}",
        f"Confidence        : {result.get('Confidence', '')} ({result.get('Confidence_Level', '')})",
        f"Recommendation    : {result.get('Recommendation', '')}",
        f"Last Close        : {result.get('Last_Close', '')}  Date: {result.get('Last_Date', '')}",
    ]
    if bullets:
        lines.append("XAI Factors:")
        lines.extend(f"  - {b}" for b in bullets)
    return "\n".join(lines)


def predict_single(symbol: str, df: pd.DataFrame = None) -> dict:
    """
    Predict the next LABEL_HORIZON-day direction for a single stock.

    Parameters
    ----------
    symbol : ticker string (e.g. "INFY").
    df     : pre-loaded merged DataFrame; loaded from disk if None.

    Returns
    -------
    dict with keys: Stock, Prediction, Expected_Movement, Confidence,
    Confidence_Level, Recommendation, Last_Date, Last_Close, XAI_Factors.
    On error, returns {"error": "<message>"}.
    """
    # Load data
    if df is None or (hasattr(df, "empty") and df.empty):
        import os
        if not os.path.exists(MERGED_CSV):
            return {"error": "merged_final.csv not found. Run: python features/build_features.py"}
        try:
            df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
        except Exception as exc:
            return {"error": f"Cannot load dataset: {exc}"}

    sdf = df[df["Stock"] == symbol].sort_values("Date").reset_index(drop=True)
    if sdf.empty:
        return {"error": f"No data for '{symbol}'. Check the symbol is in merged_final.csv."}

    # Load model
    try:
        payload = load_xgb()
    except Exception as exc:
        return {"error": f"XGBoost load failed: {exc}"}

    # Run inference
    try:
        latest = sdf.iloc[-1]
        proba = predict_proba(sdf.tail(1), payload)[0]
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    ext_labels, _ = _xgb_predict_label(sdf.tail(1), payload)
    direction = "UP" if int(ext_labels[0]) == 1 else "DOWN"
    conf = float(proba.max())
    bullets = build_explanation_bullets(latest, payload, max_bullets=5)

    result = {
        "Stock": symbol,
        "Prediction": direction,
        "Expected_Movement": expected_movement(latest, direction),
        "Confidence": f"{conf * 100:.1f}%",
        "Confidence_Level": confidence_label(conf),
        "Recommendation": recommendation(direction, conf),
        "Last_Date": str(latest.get("Date", "N/A")),
        "Last_Close": f"Rs.{float(latest.get('Close', 0)):.2f}",
        "XAI_Factors": bullets,
    }
    print(_format_output(result))
    return result


if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="  [%(levelname)s] %(message)s")
    sym = sys.argv[1] if len(sys.argv) > 1 else "INFY"
    predict_single(sym)
