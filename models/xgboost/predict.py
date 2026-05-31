"""
models/xgboost/predict.py
=========================
Inference helpers for the binary XGBoost model (DOWN / UP only).

Public API
----------
predict_proba(df, payload)  ->  ndarray (N, 2): [P(DOWN), P(UP)]
predict_label(df, payload)  ->  (ext_labels, proba)  ext_labels in {-1, 1}
edge_score(df, payload)     ->  P(UP) - 0.5
rank_signals(df, payload)   ->  df with signal columns, sorted by edge
"""

import logging
import os
import numpy as np
import pandas as pd

from config.settings import XGB_MODEL_PATH

log = logging.getLogger(__name__)

_INT_TO_EXT = {0: -1, 1: 1}
NUM_CLASS = 2


def load_xgb(path: str = XGB_MODEL_PATH) -> dict:
    """Load pickled model payload. Raises FileNotFoundError if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"XGBoost model not found at '{path}'. "
            "Run: python models/xgboost/train.py"
        )
    # R2: verify SHA-256 sidecar before unpickling
    from models.artifact_io import safe_load
    payload = safe_load(path)

    if "date_trained" in payload and payload["date_trained"]:
        log.info("XGBoost model trained on: %s", payload["date_trained"])
    log.info("Decision threshold: %.2f", payload.get("prediction_threshold", 0.5))

    for key in ("model", "feature_names", "train_medians"):
        if key not in payload:
            raise RuntimeError(f"Payload missing required key: '{key}'")
    return payload


def build_x(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """Align DataFrame to model's expected features. R5: warn on schema drift."""
    feature_names = payload["feature_names"]
    train_medians = payload.get("train_medians", {})

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        log.warning(
            "R5: %d feature column(s) missing at inference: %s. "
            "Filling with train medians; may indicate schema drift.",
            len(missing_cols), missing_cols,
        )

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else train_medians.get(col, 0.0)

    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    fill = pd.Series({c: train_medians.get(c, 0.0) for c in X.columns})
    X = X.fillna(fill).fillna(0.0)
    return X


def predict_proba(df: pd.DataFrame, payload: dict = None) -> np.ndarray:
    """Return (N, 2) probability array: [P(DOWN), P(UP)]."""
    if payload is None:
        payload = load_xgb()
    if len(df) == 0:
        return np.empty((0, NUM_CLASS), dtype=np.float32)
    return payload["model"].predict_proba(build_x(df, payload))


def predict_label(df: pd.DataFrame, payload: dict = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ext_labels, proba).
    ext_labels in {-1, 1} (external label space).
    """
    if payload is None:
        payload = load_xgb()
    proba = predict_proba(df, payload)
    threshold = float(payload.get("prediction_threshold", 0.5))
    int_labels = np.where(proba[:, 1] >= threshold, 1, 0)
    ext_labels = np.vectorize(_INT_TO_EXT.get)(int_labels)
    return ext_labels, proba


def edge_score(df: pd.DataFrame, payload: dict = None) -> np.ndarray:
    """P(UP) - 0.5. Positive = bullish signal."""
    proba = predict_proba(df, payload)
    return proba[:, 1] - 0.5


def rank_signals(df: pd.DataFrame, payload: dict = None) -> pd.DataFrame:
    """Return df with prob_down, prob_up, edge, signal columns sorted by edge."""
    if payload is None:
        payload = load_xgb()
    proba = predict_proba(df, payload)
    out = df.copy()
    out["prob_down"] = proba[:, 0]
    out["prob_up"] = proba[:, 1]
    out["edge"] = proba[:, 1] - 0.5

    strong_threshold = out["edge"].quantile(0.95)
    buy_threshold = out["edge"].quantile(0.80)

    out["signal"] = "WATCH"
    out.loc[out["edge"] >= buy_threshold, "signal"] = "BUY"
    out.loc[out["edge"] >= strong_threshold, "signal"] = "STRONG_BUY"
    out.loc[out["edge"] <= -buy_threshold, "signal"] = "SELL"
    return out.sort_values("edge", ascending=False)


def get_feature_importance(payload: dict = None, top_n: int = 10) -> dict:
    """Feature importances from the base (uncalibrated) model."""
    if payload is None:
        payload = load_xgb()
    base = payload.get("base_model")
    if base is None:
        return {}
    try:
        pairs = sorted(
            zip(payload["feature_names"], base.feature_importances_),
            key=lambda x: -x[1]
        )[:top_n]
        return {name: float(score) for name, score in pairs}
    except Exception:
        return {}
