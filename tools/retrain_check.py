"""
tools/retrain_check.py
======================
Check whether the XGBoost model needs retraining.

Exits with code 1 if either:
  - model date_trained is older than 90 days
  - rolling accuracy on the most recent 30 days is below 0.47

Run from project root:
    python tools/retrain_check.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from config.settings import XGB_MODEL_PATH, XGB_RESULTS_PATH

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_MAX_AGE_DAYS     = 90    # retrain if model is older than this
_MIN_ACCURACY     = 0.47  # retrain if recent accuracy drops below this
_RECENCY_DAYS     = 30    # rolling window for recent accuracy


def _model_age_days(path: str) -> int | None:
    """Return days since the model was trained (UTC), or None if unavailable."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    from models.artifact_io import safe_load
    payload = safe_load(path)
    # Prefer precise UTC ISO timestamp; fall back to date-only string
    ts_str = payload.get("trained_at_utc") or payload.get("date_trained")
    if not ts_str:
        return None
    trained = pd.to_datetime(ts_str, utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    age = (now_utc - trained).days
    if age > 365 or age < -1:
        log.warning(
            "Suspicious model age: %d days (trained_at=%s). Possible clock skew.",
            age, ts_str,
        )
    return age


def _recent_accuracy(results_path: str, window_days: int = _RECENCY_DAYS) -> float | None:
    """Accuracy of the most recent `window_days` rows in the results CSV."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)
    df = pd.read_csv(results_path, parse_dates=["Date"])
    if df.empty:
        return None

    cutoff = df["Date"].max() - pd.Timedelta(days=window_days)
    recent = df[df["Date"] >= cutoff]
    if recent.empty or "label" not in recent.columns:
        return None

    y_true = recent["label"].astype(int).values
    if "Predicted" in recent.columns:
        y_pred = recent["Predicted"].astype(int).values
    elif "prob_up" in recent.columns:
        y_pred = np.where(recent["prob_up"].astype(float) > 0.5, 1, -1)
    else:
        return None

    return float(accuracy_score(y_true, y_pred))


def main() -> None:
    needs_retrain = False

    # Check model age
    try:
        age = _model_age_days(XGB_MODEL_PATH)
        if age is None:
            log.warning("No date_trained metadata in model payload.")
        else:
            log.info("Model age: %d days (threshold: %d)", age, _MAX_AGE_DAYS)
            if age > _MAX_AGE_DAYS:
                log.warning("Model is older than %d days — consider retraining.", _MAX_AGE_DAYS)
                needs_retrain = True
    except FileNotFoundError:
        log.error("Model not found at %s", XGB_MODEL_PATH)
        sys.exit(1)
    except Exception as exc:
        log.exception("Could not read model date: %s", exc)
        sys.exit(1)

    # Check recent rolling accuracy
    try:
        acc = _recent_accuracy(XGB_RESULTS_PATH)
        if acc is None:
            log.warning("Recent %d-day accuracy unavailable.", _RECENCY_DAYS)
        else:
            log.info("Recent %d-day accuracy = %.3f (threshold: %.2f)",
                     _RECENCY_DAYS, acc, _MIN_ACCURACY)
            if acc < _MIN_ACCURACY:
                log.warning("Recent accuracy %.3f < %.2f — consider retraining.",
                            acc, _MIN_ACCURACY)
                needs_retrain = True
    except FileNotFoundError:
        log.warning("Results CSV not found: %s", XGB_RESULTS_PATH)
    except Exception as exc:
        log.exception("Could not compute recent accuracy: %s", exc)
        needs_retrain = True

    if needs_retrain:
        log.error("Retrain check: NEEDS RETRAIN")
        sys.exit(1)
    log.info("Retrain check: OK — model is current")
    sys.exit(0)


if __name__ == "__main__":
    main()
