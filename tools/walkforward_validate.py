"""tools/walkforward_validate.py
================================
Slide 30-day windows across VAL+TEST and compute per-window metrics.

Writes evaluation/results/walkforward.csv with columns:
  start, end, phase, n, acc, auc, brier, f1_macro
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from config.settings import ENSEMBLE_RESULTS_PATH, LSTM_RESULTS_PATH, TEST_START, VAL_END, VAL_START, XGB_RESULTS_PATH

WALKFORWARD_RESULTS_PATH = os.path.join("evaluation", "results", "walkforward.csv")


def _load_predictions() -> pd.DataFrame:
    for path in (ENSEMBLE_RESULTS_PATH, XGB_RESULTS_PATH, LSTM_RESULTS_PATH):
        val_path = path.replace(".csv", "_val.csv")
        if os.path.exists(path) and os.path.exists(val_path):
            test_df = pd.read_csv(path, parse_dates=["Date"])
            val_df = pd.read_csv(val_path, parse_dates=["Date"])
            combined = pd.concat([val_df, test_df], ignore_index=True, sort=False)
            combined["__source"] = os.path.basename(path)
            if "lstm_prob_up" in combined.columns and "prob_up" not in combined.columns:
                combined = combined.rename(columns={
                    "lstm_prob_up": "prob_up",
                    "lstm_prob_down": "prob_down",
                    "lstm_pred": "Predicted",
                })
            return combined.sort_values("Date").reset_index(drop=True)
    raise FileNotFoundError("No prediction pair found. Need a *_val.csv and matching test CSV.")


def _score_window(df_win: pd.DataFrame) -> dict:
    y_true = df_win["label"].astype(int).values
    y_true_bin = (y_true == 1).astype(int)
    probs = df_win["prob_up"].astype(float).values if "prob_up" in df_win.columns else np.zeros(len(df_win))
    preds = df_win["Predicted"].astype(int).values if "Predicted" in df_win.columns else np.where(probs >= 0.5, 1, -1)
    auc = roc_auc_score(y_true_bin, probs) if len(np.unique(y_true_bin)) > 1 else float("nan")
    return {
        "n": len(df_win),
        "acc": float(accuracy_score(y_true, preds)),
        "auc": float(auc),
        "brier": float(brier_score_loss(y_true_bin, probs)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }


def _window_phase(start: pd.Timestamp, end: pd.Timestamp, test_start_ts: pd.Timestamp, val_end_ts: pd.Timestamp) -> str:
    if end <= val_end_ts:
        return "VAL"
    if start >= test_start_ts:
        return "TEST"
    return "MIXED"


def _window_row(df: pd.DataFrame, cur: pd.Timestamp, window_days: int,
                test_start_ts: pd.Timestamp, val_end_ts: pd.Timestamp) -> dict | None:
    end_win = cur + pd.Timedelta(days=window_days)
    win = df[(df["Date"] >= cur) & (df["Date"] < end_win)]
    if win.empty or "label" not in win.columns:
        return None
    metrics = _score_window(win)
    metrics.update({
        "start": cur.strftime("%Y-%m-%d"),
        "end": (end_win - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "phase": _window_phase(cur, end_win, test_start_ts, val_end_ts),
    })
    return metrics


def run(window_days: int = 30, step_days: int = 30) -> pd.DataFrame:
    df = _load_predictions()
    cur = pd.to_datetime(VAL_START)
    end = df["Date"].max()
    test_start_ts = pd.to_datetime(TEST_START)
    val_end_ts = pd.to_datetime(VAL_END)

    rows = []
    while cur <= end:
        metrics = _window_row(df, cur, window_days, test_start_ts, val_end_ts)
        if metrics is not None:
            rows.append(metrics)
        cur = cur + pd.Timedelta(days=step_days)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(WALKFORWARD_RESULTS_PATH), exist_ok=True)
    result.to_csv(WALKFORWARD_RESULTS_PATH, index=False)
    print(f"Saved walkforward eval -> {WALKFORWARD_RESULTS_PATH}")
    return result


if __name__ == "__main__":
    run()
