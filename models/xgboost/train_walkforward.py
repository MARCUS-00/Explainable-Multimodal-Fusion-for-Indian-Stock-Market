"""models/xgboost/train_walkforward.py
====================================
Expanding-window retraining for XGBoost.

For each window:
  - fit on rows with Date < current window start
  - median-impute using train medians only
  - drop constant columns in lockstep across train and prediction sets
  - predict the next `step` calendar days
  - write window metadata (window_id, fit_start, fit_end)

CLI:
  --step
  --threshold
  --test-start
  --out
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config.settings import MERGED_CSV, RANDOM_SEED, TEST_START, XGBOOST_FEATURES, XGBOOST_PARAMS

logging.basicConfig(level=logging.INFO, format=" [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXT_TO_INT = {-1: 0, 1: 1}
INT_TO_EXT = {0: -1, 1: 1}


def _load_data() -> pd.DataFrame:
    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(f"{MERGED_CSV} not found. Run features/build_features.py first.")
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    df = df[df["label"] != 0].copy()
    return df.sort_values(["Date", "Stock"]).reset_index(drop=True)


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).map(EXT_TO_INT).values
    return X, y, available


def _build_params() -> dict:
    params = {k: v for k, v in XGBOOST_PARAMS.items()
              if k not in ("scale_pos_weight", "eval_metric", "use_label_encoder", "num_class")}
    params.update({"objective": "binary:logistic", "seed": RANDOM_SEED, "tree_method": "hist", "n_jobs": -1})
    return params


def walkforward(step: int = 30, threshold: float = 0.5, test_start: str = TEST_START, out: str | None = None) -> pd.DataFrame:
    df = _load_data()
    feature_cols = [c for c in XGBOOST_FEATURES if c in df.columns]
    if not feature_cols:
        raise RuntimeError("No XGBOOST_FEATURES available in merged dataset.")

    start_ts = pd.to_datetime(test_start)
    end_ts = df["Date"].max()
    cur = start_ts
    window_id = 0
    rows = []

    while cur <= end_ts:
        window_end = cur + pd.Timedelta(days=step)
        train_df = df[df["Date"] < cur].copy()
        pred_df = df[(df["Date"] >= cur) & (df["Date"] < window_end)].copy()
        if train_df.empty or pred_df.empty:
            cur = window_end
            continue

        X_train, y_train, feats = _prepare_xy(train_df, feature_cols)
        x_pred, _, _ = _prepare_xy(pred_df, feats)

        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians).fillna(0.0)
        x_pred = x_pred.fillna(train_medians).fillna(0.0)

        constant_cols = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
        if constant_cols:
            X_train = X_train.drop(columns=constant_cols)
            x_pred = x_pred.drop(columns=constant_cols)
        if X_train.shape[1] == 0:
            raise RuntimeError(f"All features were constant in window {window_id}.")

        model = XGBClassifier(**_build_params())
        model.fit(X_train, y_train, verbose=False)

        proba = model.predict_proba(x_pred)
        pred_int = (proba[:, 1] >= threshold).astype(int)
        pred_ext = np.vectorize(INT_TO_EXT.get)(pred_int)

        out_df = pred_df[["Date", "Stock", "label"]].copy().reset_index(drop=True)
        out_df["Predicted"] = pred_ext
        out_df["prob_down"] = proba[:, 0]
        out_df["prob_up"] = proba[:, 1]
        out_df["Confidence"] = proba.max(axis=1)
        out_df["window_id"] = window_id
        out_df["fit_start"] = train_df["Date"].min().strftime("%Y-%m-%d")
        out_df["fit_end"] = (cur - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if "Return_1d" in pred_df.columns:
            out_df["Return_1d"] = pred_df["Return_1d"].values

        y_true_ext = pred_df["label"].astype(int).values
        y_true_bin = (y_true_ext == 1).astype(int)
        acc = accuracy_score(y_true_ext, pred_ext)
        auc = roc_auc_score(y_true_bin, proba[:, 1]) if len(np.unique(y_true_bin)) > 1 else float("nan")
        log.info(
            "Window %02d fit=[%s..%s] n_train=%d n_pred=%d acc=%.3f auc=%.3f",
            window_id,
            train_df["Date"].min().date(),
            (cur - pd.Timedelta(days=1)).date(),
            len(train_df),
            len(pred_df),
            acc,
            auc,
        )

        rows.append(out_df)
        cur = window_end
        window_id += 1

    if not rows:
        raise RuntimeError("No walk-forward windows produced.")

    result = pd.concat(rows, ignore_index=True)
    out_path = out or os.path.join("evaluation", "results", "xgboost_walkforward.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved walkforward predictions -> {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-start", type=str, default=TEST_START)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    walkforward(step=args.step, threshold=args.threshold, test_start=args.test_start, out=args.out)


if __name__ == "__main__":
    main()
