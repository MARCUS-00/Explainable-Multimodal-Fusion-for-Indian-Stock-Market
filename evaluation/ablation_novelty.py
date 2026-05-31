"""
evaluation/ablation_novelty.py
================================
Ablation study: Baseline vs Novelty on test set (2025).

Four configurations:
  A  Baseline      : fixed k=5 decay, fixed threshold=0.60
  B  Learned Decay : per-stock learned decay, fixed threshold=0.60
  C  Adaptive Gate : fixed k=5 decay, adaptive threshold
  D  Both          : per-stock learned decay + adaptive threshold

Run: python evaluation/ablation_novelty.py
Output: evaluation/results/ablation_results.csv
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

from config.settings import (
    MERGED_CSV, TEST_START, XGB_MODEL_PATH,
    WATCHLIST_MIN_CONFIDENCE,
)

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_model():
    if not os.path.exists(XGB_MODEL_PATH):
        print(f"  [WARN] {XGB_MODEL_PATH} not found. Run models/xgboost/train.py first.")
        return None
    from models.artifact_io import safe_load
    return safe_load(XGB_MODEL_PATH)


def _predict(df, payload):
    from models.xgboost.predict import predict_proba, predict_label
    proba  = predict_proba(df, payload)
    labels, _ = predict_label(df, payload)
    return proba, labels


def _compute_news_decay_fixed(df: pd.DataFrame, k: int) -> pd.Series:
    """
    Recompute news_decay with a fixed k.
    news_score=0 on no-news days -> replace with NaN before ffill so last
    real sentiment is propagated, not zero.
    """
    last_score = df.groupby("Stock")["news_score"].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0.0)
    )
    return last_score * np.exp(-df["days_since_news"].fillna(999.0) / k)


def _compute_news_decay_personalised(df: pd.DataFrame) -> pd.Series:
    """
    Recompute news_decay with per-stock learned decay constants.
    Same 0->NaN->ffill fix as _compute_news_decay_fixed.
    """
    from features.sentiment_decay import apply_personalised_decay, DECAY_CONSTANTS_PATH
    import json
    if not os.path.exists(DECAY_CONSTANTS_PATH):
        log.warning("decay_constants.json not found; fallback to k=5")
        return _compute_news_decay_fixed(df, 5)
    with open(DECAY_CONSTANTS_PATH) as f:
        constants = json.load(f)
    df2 = df.copy()
    df2["last_news_score"] = df2.groupby("Stock")["news_score"].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0.0)
    )
    return apply_personalised_decay(df2, constants)


def _sharpe(preds, returns, confidence, threshold, tc=0.001):
    pos = np.where(
        (preds == 1) & (confidence >= threshold), 1.0,
        np.where((preds == -1) & (confidence >= threshold), -1.0, 0.0)
    )
    gross = pos * returns
    pos_prev = np.concatenate([[0], pos[:-1]])
    tc_arr   = np.where(pos != pos_prev, tc, 0.0)
    strat = gross - tc_arr
    std   = strat.std()
    return float(strat.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0


def _trades_per_week(confidence, threshold, n_weeks):
    return round(int((confidence >= threshold).sum()) / max(n_weeks, 1), 1)


def _mcnemar_p(correct_a, correct_b):
    n01 = int(((~correct_a) &  correct_b).sum())
    n10 = int((( correct_a) & ~correct_b).sum())
    if n01 + n10 == 0:
        return 1.0
    from scipy.stats import chi2 as chi2_dist
    chi2_stat = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    return float(1 - chi2_dist.cdf(chi2_stat, df=1))


def run_ablation() -> pd.DataFrame:
    if not os.path.exists(MERGED_CSV):
        print(f"[ERROR] {MERGED_CSV} not found.")
        sys.exit(1)

    merged = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    merged = merged.sort_values(["Stock", "Date"]).reset_index(drop=True)

    test = merged[merged["Date"] >= TEST_START].copy()
    test = test[test["label"].isin([-1, 1])].copy()

    if test.empty:
        print(f"[ERROR] No test data (TEST_START={TEST_START}).")
        sys.exit(1)

    print(f"\n  Test set: {len(test):,} rows | "
          f"{test['Stock'].nunique()} stocks | "
          f"{test['Date'].min().date()} to {test['Date'].max().date()}")

    y_true  = test["label"].values.astype(int)
    returns = test["Return_1d"].fillna(0).values if "Return_1d" in test.columns else np.zeros(len(test))
    n_weeks = max((test["Date"].max() - test["Date"].min()).days / 7, 1)

    payload = _load_model()
    if payload is None:
        return pd.DataFrame()

    from prediction.adaptive_gate import get_current_threshold
    adaptive_thresh, regime_score, regime_label = get_current_threshold(merged)
    fixed_thresh = WATCHLIST_MIN_CONFIDENCE

    print(f"  Adaptive threshold: {adaptive_thresh:.3f}  "
          f"(regime={regime_label}, score={regime_score:+.3f})")
    print(f"  Fixed threshold:    {fixed_thresh:.3f}")

    def _pred_config(test_df):
        proba, preds = _predict(test_df, payload)
        return proba, preds, proba.max(axis=1)

    # Config A: fixed k=5, fixed threshold
    test_a = test.copy()
    if "news_decay" in test_a.columns and "news_decay" in payload.get("feature_names", []):
        test_a["news_decay"] = _compute_news_decay_fixed(test_a, 5)
    proba_a, preds_a, conf_a = _pred_config(test_a)

    # Config B: learned k, fixed threshold
    test_b = test.copy()
    if "news_decay" in test_b.columns and "news_decay" in payload.get("feature_names", []):
        test_b["news_decay"] = _compute_news_decay_personalised(test_b)
    proba_b, preds_b, conf_b = _pred_config(test_b)

    # Config C: fixed k=5, adaptive threshold (same preds as A)
    proba_c, preds_c, conf_c = proba_a.copy(), preds_a.copy(), conf_a.copy()

    # Config D: learned k + adaptive threshold
    proba_d, preds_d, conf_d = proba_b.copy(), preds_b.copy(), conf_b.copy()

    configs = [
        ("A - Baseline",         preds_a, proba_a, conf_a, fixed_thresh),
        ("B - Learned Decay",    preds_b, proba_b, conf_b, fixed_thresh),
        ("C - Adaptive Gate",    preds_c, proba_c, conf_c, adaptive_thresh),
        ("D - Both (Proposed)",  preds_d, proba_d, conf_d, adaptive_thresh),
    ]

    rows = []
    for name, preds, proba, conf, thresh in configs:
        mask   = conf >= thresh
        n_pred = int(mask.sum())
        if n_pred == 0:
            rows.append({"Configuration": name, "Accuracy": 0, "Precision_UP": 0,
                         "Recall_UP": 0, "F1_UP": 0, "AUC": 0,
                         "Trades_per_week": 0, "Sharpe": 0,
                         "n_predictions": 0, "threshold": thresh})
            continue

        y_t, y_p = y_true[mask], preds[mask]
        acc  = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, pos_label=1,  zero_division=0))
        rec  = float(recall_score(y_t,   y_p, pos_label=1,  zero_division=0))
        f1   = float(f1_score(y_t,       y_p, pos_label=1,  zero_division=0))
        try:
            auc = float(roc_auc_score(y_t, proba[mask, 1]))
        except Exception:
            auc = float("nan")

        rows.append({
            "Configuration":   name,
            "Accuracy":        round(acc,  4),
            "Precision_UP":    round(prec, 4),
            "Recall_UP":       round(rec,  4),
            "F1_UP":           round(f1,   4),
            "AUC":             round(auc,  4),
            "Trades_per_week": _trades_per_week(conf, thresh, n_weeks),
            "Sharpe":          round(_sharpe(preds, returns, conf, thresh), 3),
            "n_predictions":   n_pred,
            "threshold":       round(thresh, 3),
        })

    results = pd.DataFrame(rows)

    correct_a = (preds_a == y_true)
    correct_d = (preds_d == y_true)
    p_val = _mcnemar_p(correct_a, correct_d)

    print("\n" + "=" * 78)
    print("  ABLATION: BASELINE vs NOVELTY CONTRIBUTIONS")
    print("=" * 78)
    print(f"  {'Config':<26} {'Acc':>7} {'Prec(UP)':>10} {'Recall':>8} "
          f"{'F1':>6} {'AUC':>7} {'Trd/wk':>8} {'Sharpe':>8}")
    print(f"  {'-'*74}")
    for _, row in results.iterrows():
        print(f"  {row['Configuration']:<26} "
              f"{row['Accuracy']*100:>6.1f}%"
              f"{row['Precision_UP']:>10.4f}"
              f"{row['Recall_UP']:>8.4f}"
              f"{row['F1_UP']:>6.4f}"
              f"{row['AUC']:>7.4f}"
              f"{row['Trades_per_week']:>8.1f}"
              f"{row['Sharpe']:>8.3f}")
    print(f"  {'-'*74}")

    if len(results) >= 2:
        acc_a = results.iloc[0]["Accuracy"]
        acc_d = results.iloc[-1]["Accuracy"]
        print(f"\n  Accuracy gain (A to D)    : {(acc_d-acc_a)*100:+.2f}pp")
        print(f"  McNemar p-value (A vs D) : {p_val:.4f} "
              f"({'significant' if p_val < 0.05 else 'not significant'} at alpha=0.05)")

    print("=" * 78 + "\n")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "ablation_results.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"  [OK] Ablation results saved to {out_path}")
    return results


if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("  NOVELTY ABLATION STUDY")
    print("  Baseline | Learned Decay | Adaptive Gate | Both")
    print("=" * 78)
    run_ablation()
