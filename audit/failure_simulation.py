"""
audit/failure_simulation.py
============================
Simulates 5 known failure modes and confirms the audit/metrics detect them.

Each simulation:
  1. Deliberately injects the bug into a copy of the data / model.
  2. Trains / evaluates.
  3. Reports expected symptom and whether it was detected.

Simulations
-----------
1. LOOKAHEAD BUG      — remove shift on return features → inflated AUC
2. DATA LEAKAGE       — inject future feature (next-day return) → ~1.0 AUC
3. OVERFITTING        — train on full data (no holdout) → vs walk-forward
4. FEATURE BREAK      — remove top SHAP feature → observe AUC drop
5. RANDOM LABELS      — shuffle labels → AUC collapses to ~0.5
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config.settings import MERGED_CSV, TEST_START, TRAIN_END, VAL_END, VAL_START, XGBOOST_FEATURES

logging.basicConfig(level=logging.WARNING)

EXT_TO_INT = {-1: 0, 1: 1}
NUM_CLASS  = 2
BASE_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "eval_metric": "logloss", "use_label_encoder": False,
    "tree_method": "hist", "seed": 42, "n_jobs": -1,
}
FEATURE_COLS = XGBOOST_FEATURES


def _load():
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    return df.sort_values(["Stock", "Date"]).reset_index(drop=True)


def _split(df):
    train = df[df["Date"] <= TRAIN_END]
    val   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)]
    test  = df[df["Date"] >= TEST_START]
    return train, val, test


def _avail(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def _enc(y): return np.vectorize(EXT_TO_INT.get)(np.asarray(y, dtype=int))


def _fit_eval(x_tr, y_tr, x_te, y_te, tag=""):
    mdl = XGBClassifier(**BASE_PARAMS)
    mdl.fit(x_tr, _enc(y_tr), verbose=False)
    prob = mdl.predict_proba(x_te)
    try:
        auc = roc_auc_score(_enc(y_te), prob[:, 1])
    except Exception:
        auc = float("nan")
    print(f"    AUC ({tag}): {auc:.4f}")
    return auc


def _header(n, title):
    print(f"\n{'='*60}")
    print(f"  SIM {n}: {title}")
    print(f"{'='*60}")


def sim1_lookahead(df):
    """Remove shift → features use current-day return (lookahead)."""
    _header(1, "LOOKAHEAD BUG — remove shift on returns")
    print("  Injecting: ret_1d used directly (no shift)")
    print("  Expected : inflated AUC (~0.7–0.9 vs ~0.55 baseline)")
    print("  Detection: audit scans for pct_change() without .shift()")

    train, _, test = _split(df)
    feats = _avail(df)

    # Baseline (correct, lagged)
    auc_clean = _fit_eval(
        train[feats].fillna(0), train["label"],
        test[feats].fillna(0),  test["label"],
        tag="CLEAN (lagged)"
    )

    # Buggy: replace ret_lag_1d with unshifted Return_1d
    df_bug = df.copy()
    if "Return_1d" in df_bug.columns:
        df_bug["ret_lag_1d"] = df_bug["Return_1d"]   # no shift!
    train_b, _, test_b = _split(df_bug)
    auc_bug = _fit_eval(
        train_b[feats].fillna(0), train_b["label"],
        test_b[feats].fillna(0),  test_b["label"],
        tag="BUGGY (no shift)"
    )

    detected = auc_bug > auc_clean + 0.05
    print(f"  Symptom  : AUC rose by {(auc_bug - auc_clean):.4f}")
    print(f"  Detected : {'YES ✅' if detected else 'NO ❌'}")
    return {"sim": 1, "auc_clean": auc_clean, "auc_bug": auc_bug, "detected": detected}


def sim2_data_leakage(df):
    """Inject next-day return as a feature → model gets free future signal."""
    _header(2, "DATA LEAKAGE — inject future feature")
    print("  Injecting: next_day_return = Return_1d.shift(-1) as feature")
    print("  Expected : AUC jumps toward 1.0")
    print("  Detection: audit flags any .shift(-N) outside label block")

    feats = _avail(df)
    train, _, test = _split(df)
    auc_clean = _fit_eval(
        train[feats].fillna(0), train["label"],
        test[feats].fillna(0),  test["label"],
        tag="CLEAN"
    )

    df_leak = df.copy()
    df_leak["LEAKED_next_ret"] = df_leak.groupby("Stock")["Return_1d"].shift(-1)
    feats_leak = feats + ["LEAKED_next_ret"]
    train_l, _, test_l = _split(df_leak)
    auc_leak = _fit_eval(
        train_l[feats_leak].fillna(0), train_l["label"],
        test_l[feats_leak].fillna(0),  test_l["label"],
        tag="LEAKED"
    )

    detected = auc_leak > auc_clean + 0.10
    print(f"  Symptom  : AUC rose by {(auc_leak - auc_clean):.4f}")
    print(f"  Detected : {'YES ✅' if detected else 'NO ❌'}")
    return {"sim": 2, "auc_clean": auc_clean, "auc_leak": auc_leak, "detected": detected}


def sim3_overfitting(df):
    """Train on full dataset (no holdout) vs walk-forward."""
    _header(3, "OVERFITTING — full-data train vs walk-forward")
    print("  Injecting: train on all data including test period")
    print("  Expected : test AUC inflated vs proper holdout")
    print("  Detection: walk-forward AUC < full-data AUC by significant margin")

    feats = _avail(df)
    train, _, test = _split(df)

    # Proper: train on train only
    auc_proper = _fit_eval(
        train[feats].fillna(0), train["label"],
        test[feats].fillna(0),  test["label"],
        tag="PROPER (train split)"
    )

    # Overfit: train on full df including test
    auc_overfit = _fit_eval(
        df[feats].fillna(0), df["label"],
        test[feats].fillna(0),  test["label"],
        tag="OVERFIT (full data)"
    )

    detected = auc_overfit > auc_proper + 0.03
    print(f"  Symptom  : AUC gap = {(auc_overfit - auc_proper):.4f}")
    print(f"  Detected : {'YES ✅' if detected else 'NO ❌'}")
    return {"sim": 3, "auc_proper": auc_proper, "auc_overfit": auc_overfit, "detected": detected}


def sim4_feature_break(df):
    """Remove the highest-importance feature and observe performance drop."""
    _header(4, "FEATURE BREAK — remove top feature")
    print("  Injecting: drop ret_lag_1d (typically top feature)")
    print("  Expected : AUC drops noticeably")
    print("  Detection: AUC degradation confirms feature signal is real")

    feats = _avail(df)
    train, _, test = _split(df)

    auc_full = _fit_eval(
        train[feats].fillna(0), train["label"],
        test[feats].fillna(0),  test["label"],
        tag="FULL features"
    )

    top_feat = "ret_lag_1d"
    feats_reduced = [f for f in feats if f != top_feat]
    auc_reduced = _fit_eval(
        train[feats_reduced].fillna(0), train["label"],
        test[feats_reduced].fillna(0),  test["label"],
        tag=f"DROP {top_feat}"
    )

    detected = auc_full > auc_reduced + 0.005
    print(f"  Symptom  : AUC dropped by {(auc_full - auc_reduced):.4f}")
    print(f"  Detected : {'YES ✅' if detected else 'NO ❌'}")
    return {"sim": 4, "auc_full": auc_full, "auc_reduced": auc_reduced, "detected": detected}


def sim5_random_labels(df):
    """Shuffle labels → model should not beat random (AUC ~0.5)."""
    _header(5, "RANDOM MODEL — shuffle labels")
    print("  Injecting: labels randomly shuffled")
    print("  Expected : AUC collapses to ~0.5 (binary random)")
    print("  Detection: AUC < 0.55 confirms model can't learn from noise")

    feats = _avail(df)
    train, _, test = _split(df)

    auc_real = _fit_eval(
        train[feats].fillna(0), train["label"],
        test[feats].fillna(0),  test["label"],
        tag="REAL labels"
    )

    train_shuf = train.copy()
    train_shuf["label"] = np.random.default_rng(99).permutation(train_shuf["label"].values)
    auc_shuf = _fit_eval(
        train_shuf[feats].fillna(0), train_shuf["label"],
        test[feats].fillna(0),        test["label"],
        tag="SHUFFLED labels"
    )

    detected = auc_real > auc_shuf + 0.05
    print(f"  Symptom  : AUC dropped from {auc_real:.4f} to {auc_shuf:.4f}")
    print(f"  Detected : {'YES ✅' if detected else 'NO ❌'}")
    return {"sim": 5, "auc_real": auc_real, "auc_shuffled": auc_shuf, "detected": detected}


def run_all():
    print("\n" + "=" * 60)
    print("  FAILURE CASE SIMULATION  (5 scenarios)")
    print("=" * 60)

    if not os.path.exists(MERGED_CSV):
        print(f"[ERROR] {MERGED_CSV} not found. Run build_features.py first.")
        return []

    df = _load()
    results = []
    for fn in [sim1_lookahead, sim2_data_leakage, sim3_overfitting,
               sim4_feature_break, sim5_random_labels]:
        try:
            results.append(fn(df))
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {e}")

    n_detected = sum(1 for r in results if r.get("detected"))
    print(f"\n{'='*60}")
    print(f"  SIMULATION SUMMARY: {n_detected}/{len(results)} failures detected")
    for r in results:
        status = "✅" if r.get("detected") else "❌"
        print(f"  {status} SIM {r['sim']}")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    run_all()
