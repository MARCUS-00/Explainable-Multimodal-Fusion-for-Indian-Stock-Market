"""
models/xgboost/train.py
=======================
XGBoost binary classifier: DOWN=-1, UP=1 only.

Key guarantees
--------------
* Binary classification only.
* 10-day horizon labels (set in config/settings.py LABEL_HORIZON=10).
* Two-stage training:
        Stage 1: train on X_train only → honest val metrics.
        Stage 2: retrain on X_train+X_val → production model.
* No probability calibration is currently applied — raw XGBoost
        probabilities are written to payload["model"]. The variable name
        `cal` elsewhere in this module is a historical artifact and does
        NOT denote a CalibratedClassifierCV wrap.
* No sample-weight reweighting (FIX B5): we do not apply inverse-frequency
    weights during training because they force an artificial 50/50 prior.
"""

import logging
import os
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from config.settings import (
    MERGED_CSV, RANDOM_SEED, TEST_START, TRAIN_END, VAL_END,
    VAL_START, XGB_MODEL_PATH, XGB_RESULTS_PATH, XGBOOST_PARAMS,
    XGBOOST_FEATURES, OPTUNA_TRIALS, OPTUNA_QUICK_TRIALS, OPTUNA_BEST_PARAMS,
    PREDICTION_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format=" [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Common literals
BINARY_OBJECTIVE = "binary:logistic"

# Binary: DOWN=0 (internal), UP=1 (internal)
EXT_TO_INT  = {-1: 0, 1: 1}
INT_TO_EXT  = {0: -1, 1: 1}
CLASS_NAMES = ["DOWN", "UP"]
NUM_CLASS   = 2

PRODUCTION_MODEL_ENV = "EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_PRODUCTION_MODEL"

enc = lambda y: np.vectorize(EXT_TO_INT.get)(np.asarray(y, dtype=int))
dec = lambda y: np.vectorize(INT_TO_EXT.get)(np.asarray(y, dtype=int))


def tune_xgb(X_train, y_train_int, x_val, y_val_int,
             n_trials: int = OPTUNA_TRIALS, quick: bool = False) -> dict:
    """
    Optuna TimeSeriesSplit tuning. Returns best_params dict.
    Uses validation-AUC as the objective (no test-set contamination).
    Saves best_params.json for reproducibility.
    """
    import json
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed — skipping tuning, using defaults")
        return XGBOOST_PARAMS

    trials = OPTUNA_QUICK_TRIALS if quick else n_trials
    log.info(f"Optuna tuning: {trials} trials (TimeSeriesSplit on train+val) ...")

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score as _roc

    # Combine train + val chronologically for CV
    x_tv = pd.concat([X_train, x_val])
    y_tv = np.concatenate([y_train_int, y_val_int])
    # Use the last fold to approximate the real val split
    n_splits = 4

    def _objective(trial):
        params = {
            "objective":        BINARY_OBJECTIVE,
            "seed":             RANDOM_SEED,
            "n_jobs":           -1,
            "tree_method":      "hist",
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []
        for tr_idx, val_idx in tscv.split(x_tv):
            x_tr, x_va = x_tv.iloc[tr_idx], x_tv.iloc[val_idx]
            y_tr, y_va = y_tv[tr_idx], y_tv[val_idx]
            if len(np.unique(y_va)) < 2:
                continue
            m = XGBClassifier(**params)
            # FIX B5: do NOT apply inverse-frequency sample weights during tuning.
            m.fit(x_tr, y_tr, verbose=False)
            prob = m.predict_proba(x_va)[:, 1]
            aucs.append(_roc(y_va, prob))
        return float(np.mean(aucs)) if aucs else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(_objective, n_trials=trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": BINARY_OBJECTIVE,
        "seed": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    })
    # R1: bind cache to (train shape, feature list, seed).
    cache_key_src = f"{X_train.shape}|{sorted(X_train.columns.tolist())}|{RANDOM_SEED}"
    best["_cache_key"] = hashlib.sha256(cache_key_src.encode()).hexdigest()
    log.info(f"Best Optuna AUC={study.best_value:.4f}  params={ {k:v for k,v in best.items() if k != '_cache_key'} }")

    os.makedirs(os.path.dirname(OPTUNA_BEST_PARAMS), exist_ok=True)
    with open(OPTUNA_BEST_PARAMS, "w") as fh:
        json.dump(best, fh, indent=2)
    log.info(f"Best params saved → {OPTUNA_BEST_PARAMS}")
    return best


def load_data():
    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(f"Missing {MERGED_CSV}. Run features/build_features.py first.")
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    # Drop ambiguous rows (label==0) — binary model only
    df = df[df["label"] != 0].copy()
    log.info(f"Loaded binary dataset: {df.shape} (ambiguous rows removed)")
    return df.sort_values(["Stock", "Date"]).reset_index(drop=True)


def time_split(df):
    train = df[df["Date"] <= TRAIN_END].copy()
    val   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test  = df[df["Date"] >= TEST_START].copy()
    log.info(f"Splits — train={len(train)} val={len(val)} test={len(test)}")
    if any(len(s) == 0 for s in [train, val, test]):
        raise RuntimeError("Empty split detected. Check date config.")
    return train, val, test


def prepare_xy(df, feature_cols):
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning(f"Missing features (skipped): {missing}")
    X = df[available].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_ext = df["label"].astype(int).values
    y_int = enc(y_ext)
    return X, y_ext, y_int, available


def sample_weights(y_int):
    """
    Class-count diagnostic logger ONLY — return value is intentionally NOT
    passed to fit() anywhere in the pipeline.

    FIX B5: the previous body computed inverse-frequency weights
    (len(y) / (n_class * counts)) and passed them to XGBoost during training.
    That forced a 50/50 prior, causing the model to predict DOWN ~93% of the
    time on test (test acc 0.487 < trivial baseline 0.542). Both the Optuna
    objective and the main fit now run unweighted. We deliberately removed
    the inverse-frequency arithmetic from this function so a future refactor
    cannot reintroduce the bug by simply re-enabling the call site.
    """
    counts = np.bincount(y_int, minlength=NUM_CLASS).astype(int)
    total = int(counts.sum())
    log.info(
        f"Class skew (DIAGNOSTIC ONLY) — "
        f"DOWN={counts[0]} ({counts[0]/max(total,1)*100:.1f}%) "
        f"UP={counts[1]} ({counts[1]/max(total,1)*100:.1f}%)"
    )
    return np.ones(len(y_int), dtype=float)


def print_metrics(name, y_true_ext, y_pred_ext, y_prob):
    acc = accuracy_score(y_true_ext, y_pred_ext)
    try:
        auc = roc_auc_score(y_true_ext, y_prob[:, 1])  # binary AUC on P(UP)
    except Exception:
        auc = float("nan")
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(y_true_ext, y_pred_ext,
                                labels=[-1, 1], target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    return acc, auc


def _optuna_cache_key_for_train(X_train: pd.DataFrame) -> str:
    cache_key_src = f"{X_train.shape}|{sorted(X_train.columns.tolist())}|{RANDOM_SEED}"
    return hashlib.sha256(cache_key_src.encode()).hexdigest()


def _load_cached_optuna_params(current_key: str) -> dict | None:
    if not os.path.exists(OPTUNA_BEST_PARAMS):
        return None
    import json
    with open(OPTUNA_BEST_PARAMS) as fh:
        cached = json.load(fh)
    saved_key = cached.get("_cache_key")
    if saved_key == current_key:
        log.info(f"Loaded Optuna params from {OPTUNA_BEST_PARAMS} (cache_key match)")
        return cached
    log.warning(
        "Optuna cache stale (saved=%s, current=%s). Re-tuning.",
        str(saved_key)[:12], current_key[:12],
    )
    return None


def _strip_private_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if not k.startswith("_")}


def _load_or_tune_params(X_train, y_train_int, x_val, y_val_int, tune: bool, quick: bool) -> dict:
    params = {k: v for k, v in XGBOOST_PARAMS.items()
              if k not in ("scale_pos_weight", "eval_metric", "use_label_encoder", "num_class")}
    params.update({"objective": "binary:logistic", "seed": RANDOM_SEED})

    _env_tune = os.environ.get("XGB_TUNE", "").lower() in ("1", "true", "yes")
    if tune or _env_tune:
        current_key = _optuna_cache_key_for_train(X_train)
        cached = _load_cached_optuna_params(current_key)
        if cached is not None:
            params = cached
        else:
            n_trials = OPTUNA_QUICK_TRIALS if quick else OPTUNA_TRIALS
            log.info(f"Running Optuna: {n_trials} trials ({'quick' if quick else 'full'}) ...")
            params = tune_xgb(X_train, y_train_int, x_val, y_val_int, quick=quick)
    else:
        log.info("Optuna tuning skipped — using config defaults (pass tune=True to enable)")

    return _strip_private_params(params)


def _fit_and_score_models(X_train, y_train_int, x_val, y_val_ext, y_val_int, x_test, y_test_ext, params,
                          feats, train_medians):
    log.info("Stage 1: training on TRAIN only for honest val diagnostics ...")
    _ = sample_weights(y_train_int)
    m_diag = XGBClassifier(**params)
    m_diag.fit(X_train, y_train_int, verbose=False)

    prob_train_diag = m_diag.predict_proba(X_train)
    y_train_ext = dec(y_train_int)
    train_acc = accuracy_score(y_train_ext, dec(prob_train_diag.argmax(1)))
    try:
        train_auc = roc_auc_score(y_train_ext, prob_train_diag[:, 1])
    except Exception:
        train_auc = float("nan")

    prob_val_diag = m_diag.predict_proba(x_val)
    val_acc, val_auc = print_metrics("VAL (2024, held-out)", y_val_ext,
                                     dec(prob_val_diag.argmax(1)), prob_val_diag)

    log.info("Stage 2: retraining base for SHAP on TRAIN+VAL ...")
    x_tv = pd.concat([X_train, x_val])
    y_tv = np.concatenate([y_train_int, y_val_int])
    base = XGBClassifier(**params)
    base.fit(x_tv, y_tv, verbose=False)

    _prod = os.environ.get(PRODUCTION_MODEL_ENV, "train_plus_val").lower()
    if _prod == "train_only":
        log.info("Production model = Stage 1 (train-only)")
        cal = m_diag
    elif _prod == "train_plus_val":
        log.info("Production model = Stage 2 (train+val); matches docstring")
        cal = base
    else:
        raise ValueError(
            f"{PRODUCTION_MODEL_ENV}={_prod!r} not in "
            "{'train_only', 'train_plus_val'}"
        )

    from sklearn.metrics import f1_score as _f1
    # Always use train-only model for threshold tuning — val is in cal's train set
    # when the production model is train_plus_val.
    prob_val_cal = m_diag.predict_proba(x_val)[:, 1]
    diag_t, diag_score = PREDICTION_THRESHOLD, -1.0
    for _t in np.arange(0.30, 0.71, 0.01):
        _y_pred = np.where(prob_val_cal >= _t, 1, 0)
        _score = _f1(y_val_int, _y_pred, average="macro")
        if _score > diag_score:
            diag_score = _score
            diag_t = float(_t)
    log.info("Val-tuned threshold candidate: %.2f (val macro-F1=%.4f). "
             "Keeping default %.2f because val→test class shift makes the "
             "tuned threshold overshoot.", diag_t, diag_score, PREDICTION_THRESHOLD)
    chosen_threshold = float(PREDICTION_THRESHOLD)

    prob_test = cal.predict_proba(x_test)
    pred_test = dec(np.where(prob_test[:, 1] >= chosen_threshold, 1, 0))
    test_acc, test_auc = print_metrics("TEST (2025)", y_test_ext, pred_test, prob_test)

    gap = abs(val_acc - test_acc)
    log.info(f"Val→Test gap={gap*100:.1f}pp {'— watch for overfit' if gap > 0.10 else '— OK'}")

    try:
        fi = sorted(zip(feats, base.feature_importances_), key=lambda x: -x[1])[:15]
        print("\nTop-15 Features:")
        for f, v in fi:
            print(f"  {f:<40s} {v:.4f}")
    except Exception:
        pass

    payload = {
        "model":              cal,
        "base_model":         base,
        "feature_names":      feats,
        "train_medians":      {c: float(train_medians.get(c, 0.0)) for c in feats},
        "ext_to_int":         EXT_TO_INT,
        "int_to_ext":         INT_TO_EXT,
        "num_class":          NUM_CLASS,
        "proba_cols":         ["prob_down", "prob_up"],
        "prediction_threshold": float(chosen_threshold),
        "dropped_constants":  [],
    }
    try:
        payload["date_trained"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        payload["trained_at_utc"] = pd.Timestamp.utcnow().isoformat() + "Z"
    except Exception:
        payload["date_trained"] = None
        payload["trained_at_utc"] = None

    return payload, prob_test, pred_test, train_acc, train_auc, val_acc, val_auc, test_acc, test_auc, gap, chosen_threshold, m_diag


def main(tune: bool = False, quick: bool = False):
    """
    Train XGBoost.

    Parameters
    ----------
    tune  : run Optuna hyperparameter search (default False).
            Skipped if best_params.json already exists; delete the file to force rerun.
    quick : use OPTUNA_QUICK_TRIALS instead of OPTUNA_TRIALS (faster, less thorough).
    """
    print("\n" + "="*60)
    print(" XGBOOST TRAIN  (DOWN / UP binary, 10-day horizon)")
    print("="*60)

    df = load_data()
    train_df, val_df, test_df = time_split(df)

    X_train, _, y_train_int, feats = prepare_xy(train_df, XGBOOST_FEATURES)
    x_val,   y_val_ext, y_val_int, _ = prepare_xy(val_df, feats)
    x_test,  y_test_ext, _,       _ = prepare_xy(test_df, feats)

    train_medians = X_train.median().to_dict()
    for X in [X_train, x_val, x_test]:
        X.fillna(pd.Series(train_medians), inplace=True)

    const = X_train.columns[X_train.nunique() <= 1].tolist()
    if const:
        log.warning(
            "R5: Dropping %d constant column(s) on train: %s. "
            "Recorded in payload as 'dropped_constants'.",
            len(const), const,
        )
        for X in [X_train, x_val, x_test]:
            X.drop(columns=const, inplace=True)
        feats = [f for f in feats if f not in const]
    _dropped_constants = list(const)

    log.info(f"Class counts — DOWN={int((y_train_int==0).sum())} UP={int((y_train_int==1).sum())}")
    params = _load_or_tune_params(X_train, y_train_int, x_val, y_val_int, tune, quick)

    payload, prob_test, pred_test, train_acc, train_auc, val_acc, val_auc, test_acc, test_auc, gap, chosen_threshold, m_diag = _fit_and_score_models(
        X_train, y_train_int, x_val, y_val_ext, y_val_int, x_test, y_test_ext, params,
        feats, train_medians,
    )

    payload["dropped_constants"] = _dropped_constants
    from models.artifact_io import safe_dump
    safe_dump(payload, XGB_MODEL_PATH)
    log.info(f"Model saved → {XGB_MODEL_PATH}")

    res = test_df[["Date", "Stock", "label"]].copy().reset_index(drop=True)
    res["Predicted"]  = pred_test
    res["prob_down"]  = prob_test[:, 0]
    res["prob_up"]    = prob_test[:, 1]
    res["Confidence"] = prob_test.max(axis=1)
    if "Return_1d" in test_df.columns:
        res["Return_1d"] = test_df["Return_1d"].values
    os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
    res.to_csv(XGB_RESULTS_PATH, index=False)
    log.info(f"Results → {XGB_RESULTS_PATH}")

    # Use the train-only model (m_diag) so xgboost_results_val.csv stays
    # honestly out-of-sample. payload["model"] may be the train+val refit,
    # which previously inflated val AUC to ~0.94 and broke ensemble weighting.
    prob_val_for_save = m_diag.predict_proba(x_val)
    val_res = val_df[["Date", "Stock", "label"]].copy().reset_index(drop=True)
    val_res["Predicted"]  = dec(np.where(prob_val_for_save[:, 1] >= chosen_threshold, 1, 0))
    val_res["prob_down"]  = prob_val_for_save[:, 0]
    val_res["prob_up"]    = prob_val_for_save[:, 1]
    val_res["Confidence"] = prob_val_for_save.max(axis=1)
    val_path = XGB_RESULTS_PATH.replace(".csv", "_val.csv")
    val_res = val_res[["Date", "Stock", "label", "Predicted", "prob_down", "prob_up", "Confidence"]]
    val_res.to_csv(val_path, index=False)
    log.info(f"VAL predictions → {val_path}")

    gap_flag     = "⚠ GAP"    if gap > 0.10 else "OK"

    print("\n" + "="*62)
    print("  XGBOOST — TRAIN / VAL / TEST ACCURACY SUMMARY")
    print("="*62)
    print(f"  {'Split':<22} {'Accuracy':>10}  {'AUC':>8}  {'Notes'}")
    print(f"  {'-'*58}")
    print(f"  {'Train (in-sample)':<22} {train_acc*100:>9.2f}%  {train_auc:>8.4f}  [in-sample, expected high]")
    print(f"  {'Val   (2024, honest)':<22} {val_acc*100:>9.2f}%  {val_auc:>8.4f}  [honest held-out]")
    print(f"  {'Test  (2025, calib.)':<22} {test_acc*100:>9.2f}%  {test_auc:>8.4f}  [{gap_flag}]")
    print(f"  {'-'*58}")
    print(f"  Train→Val  gap : {(train_acc-val_acc)*100:>5.1f}pp  (train is in-sample; large gap expected)")
    print(f"  Val→Test   horizon gap : {gap*100:>5.1f}pp  ({gap_flag})")
    print("="*62 + "\n")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Train XGBoost stock direction model")
    _p.add_argument("--tune",  action="store_true", default=False,
                    help="Run Optuna hyperparameter search")
    _p.add_argument("--quick", action="store_true", default=False,
                    help="Use OPTUNA_QUICK_TRIALS instead of OPTUNA_TRIALS")
    _a = _p.parse_args()
    main(tune=_a.tune, quick=_a.quick)
