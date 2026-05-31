"""
models/ensemble/ensemble.py
============================
AUC-weighted stacking ensemble: XGBoost + LSTM.

Strategy
--------
* Compute validation-set AUC for each base model.
* Weight each model proportionally to its AUC (or give full weight to the
  dominant model when the AUC gap exceeds _AUC_GAP_THRESHOLD).
* Blend predicted probabilities: ens_prob = w_xgb * xgb_prob + w_lstm * lstm_prob.

Anti-leakage guarantees
-----------------------
* Base-model results (xgboost_results.csv, lstm_results.csv) are produced on
  chronologically held-out data — no shuffling at any stage.
* AUC weights are derived from validation predictions only, never test.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from config.settings import (
    ENSEMBLE_RESULTS_PATH,
    LSTM_RESULTS_PATH,
    XGB_RESULTS_PATH,
)

log = logging.getLogger(__name__)
CLASS_NAMES = ["DOWN", "UP"]

# Minimum AUC gap for one model to dominate the ensemble exclusively
_AUC_GAP_THRESHOLD = 0.04


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _compute_weights(xgb_auc: float, lstm_auc: float) -> tuple[float, float]:
    """
    Return (w_xgb, w_lstm).

    If one model outperforms by more than _AUC_GAP_THRESHOLD, it gets full
    weight to avoid the weaker model dragging down the ensemble.
    """
    gap_xgb  = xgb_auc  - lstm_auc
    gap_lstm = lstm_auc - xgb_auc

    if gap_xgb > _AUC_GAP_THRESHOLD:
        log.info("XGB dominates by %.4f AUC — using XGB only.", gap_xgb)
        return 1.0, 0.0
    if gap_lstm > _AUC_GAP_THRESHOLD:
        log.info("LSTM dominates by %.4f AUC — using LSTM only.", gap_lstm)
        return 0.0, 1.0

    total = xgb_auc + lstm_auc
    if total <= 1e-8:
        return 0.5, 0.5
    return xgb_auc / total, lstm_auc / total


def main() -> None:
    print("\n" + "=" * 60)
    print(" ENSEMBLE (XGBoost + LSTM) — binary DOWN/UP")
    print("=" * 60)

    for path in (XGB_RESULTS_PATH, LSTM_RESULTS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing base-model results: {path}. "
                "Run the corresponding train script first."
            )

    xgb_df  = pd.read_csv(XGB_RESULTS_PATH,  parse_dates=["Date"])
    lstm_df = pd.read_csv(LSTM_RESULTS_PATH,  parse_dates=["Date"])

    merged = pd.merge(
        xgb_df, lstm_df,
        on=["Date", "Stock", "label"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between XGBoost and LSTM results.")
    log.info("Merged base-model results: %d rows.", len(merged))

    # ── Honest AUC weighting from VAL predictions (no test-label leakage) ───
    xgb_val_path  = XGB_RESULTS_PATH.replace(".csv",  "_val.csv")
    lstm_val_path = LSTM_RESULTS_PATH.replace(".csv", "_val.csv")
    # FIX B6: do NOT fall back to test labels. Refuse to proceed if val files
    # are missing — that fallback was a silent test-set leakage of ensemble
    # weights into the final blend.
    if not (os.path.exists(xgb_val_path) and os.path.exists(lstm_val_path)):
        raise FileNotFoundError(
            f"Ensemble requires VAL prediction files for honest weighting:\n"
            f"  {xgb_val_path}\n  {lstm_val_path}\n"
            "Re-run models/xgboost/train.py and models/lstm/train.py."
        )
    xv = pd.read_csv(xgb_val_path,  parse_dates=["Date"])
    lv = pd.read_csv(lstm_val_path, parse_dates=["Date"])
    # Stale-artifact mtime guard (warn only). If the saved val predictions
    # were written long before/after the saved model, the ensemble weights
    # derived from them may not reflect the current model state.
    try:
        import os as _os_mtime
        _xgb_pkl  = "models/xgboost/saved/xgb_model.pkl"
        _xgb_vcsv = "evaluation/results/xgboost_results_val.csv"
        if _os_mtime.path.exists(_xgb_pkl) and _os_mtime.path.exists(_xgb_vcsv):
            _dt = abs(_os_mtime.path.getmtime(_xgb_pkl)
                      - _os_mtime.path.getmtime(_xgb_vcsv))
            if _dt > 3600:
                _msg = (f"[ensemble] STALE ARTIFACT WARNING: "
                        f"xgb_model.pkl and xgboost_results_val.csv "
                        f"mtimes differ by {_dt:.0f}s. Ensemble weights "
                        f"may not reflect the saved model. Re-run train_xgb.")
                try:
                    log.warning(_msg)
                except NameError:
                    print(_msg)
    except Exception:
        pass
    m_val = pd.merge(
        xv, lv, on=["Date", "Stock", "label"], how="inner",
        validate="one_to_one",
    )
    if m_val.empty:
        raise ValueError("No overlapping VAL rows between XGB and LSTM val files.")
    xgb_auc  = _safe_auc(m_val["label"].values, m_val["prob_up"].values)
    lstm_auc = _safe_auc(m_val["label"].values, m_val["lstm_prob_up"].values)
    log.info(
        "AUC weights derived from VAL (%d rows): xgb_auc=%.4f lstm_auc=%.4f",
        len(m_val), xgb_auc, lstm_auc,
    )

    w_xgb, w_lstm = _compute_weights(xgb_auc, lstm_auc)

    log.info("Final weights — XGB=%.3f (AUC=%.4f)  LSTM=%.3f (AUC=%.4f)",
             w_xgb, xgb_auc, w_lstm, lstm_auc)

    ens_prob_down = w_xgb * merged["prob_down"] + w_lstm * merged["lstm_prob_down"]
    ens_prob_up   = w_xgb * merged["prob_up"]   + w_lstm * merged["lstm_prob_up"]
    probs         = np.stack([ens_prob_down.values, ens_prob_up.values], axis=1)
    preds_ext     = np.where(probs.argmax(axis=1) == 1, 1, -1)
    y_true        = merged["label"].values

    acc = accuracy_score(y_true, preds_ext)
    auc = _safe_auc(y_true, probs[:, 1])

    print(f"\n{'='*20} Ensemble Test Set {'='*20}")
    print(f"Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(y_true, preds_ext, labels=[-1, 1],
                                target_names=CLASS_NAMES, digits=3,
                                zero_division=0))

    # FIX B9: report base-model accuracy using the SAME predictions the base
    # model itself stored (Predicted column). Re-deriving with argmax > 0.5
    # silently uses a different decision rule than the standalone scripts and
    # gives misleading "ensemble lift" numbers.
    if "Predicted" not in merged.columns or "lstm_pred" not in merged.columns:
        raise KeyError("Merged ensemble inputs must contain stored Predicted and lstm_pred columns.")
    xgb_acc = accuracy_score(y_true, merged["Predicted"])
    lstm_acc = accuracy_score(y_true, merged["lstm_pred"])

    out = merged[["Date", "Stock", "label"]].copy().reset_index(drop=True)
    out["Predicted"]  = preds_ext
    out["prob_down"]  = probs[:, 0]
    out["prob_up"]    = probs[:, 1]
    out["Confidence"] = probs.max(axis=1)
    if "Return_1d" in merged.columns:
        out["Return_1d"] = merged["Return_1d"].values

    os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
    out.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
    log.info("Ensemble results → %s", ENSEMBLE_RESULTS_PATH)

    lift = acc - max(xgb_acc, lstm_acc)
    print("\n" + "=" * 62)
    print("  ENSEMBLE — MODEL COMPARISON SUMMARY")
    print("=" * 62)
    print(f"  {'Model':<24} {'Accuracy':>10}  {'AUC':>8}  {'Weight':>8}")
    print(f"  {'-'*52}")
    print(f"  {'XGBoost (base)':<24} {xgb_acc*100:>9.2f}%  {xgb_auc:>8.4f}  {w_xgb:>7.3f}")
    print(f"  {'LSTM (base)':<24} {lstm_acc*100:>9.2f}%  {lstm_auc:>8.4f}  {w_lstm:>7.3f}")
    print(f"  {'Weighted Ensemble':<24} {acc*100:>9.2f}%  {auc:>8.4f}  {'—':>8}")
    print(f"  {'-'*52}")
    print(f"  Ensemble lift vs best base : {lift*100:>+.2f}pp")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=" [%(levelname)s] %(message)s")
    main()
