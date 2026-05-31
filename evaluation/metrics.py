"""
evaluation/metrics.py
=====================
Evaluation metrics for the binary XGBoost model (DOWN / UP).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

CLASS_NAMES = ["DOWN", "UP"]
EXT_TO_INT  = {-1: 0, 1: 1}
NUM_CLASS   = 2


def _enc(y):
    return np.vectorize(EXT_TO_INT.get)(np.asarray(y, dtype=int))


def evaluate(y_true, y_pred, y_proba, model_name="Model"):
    """
    Parameters
    ----------
    y_true  : ground truth in {-1, 1}.
    y_pred  : predicted labels in {-1, 1}.
    y_proba : (N, 2) array [P(DOWN), P(UP)].
    """
    y_true_i = _enc(y_true)
    y_pred_i = _enc(y_pred)

    acc = accuracy_score(y_true_i, y_pred_i)
    try:
        auc = roc_auc_score(y_true_i, y_proba[:, 1])
    except Exception:
        auc = float("nan")

    report = classification_report(
        y_true_i, y_pred_i, labels=[0, 1],
        target_names=CLASS_NAMES, digits=3,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1])

    print(f"\n{'='*55}")
    print(f"  {model_name} EVALUATION  (n={len(y_true)})")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print()
    print(classification_report(y_true_i, y_pred_i, labels=[0, 1],
                                target_names=CLASS_NAMES, digits=3, zero_division=0))
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':10s}" + "".join(f"{c:>8s}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:10s}" + "".join(f"{v:8d}" for v in row))

    print("\n  Calibration check:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true_i == i
        if mask.sum() > 0:
            mp = y_proba[mask, i].mean()
            br = mask.mean()
            print(f"    {name}: base_rate={br:.3f}  mean_prob_when_true={mp:.3f}")
    print(f"{'='*55}\n")

    return {"accuracy": acc, "auc": auc, "report": report, "confusion_matrix": cm.tolist()}


def print_class_distribution(df, label_col="label"):
    labels = df[label_col].astype(int)
    n = len(labels)
    print("\n  Label distribution:")
    for ext_lbl, name in [(-1, "DOWN"), (1, "UP")]:
        cnt = (labels == ext_lbl).sum()
        print(f"    {name:5s}: {cnt:6d}  ({cnt/n*100:.1f}%)")
    print()


def evaluate_by_sector(results_df: pd.DataFrame, sector_map: dict):
    """
    Compute accuracy and AUC per sector and print a ranked table.

    results_df: DataFrame with columns ['Stock', 'label', 'Predicted', 'prob_up']
    sector_map: dict mapping stock -> sector name
    """
    df = results_df.copy()
    df = df.loc[df['Stock'].isin(sector_map.keys())].copy()
    if df.empty:
        print("No rows for sector evaluation")
        return {}
    df['Sector'] = df['Stock'].map(sector_map)
    stats = []
    for sec, g in df.groupby('Sector'):
        y_true = g['label'].astype(int).values
        y_pred = g['Predicted'].astype(int).values if 'Predicted' in g.columns else (g['prob_up'] > 0.5).astype(int).values
        try:
            auc = roc_auc_score(y_true, g['prob_up'].values)
        except Exception:
            auc = float('nan')
        acc = accuracy_score(y_true, y_pred)
        stats.append((sec, len(g), acc, auc))
    stats_df = pd.DataFrame(stats, columns=['Sector', 'N', 'Accuracy', 'AUC']).sort_values('Accuracy', ascending=False)
    print('\n' + '='*60)
    print('  Per-Sector Evaluation (ranked by Accuracy)')
    print('='*60)
    for _, r in stats_df.iterrows():
        print(f"  {r['Sector']:<25s} N={int(r['N']):4d}  Acc={r['Accuracy']:.3f}  AUC={r['AUC']:.3f}")
    print('='*60 + '\n')
    return stats_df.set_index('Sector').to_dict(orient='index')


if __name__ == "__main__":
    from config.settings import MERGED_CSV, TEST_START
    from models.xgboost.predict import predict_label, load_xgb

    payload = load_xgb()
    df      = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    test_df = df[(df["Date"] >= TEST_START) & (df["label"] != 0)].copy()

    print_class_distribution(test_df)
    labels, proba = predict_label(test_df, payload)
    evaluate(y_true=test_df["label"].values, y_pred=labels,
             y_proba=proba, model_name="XGBoost Binary (Test Set)")
