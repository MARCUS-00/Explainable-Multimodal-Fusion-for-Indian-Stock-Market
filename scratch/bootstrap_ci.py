import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score, accuracy_score
rng = np.random.default_rng(42)
r = pd.read_csv('evaluation/results/xgboost_results.csv')
y = (r['label'] == 1).astype(int).values
p = r['prob_up'].values
pred = r['Predicted'].astype(int).values
label = r['label'].astype(int).values
n = len(y); B = 1000
aucs, accs = [], []
for _ in range(B):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2: continue
    aucs.append(roc_auc_score(y[idx], p[idx]))
    accs.append(accuracy_score(label[idx], pred[idx]))
aucs, accs = np.array(aucs), np.array(accs)
out = {
  'test_auc_point':  float(roc_auc_score(y, p)),
  'test_auc_ci95':   [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
  'test_acc_point':  float(accuracy_score(label, pred)),
  'test_acc_ci95':   [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
  'n_test':          int(n),
  'up_rate':         float(y.mean()),
  'n_bootstrap':     int(B),
}
import os; os.makedirs('scratch', exist_ok=True)
json.dump(out, open('scratch/bootstrap_ci.json', 'w'), indent=2)
print(out)
