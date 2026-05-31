import json
from models.xgboost import train_walkforward as wf
_orig = wf._build_params
def _tuned():
    p = _orig()
    with open('models/xgboost/saved/best_params.json') as fh:
        b = json.load(fh)
    p.update({k: v for k, v in b.items() if not k.startswith('_')})
    return p
wf._build_params = _tuned
wf.walkforward(step=30, threshold=0.5,
               out='evaluation/results/xgboost_walkforward_tuned.csv')
