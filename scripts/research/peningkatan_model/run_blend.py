"""Ensemble rata-rata probabilitas berbobot; bobot dipilih di OOF, uji sekali.

Pemakaian: python run_blend.py nama_proba_1 nama_proba_2 [nama_proba_3 ...]
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import exp_lib as L
import thr_lib as T
from Model.pipeline import evaluate as ev

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
fold_ids = np.zeros(len(dev_idx), dtype=int)
for k, (_, va) in enumerate(folds):
    fold_ids[va] = k
names = sys.argv[1:]
P = [L.load_proba(n) for n in names]
y_dev, y_te = y[dev_idx], y[test_idx]

step = 0.1 if len(names) <= 3 else 0.2
grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
best = None
for w in grid:
    p = sum(wi * pi[0][dev_idx] for wi, pi in zip(w, P))
    thr = T.tune_grid(y_dev, p)
    m = ev.evaluate(y_dev, p, thr)["f1_macro"]
    if best is None or m > best[0]:
        best = (m, w)
m, w = best
p_dev = sum(wi * pi[0][dev_idx] for wi, pi in zip(w, P))
p_te = sum(wi * pi[1] for wi, pi in zip(w, P))
thr = T.tune_grid(y_dev, p_dev)
te = ev.evaluate(y_te, p_te, thr)
cvt = T.cv_threshold_score(y_dev, p_dev, fold_ids, lambda a, b: T.tune_grid(a, b))
print(json.dumps({"blend": dict(zip(names, [round(x, 2) for x in w])), "oof_macro": round(m, 4),
                  "cvthr_macro": cvt["macro"], "test_macro": te["f1_macro"], "test_micro": te["f1_micro"],
                  "test_p3": te["precision_at_3"], "test_samples": te["f1_samples"]}))
