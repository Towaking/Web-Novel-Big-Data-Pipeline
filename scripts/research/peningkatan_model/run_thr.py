import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import exp_lib as L
import thr_lib as T

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
fold_ids = np.zeros(len(dev_idx), dtype=int)
for k, (_, va) in enumerate(folds):
    fold_ids[va] = k

name = sys.argv[1] if len(sys.argv) > 1 else "b1_v0_base"
oof, pt = L.load_proba(name)
y_dev, p_dev, y_te = y[dev_idx], oof[dev_idx], y[test_idx]

tuners = {
    "grid_0.05-0.80_0.025": lambda a, b: T.tune_grid(a, b),
    "grid_0.01-0.95_0.01": lambda a, b: T.tune_grid(a, b, low=0.01, high=0.95, step=0.01),
    "grid_0.02-0.90_0.005": lambda a, b: T.tune_grid(a, b, low=0.02, high=0.90, step=0.005),
    "exact_0.01-0.95": lambda a, b: T.tune_exact(a, b),
}
for tname, tuner in tuners.items():
    for alo in (False, True):
        cvs = T.cv_threshold_score(y_dev, p_dev, fold_ids, tuner, alo)
        thr = tuner(y_dev, p_dev)
        te = T.metrics(y_te, pt, thr, alo)
        print(json.dumps({"proba": name, "tuner": tname, "at_least_one": alo,
                          "cvthr_macro": cvs["macro"], "cvthr_micro": cvs["micro"], "cvthr_samples": cvs["samples"],
                          "test_macro": te["macro"], "test_micro": te["micro"], "test_samples": te["samples"]}))
