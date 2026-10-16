"""Ukuran artefak RF (joblib compress=3) untuk beberapa konfigurasi, dilatih di set dev."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, io, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import joblib
import numpy as np

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from run_batch1_specs import get_variant

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)
specs, W = get_variant("v1_maxfeat_big")
blocks = B.build(txt, pdf, S, specs)
xtr, xev = L.combine(blocks, W, "final")
xtr_k, xev_k = C.Chi2TopK(3000)(xtr, xev, y[dev_idx])

for name, make in [
    ("rf_n200_leaf2_sqrt", C.rf(n=200, min_leaf=2, cw=None)),
    ("rf_n300_leaf2_log2", C.rf(n=300, min_leaf=2, cw=None, max_features="log2")),
    ("rf_n300_leaf2_sqrt_ms0.5", C.rf(n=300, min_leaf=2, cw=None, max_samples=0.5)),
    ("rf_n300_leaf3_sqrt", C.rf(n=300, min_leaf=3, cw=None)),
]:
    t0 = time.time()
    m = make()
    m.fit(xtr_k, y[dev_idx])
    fit_s = time.time() - t0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.joblib")
        joblib.dump(m, path, compress=3)
        mb = os.path.getsize(path) / 1_048_576
    nodes = sum(e.tree_.node_count for e in m.estimators_)
    print(name, "fit %.0fs" % fit_s, "joblib %.0f MB" % mb, "nodes %d" % nodes, flush=True)
