"""Random Forest multi-output (satu hutan untuk 35 genre, gaya RF-PCT)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, pickle, io
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from run_batch1_specs import get_variant

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)
rep = os.environ.get("REP", "v1_maxfeat_big")
specs, W = get_variant(rep)
if os.environ.get("TUNED_W"):
    W.update({"corpus": 2.31, "synopsis": 0.95, "tags": 1.48, "meta": 0.48})
    rep = rep + "_tw"
blocks = B.build(txt, pdf, S, specs)


class SVDT:
    def __init__(self, k):
        self.k = k

    def __call__(self, xtr, xev, ytr):
        svd = TruncatedSVD(self.k, random_state=42)
        a = svd.fit_transform(xtr)
        return a.astype(np.float32), svd.transform(xev).astype(np.float32)


def model_size_mb(model):
    buf = io.BytesIO()
    pickle.dump(model, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return round(buf.tell() / 1_048_576, 1)


EXPS = {
    "rf_chi2k3000_n200_leaf2_bal": (C.rf(n=200, min_leaf=2), C.Chi2TopK(3000)),
    "rf_chi2k10000_n200_leaf2_bal": (C.rf(n=200, min_leaf=2), C.Chi2TopK(10000)),
    "rf_chi2k3000_n200_leaf2_none": (C.rf(n=200, min_leaf=2, cw=None), C.Chi2TopK(3000)),
    "rf_chi2k3000_n200_leaf5_bal": (C.rf(n=200, min_leaf=5), C.Chi2TopK(3000)),
    "rf_chi2k3000_n200_mf0.1_leaf2_bal": (C.rf(n=200, min_leaf=2, max_features=0.1), C.Chi2TopK(3000)),
    "et_chi2k3000_n200_leaf2_bal": (C.rf(n=200, min_leaf=2, extra=True), C.Chi2TopK(3000)),
    "rf_svd300_n200_leaf2_bal": (C.rf(n=200, min_leaf=2), SVDT(300)),
    "rf_chi2k1500_n200_leaf2_none": (C.rf(n=200, min_leaf=2, cw=None), C.Chi2TopK(1500)),
    "rf_chi2k10000_n200_leaf2_none": (C.rf(n=200, min_leaf=2, cw=None), C.Chi2TopK(10000)),
    "rf_chi2k3000_n200_leaf1_none": (C.rf(n=200, min_leaf=1, cw=None), C.Chi2TopK(3000)),
    "rf_chi2k3000_n200_leaf5_none": (C.rf(n=200, min_leaf=5, cw=None), C.Chi2TopK(3000)),
    "rf_chi2k3000_n200_mf0.05_leaf2_none": (C.rf(n=200, min_leaf=2, cw=None, max_features=0.05), C.Chi2TopK(3000)),
    "et_chi2k3000_n200_leaf2_none": (C.rf(n=200, min_leaf=2, cw=None, extra=True), C.Chi2TopK(3000)),
    "rf_chi2k3000_n500_leaf2_none": (C.rf(n=500, min_leaf=2, cw=None), C.Chi2TopK(3000)),
    "rf_chi2k3000_n300_mln1000_none": (C.rf(n=300, min_leaf=1, cw=None, max_leaf_nodes=1000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n300_mln2000_none": (C.rf(n=300, min_leaf=1, cw=None, max_leaf_nodes=2000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n300_mln4000_none": (C.rf(n=300, min_leaf=1, cw=None, max_leaf_nodes=4000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n100_mln4000_none": (C.rf(n=100, min_leaf=1, cw=None, max_leaf_nodes=4000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n150_mln3000_none": (C.rf(n=150, min_leaf=1, cw=None, max_leaf_nodes=3000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n200_mln2000_none": (C.rf(n=200, min_leaf=1, cw=None, max_leaf_nodes=2000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n150_mln3000_none": (C.rf(n=150, min_leaf=1, cw=None, max_leaf_nodes=3000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n50_mln4000_none": (C.rf(n=50, min_leaf=1, cw=None, max_leaf_nodes=4000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n100_mln1000_none": (C.rf(n=100, min_leaf=1, cw=None, max_leaf_nodes=1000), C.Chi2TopK(3000)),
    "rf_chi2k3000_n60_mln2000_none": (C.rf(n=60, min_leaf=1, cw=None, max_leaf_nodes=2000), C.Chi2TopK(3000)),
}

only = sys.argv[1:] or list(EXPS)
for name in only:
    make, tf = EXPS[name]
    t0 = time.time()
    pf, pt, dt = L.oof_and_test(make, blocks, W, y, S, transform=tf)
    L.save_proba(f"rf_{rep}_{name}", pf, pt)
    L.score(y, dev_idx, test_idx, labels, pf, pt, f"rf_{rep}_{name}", {"secs": round(dt, 1)})
