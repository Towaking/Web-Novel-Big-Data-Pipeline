"""Stacking antar label (Godbole & Sarawagi 2004) di atas SVM terbaik.

Level-1 dihitung sekali per split lalu dipakai ulang untuk beberapa meta-model.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import sparse

import exp_lib as L
import blocks_lib as B
import clf_lib as C
import stack_lib as SL
from run_batch1_specs import get_variant

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)
rep = os.environ.get("REP", "v1_maxfeat_big")
specs, W = get_variant(rep)
W.update({"corpus": 2.31, "synopsis": 0.95, "tags": 1.48, "meta": 0.48})
blocks = B.build(txt, pdf, S, specs)
Cs = float(os.environ.get("SVM_C", "0.0354"))

level1 = {}
for sname, tr, evx in S:
    xtr, xev = L.combine(blocks, W, sname)
    ztr, zev, _ = SL.level1_features(C.svm(C=Cs, cw=None), xtr, xev, y[tr])
    level1[sname] = (xtr, xev, ztr, zev)
    print("level-1 selesai:", sname, flush=True)

for meta_C, with_x, z_scale in [(1.0, False, 1.0), (0.1, False, 1.0), (4.0, True, 0.3), (4.0, True, 1.0)]:
    name = f"stack_{rep}_svmC{Cs}_metaC{meta_C}_x{int(with_x)}_z{z_scale}"
    proba_full = np.full(y.shape, np.nan, dtype=np.float32)
    proba_test = None
    for sname, tr, evx in S:
        xtr, xev, ztr, zev = level1[sname]
        if with_x:
            ftr = sparse.hstack([xtr, sparse.csr_matrix(ztr * z_scale)]).tocsr()
            fev = sparse.hstack([xev, sparse.csr_matrix(zev * z_scale)]).tocsr()
        else:
            ftr, fev = ztr * z_scale, zev * z_scale
        m = SL.meta_model(meta_C)
        m.fit(ftr, y[tr])
        p = L.proba_of(m, fev)
        if sname == "final":
            proba_test = p
        else:
            proba_full[evx] = p
    L.save_proba(name, proba_full, proba_test)
    L.score(y, dev_idx, test_idx, labels, proba_full, proba_test, name)
