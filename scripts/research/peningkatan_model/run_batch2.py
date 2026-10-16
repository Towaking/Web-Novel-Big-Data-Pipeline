"""Batch 2: classifier di atas representasi terpilih (argumen: nama varian spesifikasi)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from run_batch1_specs import get_variant

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)

rep = os.environ.get("REP", "v1_maxfeat_big")
specs, W = get_variant(rep)
blocks = B.build(txt, pdf, S, specs)

EXPS = {
    # SVM: C dan class_weight
    **{f"svm_C{c}_bal": (C.svm(C=c), None) for c in (0.1, 0.2, 0.5, 1.0)},
    **{f"svm_C{c}_none": (C.svm(C=c, cw=None), None) for c in (0.3, 1.0)},
    "svm_C0.3_bal_iso": (C.svm(C=0.3, method="isotonic"), None),
    "svm_C0.3_bal_ens0": (C.svm(C=0.3, ensemble=False), None),
    # LogReg
    **{f"lr_C{c}_none": (C.lr(C=c), None) for c in (4.0, 16.0, 64.0)},
    **{f"lr_C{c}_bal": (C.lr(C=c, cw="balanced"), None) for c in (4.0, 16.0)},
    # NB-weighted (Wang & Manning 2012)
    "nbsvm_C0.3_bal": (C.nb_svm(C=0.3), None),
    "nbsvm_C0.1_bal": (C.nb_svm(C=0.1), None),
    "nblr_C4_none": (C.nb_lr(C=4.0), None),
    "nblr_C16_none": (C.nb_lr(C=16.0), None),
    # Naive Bayes (Rennie dkk. 2003) - pembanding cepat
    "cnb_a0.3": (C.cnb(0.3), None),
    "mnb_a0.03": (C.mnb(0.03), None),
    # SGD modified huber
    "sgd_mh_1e-5": (C.sgd_mh(1e-5), None),
}

only = sys.argv[1:] or list(EXPS)
for name in only:
    make, tf = EXPS[name]
    pf, pt, dt = L.oof_and_test(make, blocks, W, y, S, transform=tf)
    L.save_proba(f"b2_{rep}_{name}", pf, pt)
    L.score(y, dev_idx, test_idx, labels, pf, pt, f"b2_{rep}_{name}", {"secs": round(dt, 1)})
