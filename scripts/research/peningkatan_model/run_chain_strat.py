"""Dua metode yang belum diuji: classifier chains dan stratifikasi iteratif untuk fold OOF.

- Ensemble of classifier chains (Read dkk. 2011): 3 rantai urutan acak, basis Logistic Regression
  (C=0,25, fitur final), dibanding LogReg binary relevance pada fold yang sama.
- Stratifikasi iteratif (Sechidis dkk. 2011, implementasi scikit-multilearn): fold OOF untuk
  penyetelan ambang SVM final, dibanding KFold acak. Set uji tetap sama.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.svm import LinearSVC

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from Model.pipeline import evaluate as ev
from run_batch1_specs import get_variant

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)
specs, W = get_variant("v12_75k")
W.update({"corpus": 2.31, "synopsis": 0.95, "tags": 1.48, "meta": 0.48})
blocks = B.build(txt, pdf, S, specs)
NJ = int(os.environ.get("N_JOBS", "-1"))
OUT = os.path.join(os.path.dirname(__file__), "results", "chain_strat_results.jsonl")


def log(rec):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


class ChainEnsemble:
    """Rata-rata probabilitas beberapa ClassifierChain berurutan acak (ECC)."""

    def __init__(self, n_chains=3, C=0.25):
        self.n_chains, self.C = n_chains, C

    def fit(self, X, Y):
        def one(seed):
            return ClassifierChain(LogisticRegression(C=self.C, max_iter=5000), order="random",
                                   random_state=seed).fit(X, Y)
        self.chains_ = Parallel(n_jobs=min(self.n_chains, 3))(delayed(one)(s) for s in range(self.n_chains))
        return self

    def predict_proba(self, X):
        return np.mean([c.predict_proba(X) for c in self.chains_], axis=0)


def score(name, pf, pt, secs):
    thr = ev.tune_thresholds(y[dev_idx], pf[dev_idx], labels)
    o, t = ev.evaluate(y[dev_idx], pf[dev_idx], thr), ev.evaluate(y[test_idx], pt, thr)
    log({"name": name, "oof_macro": o["f1_macro"], "test_macro": t["f1_macro"], "test_micro": t["f1_micro"],
         "test_p3": t["precision_at_3"], "secs": round(secs, 1)})


which = sys.argv[1:] or ["chain", "strat"]
if "chain" in which:
    pf, pt, dt = L.oof_and_test(lambda: ChainEnsemble(3, 0.25), blocks, W, y, S)
    L.save_proba("ecc_lr_C0.25", pf, pt)
    score("ecc_3rantai_lr_C0.25", pf, pt, dt)
    pf, pt, dt = L.oof_and_test(C.lr(C=0.25), blocks, W, y, S)
    score("br_lr_C0.25_pembanding", pf, pt, dt)

if "strat" in which:
    from skmultilearn.model_selection import IterativeStratification
    # scikit-multilearn 0.2 menolak random_state di scikit-learn baru; seed global menggantikannya.
    np.random.seed(42)
    strat = IterativeStratification(n_splits=3, order=1)
    sfolds = list(strat.split(np.zeros((len(dev_idx), 1)), y[dev_idx]))
    S_strat = [(f"is{k}", dev_idx[tr], dev_idx[va]) for k, (tr, va) in enumerate(sfolds)]
    S_strat.append(("final", dev_idx, test_idx))
    blocks_s = B.build(txt, pdf, S_strat, specs)
    t0 = time.time()
    pf, pt, dt = L.oof_and_test(C.svm(C=0.0354, cw=None), blocks_s, W, y, S_strat)
    score("svm_final_fold_stratifikasi_iteratif", pf, pt, dt)
    for k, (_, va) in enumerate(sfolds):
        pos = y[dev_idx][va].sum(axis=0)
        print(f"fold {k}: positif genre terkecil {pos.min()} | rasio maks/min antar fold dihitung di bawah")
    counts = np.array([y[dev_idx][va].sum(axis=0) for _, va in sfolds])
    kcounts = np.array([y[dev_idx][va].sum(axis=0) for _, va in folds])
    log({"name": "sebaran_positif_per_fold", "stratifikasi_iteratif_cv_rata2": round(float((counts.std(0) / counts.mean(0)).mean()), 4),
         "kfold_acak_cv_rata2": round(float((kcounts.std(0) / kcounts.mean(0)).mean()), 4)})
