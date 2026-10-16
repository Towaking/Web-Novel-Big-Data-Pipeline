"""Stacking antar label (Godbole & Sarawagi 2004; 'meta binary relevance').

Level 1: classifier per label (mis. SVM) -> probabilitas 35 genre.
Level 2: per label, LR di atas logit probabilitas SEMUA genre (+ opsional
fitur asli), sehingga korelasi antar genre (Shoujo Ai<->Yuri, Wuxia<->Martial
Arts) bisa dipelajari.

Kejujuran evaluasi: fitur level-2 untuk baris latih berasal dari prediksi
OOF di dalam split latih itu sendiri (CV dalam), sedangkan baris evaluasi
memakai level-1 yang dilatih pada seluruh baris latih split tersebut.
"""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

import exp_lib as L


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def level1_features(make_l1, xtr, xev, ytr, inner_folds=3, seed=42):
    """Kembalikan (Z_train_oof, Z_eval) logit probabilitas level-1."""
    z_tr = np.zeros(ytr.shape, dtype=np.float32)
    for a, b in KFold(inner_folds, shuffle=True, random_state=seed).split(np.arange(xtr.shape[0])):
        m = make_l1()
        m.fit(xtr[a], ytr[a])
        z_tr[b] = L.proba_of(m, xtr[b])
    m = make_l1()
    m.fit(xtr, ytr)
    z_ev = L.proba_of(m, xev)
    return _logit(z_tr), _logit(z_ev), z_ev


def meta_model(C=1.0, cw=None):
    return OneVsRestClassifier(LogisticRegression(C=C, class_weight=cw, max_iter=3000), n_jobs=-1)


def run_stacking(make_l1, blocks, weights, y, split_list, C=1.0, with_x=False, x_scale=1.0,
                 cw=None, z_scale=1.0):
    proba_full = np.full(y.shape, np.nan, dtype=np.float32)
    proba_test = None
    cache = {}
    for sname, tr, evx in split_list:
        xtr, xev = L.combine(blocks, weights, sname)
        ztr, zev, _ = level1_features(make_l1, xtr, xev, y[tr])
        cache[sname] = (ztr, zev)
        if with_x:
            ftr = sparse.hstack([xtr * x_scale, sparse.csr_matrix(ztr * z_scale)]).tocsr()
            fev = sparse.hstack([xev * x_scale, sparse.csr_matrix(zev * z_scale)]).tocsr()
        else:
            ftr, fev = ztr, zev
        m = meta_model(C, cw)
        m.fit(ftr, y[tr])
        p = L.proba_of(m, fev)
        if sname == "final":
            proba_test = p
        else:
            proba_full[evx] = p
    return proba_full, proba_test, cache
