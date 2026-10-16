"""Harness eksperimen Head A: cache blok fitur per fold, evaluasi OOF + uji.

Protokol identik dengan Model/pipeline/train.py:
  dev/test = train_test_split(test_size=0.2, random_state=42) atas baris yang
  diurutkan series_id; OOF = KFold(3, shuffle, 42) di set dev; ambang per label
  disetel pada OOF; set uji dievaluasi sekali.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold, train_test_split

REPO = Path(__file__).resolve().parents[3]
SP = Path(__file__).resolve().parent
CACHE = REPO / "Model/artifacts/_research_cache"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = SP / "results" / "results.jsonl"

import sys
sys.path.insert(0, str(REPO))

from Model.pipeline import evaluate as ev  # noqa: E402
from Model.pipeline.preprocessing import build_label_matrix, stem_texts  # noqa: E402

RS = 42


def load_frame():
    pdf = pd.read_parquet(REPO / "Model/artifacts/_research_frame39k.parquet")
    stop = json.loads((REPO / "Model/artifacts/_research_stopwords39k.json").read_text(encoding="utf-8"))
    return pdf, stop


def setup():
    pdf, stop = load_frame()
    y, labels, _ = build_label_matrix(pdf["genre_list"], 50)
    dev_idx, test_idx = train_test_split(np.arange(len(pdf)), test_size=0.2, random_state=RS)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=RS).split(dev_idx))
    return pdf, stop, y, labels, dev_idx, test_idx, folds


def splits(dev_idx, test_idx, folds):
    """Daftar (nama, idx_latih, idx_eval) dalam indeks GLOBAL: 3 fold OOF + final."""
    out = [(f"f{k}", dev_idx[tr], dev_idx[va]) for k, (tr, va) in enumerate(folds)]
    out.append(("final", dev_idx, test_idx))
    return out


def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


def cached_block(name: str, texts: pd.Series, vec_factory, split_list, stem=True, tag=""):
    """Fit vectorizer per split pada baris latih; cache (Xtr, Xev) ke disk."""
    out = {}
    stemmed = None
    for sname, tr, evx in split_list:
        key = _key(name, tag, sname, len(tr), int(tr[:50].sum()), stem)
        path = CACHE / f"{name}_{key}.pkl"
        if path.is_file():
            with open(path, "rb") as fh:
                out[sname] = pickle.load(fh)
            continue
        if stemmed is None:
            stemmed = pd.Series(stem_texts(texts) if stem else texts.fillna("").astype(str).tolist())
        vec = vec_factory()
        xtr = vec.fit_transform(stemmed.iloc[tr])
        xev = vec.transform(stemmed.iloc[evx])
        out[sname] = (xtr.tocsr().astype(np.float32), xev.tocsr().astype(np.float32))
        with open(path, "wb") as fh:
            pickle.dump(out[sname], fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def meta_block(pdf, split_list):
    from sklearn.preprocessing import OneHotEncoder
    cols = pdf[["type_origin", "type"]].fillna("UNKNOWN")
    out = {}
    for sname, tr, evx in split_list:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        out[sname] = (enc.fit_transform(cols.iloc[tr]).tocsr().astype(np.float32),
                      enc.transform(cols.iloc[evx]).tocsr().astype(np.float32))
    return out


def combine(blocks: dict, weights: dict, sname: str):
    xs_tr, xs_ev = [], []
    for name, w in weights.items():
        if w == 0 or name not in blocks:
            continue
        xtr, xev = blocks[name][sname]
        xs_tr.append(xtr * w)
        xs_ev.append(xev * w)
    return sparse.hstack(xs_tr).tocsr(), sparse.hstack(xs_ev).tocsr()


def oof_and_test(make_clf, blocks, weights, y, split_list, fit_kwargs=None, need_test=True,
                 transform=None):
    """Kembalikan (proba_oof_dev_order, proba_test, waktu)."""
    t0 = time.time()
    n = y.shape[0]
    proba_full = np.full(y.shape, np.nan, dtype=np.float32)
    proba_test = None
    for sname, tr, evx in split_list:
        if sname == "final" and not need_test:
            continue
        xtr, xev = combine(blocks, weights, sname)
        if transform is not None:
            xtr, xev = transform(xtr, xev, y[tr])
        clf = make_clf()
        clf.fit(xtr, y[tr], **(fit_kwargs or {}))
        p = proba_of(clf, xev)
        if sname == "final":
            proba_test = p
        else:
            proba_full[evx] = p
    return proba_full, proba_test, time.time() - t0


def proba_of(clf, x):
    p = clf.predict_proba(x)
    if isinstance(p, list):
        p = np.column_stack([q[:, 1] if q.shape[1] > 1 else np.zeros(q.shape[0]) for q in p])
    p = np.asarray(p, dtype=np.float32)
    if p.ndim == 3:
        p = p[:, :, 1].T
    return p


def score(y, dev_idx, test_idx, labels, proba_full, proba_test, name, extra=None, thr_fn=None,
          log=True):
    y_dev = y[dev_idx]
    p_oof = proba_full[dev_idx]
    thr_fn = thr_fn or (lambda yt, pp: ev.tune_thresholds(yt, pp, labels))
    thr = thr_fn(y_dev, p_oof)
    oof = ev.evaluate(y_dev, p_oof, thr)
    res = {"name": name, "oof_macro": oof["f1_macro"], "oof_micro": oof["f1_micro"],
           "oof_p3": oof["precision_at_3"]}
    if proba_test is not None:
        te = ev.evaluate(y[test_idx], proba_test, thr)
        res.update({"test_macro": te["f1_macro"], "test_micro": te["f1_micro"],
                    "test_samples": te["f1_samples"], "test_p1": te["precision_at_1"],
                    "test_p3": te["precision_at_3"], "test_hamming": te["hamming_loss"]})
    if extra:
        res.update(extra)
    if log:
        with open(RESULTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
    return res, thr


def save_proba(name, proba_full, proba_test):
    np.savez_compressed(CACHE / f"proba_{name}.npz", oof=proba_full, test=proba_test)


def load_proba(name):
    d = np.load(CACHE / f"proba_{name}.npz")
    return d["oof"], d["test"]
