"""Uji berpasangan lintas seed: konfigurasi lama vs baru pada pembagian yang SAMA.

Per seed: split dev/test baru (80/20), OOF 3-fold di dev untuk ambang, fit
final di dev, skor uji. Ensemble SVM+RF dirakit dari probabilitas yang sama,
bobotnya dipilih di OOF seed itu. Statistik dihitung di paired_stats.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

import exp_lib as L
import clf_lib as CL
from Model.pipeline import evaluate as ev
from Model.pipeline.preprocessing import build_label_matrix, stem_texts
from textprep_exp import rebuild

NJ = int(os.environ.get("N_JOBS", "-1"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,123,2024,99").split(",")]
OUT = os.path.join(os.path.dirname(__file__), "results", "paired_results.jsonl")

FEATS = {
    "new75": {"text": "fix", "blocks": {"corpus_clean": (3, 75_000, 2.31), "synopsis_clean": (3, 75_000, 0.95),
                                         "tags_clean": (2, 15_000, 1.48)}, "meta_w": 0.48},
}
CONFIGS = {
    "svm_new75": ("new75", "svm", {"C": 0.0354, "cw": None}),
    "rf_new75": ("new75", "rf", {"n": 60, "mln": 2000}),
}

pdf, stop = L.load_frame()
y, labels, _ = build_label_matrix(pdf["genre_list"], 50)
texts = {"p2": {"corpus_clean": pdf["corpus_clean"].fillna(""), "synopsis_clean": pdf["synopsis_clean"].fillna(""),
                "tags_clean": pdf["tags"].fillna("").str.lower().str.replace(",", " ", regex=False)},
         "fix": rebuild(pdf, stop, True)}
stemmed = {k: {c: np.array(stem_texts(v[c]), dtype=object) for c in ("corpus_clean", "synopsis_clean", "tags_clean")}
           for k, v in texts.items()}
meta_cols = pdf[["type_origin", "type"]].fillna("UNKNOWN")


def features(spec, tr, ev_idx):
    xs_tr, xs_ev = [], []
    st = stemmed[spec["text"]]
    for col, (mdf, mf, w) in spec["blocks"].items():
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=mdf, max_df=0.85, max_features=mf, sublinear_tf=True,
                              dtype=np.float32)
        xs_tr.append(vec.fit_transform(st[col][tr]) * w)
        xs_ev.append(vec.transform(st[col][ev_idx]) * w)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    xs_tr.append(enc.fit_transform(meta_cols.iloc[tr]) * spec["meta_w"])
    xs_ev.append(enc.transform(meta_cols.iloc[ev_idx]) * spec["meta_w"])
    return sparse.hstack(xs_tr).tocsr(), sparse.hstack(xs_ev).tocsr()


def fit_predict(kind, params, xa, ya, xb):
    if kind == "svm":
        m = OneVsRestClassifier(CalibratedClassifierCV(
            LinearSVC(C=params["C"], class_weight=params["cw"], max_iter=10000, dual="auto", random_state=42),
            method="sigmoid", cv=3), n_jobs=NJ)
    elif kind == "lr":
        m = OneVsRestClassifier(LogisticRegression(C=params["C"], max_iter=5000), n_jobs=NJ)
    else:
        xa, xb = CL.Chi2TopK(3000)(xa, xb, ya)
        m = RandomForestClassifier(n_estimators=params.get("n", 100), max_leaf_nodes=params.get("mln", 4000),
                                   max_features="sqrt", n_jobs=NJ, random_state=42)
    m.fit(xa, ya)
    return L.proba_of(m, xb)


def log(rec):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


for seed in SEEDS:
    dev, test = train_test_split(np.arange(len(pdf)), test_size=0.2, random_state=seed)
    splits = [(dev[a], dev[b], b) for a, b in KFold(3, shuffle=True, random_state=42).split(dev)]
    probs = {}
    for fname, spec in FEATS.items():
        t0 = time.time()
        mats = [features(spec, a, b) for a, b, _ in splits] + [features(spec, dev, test)]
        for cname, (fn, kind, params) in CONFIGS.items():
            if fn != fname:
                continue
            oof = np.zeros((len(dev), y.shape[1]), dtype=np.float32)
            for (a, b, pos), (xa, xb) in zip(splits, mats[:3]):
                oof[pos] = fit_predict(kind, params, xa, y[a], xb)
            pt = fit_predict(kind, params, mats[3][0], y[dev], mats[3][1])
            probs[cname] = (oof, pt)
            thr = ev.tune_thresholds(y[dev], oof, labels)
            te = ev.evaluate(y[test], pt, thr)
            log({"seed": seed, "config": cname, "oof_macro": ev.evaluate(y[dev], oof, thr)["f1_macro"],
                 "test_macro": te["f1_macro"], "test_micro": te["f1_micro"], "test_p3": te["precision_at_3"],
                 "secs": round(time.time() - t0, 1)})
    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    for space in ("prob", "logit"):
        def mix(w, a, b):
            if space == "prob":
                return w * a + (1 - w) * b
            return 1 / (1 + np.exp(-(w * logit(a) + (1 - w) * logit(b))))
        best = None
        for w in np.round(np.arange(0, 1.0001, 0.1), 1):
            oof = mix(w, probs["svm_new75"][0], probs["rf_new75"][0])
            thr = ev.tune_thresholds(y[dev], oof, labels)
            m = ev.evaluate(y[dev], oof, thr)["f1_macro"]
            if best is None or m > best[0]:
                best = (m, w, thr)
        m, w, thr = best
        pt = mix(w, probs["svm_new75"][1], probs["rf_new75"][1])
        te = ev.evaluate(y[test], pt, thr)
        log({"seed": seed, "config": f"ens_{space}75", "w_svm": float(w), "oof_macro": m, "test_macro": te["f1_macro"],
             "test_micro": te["f1_micro"], "test_p3": te["precision_at_3"], "test_p1": te["precision_at_1"]})
