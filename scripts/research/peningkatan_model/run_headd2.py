"""Head D batch 2: di atas fitur terbaik (embedding + numerik + skor ridge OOF a=3)."""
import os
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

import exp_lib as L
import headd_lib as H
from textprep_exp import rebuild

pdf, stop, *_ = L.setup()
txt = rebuild(pdf, stop, True)
rows, tr, te = H.split_reliable(pdf)
y_all = pdf["rating_score"].to_numpy(np.float32)
votes = pdf["rating_votes"].fillna(0).to_numpy(np.float32)
ytr, yte = y_all[tr], y_all[te]
emb = H.shared_embeddings(txt["corpus_clean"])
num = H.numeric_frame(pdf).to_numpy()
r = np.load(os.path.join(L.CACHE, "ridge_a3.0.npz"))
X = np.hstack([emb, num])
Xtr = np.hstack([X[tr], r["oof"][:, None]])
Xte = np.hstack([X[te], r["te"][:, None]])
_, tags_mh = H.tag_multihot(pdf["tags"])
tags_mh = tags_mh.toarray().astype(np.float32)
OUT = os.path.join(os.path.dirname(__file__), "results", "headd_results.jsonl")


def log(name, m, extra=None):
    rec = {"name": name, **m, **(extra or {})}
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def hgb(**kw):
    base = dict(max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
                random_state=42, early_stopping=True, validation_fraction=0.15)
    base.update(kw)
    return lambda: HistGradientBoostingRegressor(**base)


def cv_mae(make, Xa, y, w=None, folds=5):
    maes = []
    for a, b in KFold(folds, shuffle=True, random_state=42).split(Xa):
        m = make()
        m.fit(Xa[a], y[a], **({} if w is None else {"sample_weight": w[a]}))
        maes.append(np.mean(np.abs(m.predict(Xa[b]) - y[b])))
    return round(float(np.mean(maes)), 4)


def run(name, make, Xa, Xb, w=None):
    t0 = time.time()
    cv = cv_mae(make, Xa, ytr, w)
    m = make()
    m.fit(Xa, ytr, **({} if w is None else {"sample_weight": w}))
    log(name, H.reg_metrics(yte, m.predict(Xb), ytr.mean()), {"cv_mae": cv, "secs": round(time.time() - t0, 1)})


which = sys.argv[1:] or ["weights", "loss", "trees", "tags"]
if "weights" in which:
    for wname, w in [("log1p", np.log1p(votes[tr])), ("shrink20", votes[tr] / (votes[tr] + 20)),
                     ("sqrt", np.sqrt(votes[tr]))]:
        run(f"d3_ridge3_w_{wname}", hgb(), Xtr, Xte, w)
if "loss" in which:
    run("d4_ridge3_abs", hgb(loss="absolute_error"), Xtr, Xte)
if "trees" in which:
    run("d5_rf_n300_leaf5_mf0.3", lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=8, random_state=42), Xtr, Xte)
    run("d5_et_n300_leaf5_mf0.3", lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=8, random_state=42), Xtr, Xte)
if "tags" in which:
    Xtr_t = np.hstack([Xtr, tags_mh[tr]])
    Xte_t = np.hstack([Xte, tags_mh[te]])
    run("d6_ridge3_tagsmh", hgb(), Xtr_t, Xte_t)
