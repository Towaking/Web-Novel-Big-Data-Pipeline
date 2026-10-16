"""Head D: baseline vs perbaikan. Pemilihan via CV 5-fold di set latih; uji sekali."""
import os
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

import exp_lib as L
import headd_lib as H
from textprep_exp import rebuild

pdf, stop, *_ = L.setup()
txt = rebuild(pdf, stop, True)
rows, tr, te = H.split_reliable(pdf)
print("reliable", len(rows), "train", len(tr), "test", len(te))
y_all = pdf["rating_score"].to_numpy(np.float32)
votes = pdf["rating_votes"].fillna(0).to_numpy(np.float32)

t0 = time.time()
emb_p2 = H.shared_embeddings(pdf["corpus_clean"].fillna(""))   # seperti artefak sekarang
emb_fix = H.shared_embeddings(txt["corpus_clean"])             # korpus tanpa sensor
num = H.numeric_frame(pdf)
print("emb %.1fs" % (time.time() - t0), num.shape)

ytr, yte = y_all[tr], y_all[te]
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
    return HistGradientBoostingRegressor(**base)


def cv_mae(make, X, y, w=None, folds=5):
    maes = []
    for a, b in KFold(folds, shuffle=True, random_state=42).split(X):
        m = make()
        m.fit(X[a], y[a], **({} if w is None else {"sample_weight": w[a]}))
        maes.append(np.mean(np.abs(m.predict(X[b]) - y[b])))
    return float(np.mean(maes))


def fit_test(make, Xtr, Xte, w=None):
    m = make()
    m.fit(Xtr, ytr, **({} if w is None else {"sample_weight": w}))
    return m.predict(Xte)


# 0) baseline persis Fase 4
X0 = np.hstack([emb_p2, num.to_numpy()])
p = fit_test(lambda: hgb(), X0[tr], X0[te])
log("d0_baseline", H.reg_metrics(yte, p, ytr.mean()), {"cv_mae": round(cv_mae(lambda: hgb(), X0[tr], ytr), 4)})

# 1) korpus tanpa sensor
X1 = np.hstack([emb_fix, num.to_numpy()])
p = fit_test(lambda: hgb(), X1[tr], X1[te])
log("d1_uncensor", H.reg_metrics(yte, p, ytr.mean()), {"cv_mae": round(cv_mae(lambda: hgb(), X1[tr], ytr), 4)})

# 2) + skor ridge teks (OOF) - stacked generalization
text = (txt["corpus_clean"] + " " + txt["tags_clean"]).reset_index(drop=True)
for alpha in (1.0, 3.0, 10.0):
    oof, te_r = H.ridge_text_oof(text.iloc[tr].reset_index(drop=True), ytr, text.iloc[te].reset_index(drop=True), alpha=alpha)
    log(f"d2_ridge_only_a{alpha}", H.reg_metrics(yte, te_r, ytr.mean()), {"oof_mae": round(float(np.mean(np.abs(oof - ytr))), 4)})
    Xtr2 = np.hstack([X1[tr], oof[:, None]])
    Xte2 = np.hstack([X1[te], te_r[:, None]])
    p = fit_test(lambda: hgb(), Xtr2, Xte2)
    log(f"d2_hgb_plus_ridge_a{alpha}", H.reg_metrics(yte, p, ytr.mean()), {"cv_mae": round(cv_mae(lambda: hgb(), Xtr2, ytr), 4)})
    np.savez(os.path.join(L.CACHE, f"ridge_a{alpha}.npz"), oof=oof, te=te_r)
