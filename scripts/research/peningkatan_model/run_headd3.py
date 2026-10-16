"""Head D batch 3: gabungan (ridge + tag multi-hot + bobot log1p) lalu TPE untuk HGB."""
import os
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
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
_, tags_mh = H.tag_multihot(pdf["tags"])
tags_mh = tags_mh.toarray().astype(np.float32)
X = np.hstack([emb, num])
Xtr = np.hstack([X[tr], r["oof"][:, None], tags_mh[tr]])
Xte = np.hstack([X[te], r["te"][:, None], tags_mh[te]])
w_tr = np.log1p(votes[tr])
OUT = os.path.join(os.path.dirname(__file__), "results", "headd_results.jsonl")
folds = list(KFold(3, shuffle=True, random_state=42).split(Xtr))


def log(name, m, extra=None):
    rec = {"name": name, **m, **(extra or {})}
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def hgb(params):
    base = dict(max_iter=800, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
                random_state=42, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)
    base.update(params)
    return HistGradientBoostingRegressor(**base)


def cv(params, w=w_tr):
    maes = []
    for a, b in folds:
        m = hgb(params)
        m.fit(Xtr[a], ytr[a], sample_weight=None if w is None else w[a])
        maes.append(np.mean(np.abs(m.predict(Xtr[b]) - ytr[b])))
    return float(np.mean(maes))


def test(params, w=w_tr):
    m = hgb(params)
    m.fit(Xtr, ytr, sample_weight=w)
    return H.reg_metrics(yte, m.predict(Xte), ytr.mean()), m


DEFAULT = dict(max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0, n_iter_no_change=10)
mt, _ = test(DEFAULT)
log("d7_combo_default", mt, {"cv_mae": round(cv(DEFAULT), 4)})


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }
    return cv(params)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
study.enqueue_trial({"learning_rate": 0.06, "max_leaf_nodes": 31, "max_depth": 6, "min_samples_leaf": 20,
                     "l2_regularization": 1.0, "max_features": 1.0})
n = int(os.environ.get("N_TRIALS", "40"))
study.optimize(objective, n_trials=n, callbacks=[lambda s, t: print(json.dumps({"trial": t.number, "cv_mae": round(t.value, 4), **t.params}), flush=True)])
best = dict(study.best_params)
mt, model = test(best)
log("d8_combo_tpe", mt, {"cv_mae": round(study.best_value, 4), "params": best, "n_iter": int(model.n_iter_)})
