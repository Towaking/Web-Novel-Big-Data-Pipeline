"""TPE (Bergstra dkk. 2011, via Optuna/Akiba dkk. 2019) atas bobot blok + C.

Objektif = makro-F1 OOF (3-fold, ambang per genre disetel di OOF) - sama persis
dengan skor pemilihan di Model/pipeline/train.py. Set uji TIDAK disentuh di
sini; skor uji konfigurasi terbaik dihitung terpisah, sekali.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optuna

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from Model.pipeline import evaluate as ev
from run_batch1_specs import get_variant

MODEL = os.environ.get("MODEL", "svm")
REP = os.environ.get("REP", "v1_maxfeat_big")
N_TRIALS = int(os.environ.get("N_TRIALS", "40"))

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
S_oof = [s for s in S if s[0] != "final"]
txt = B.text_columns(pdf, stop, True)
specs, W0 = get_variant(REP)
blocks = B.build(txt, pdf, S, specs)
extra = [b for b in W0 if b not in ("corpus", "synopsis", "tags", "meta")]


def make(params):
    if MODEL == "svm":
        return C.svm(C=params["C"], cw=params["cw"])
    return C.lr(C=params["C"], cw=params["cw"])


def objective(trial):
    w = {
        "corpus": trial.suggest_float("w_corpus", 0.3, 2.5),
        "synopsis": trial.suggest_float("w_synopsis", 0.0, 2.0),
        "tags": trial.suggest_float("w_tags", 0.5, 3.0),
        "meta": trial.suggest_float("w_meta", 0.0, 1.5),
    }
    for b in extra:
        w[b] = trial.suggest_float(f"w_{b}", 0.0, 2.5)
    if MODEL == "svm":
        params = {"C": trial.suggest_float("C", 0.03, 2.0, log=True),
                  "cw": trial.suggest_categorical("cw", ["balanced", "none"])}
    else:
        params = {"C": trial.suggest_float("C", 0.3, 20.0, log=True),
                  "cw": trial.suggest_categorical("cw", ["balanced", "none"])}
    params["cw"] = None if params["cw"] == "none" else params["cw"]
    t0 = time.time()
    pf, _, _ = L.oof_and_test(make(params), blocks, w, y, S_oof, need_test=False)
    p = pf[dev_idx]
    thr = ev.tune_thresholds(y[dev_idx], p, labels)
    score = ev.evaluate(y[dev_idx], p, thr)["f1_macro"]
    trial.set_user_attr("secs", round(time.time() - t0, 1))
    return score


study_name = f"{MODEL}_{REP}"
storage = f"sqlite:///{L.CACHE / 'optuna.db'}"
study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage,
                            load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=10))
if len(study.trials) == 0:
    start = {"w_corpus": W0["corpus"], "w_synopsis": W0["synopsis"], "w_tags": W0["tags"], "w_meta": W0["meta"],
             "C": 0.3 if MODEL == "svm" else 4.0, "cw": "balanced" if MODEL == "svm" else "none"}
    for b in extra:
        start[f"w_{b}"] = W0[b]
    study.enqueue_trial(start)
remaining = N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
study.optimize(objective, n_trials=max(remaining, 0), show_progress_bar=False,
               callbacks=[lambda s, t: print(json.dumps({"trial": t.number, "value": t.value, **t.params, **t.user_attrs}), flush=True)])
print("BEST", json.dumps({"value": study.best_value, **study.best_params}))
