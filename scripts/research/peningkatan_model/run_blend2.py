"""Strategi penggabungan SVM + RF: ruang probabilitas vs log-odds, dan batasan P@3.

Semua bobot dipilih di OOF set pengembangan; set uji hanya untuk laporan.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import exp_lib as L
import thr_lib as T
from Model.pipeline import evaluate as ev

SVM, RF = "final_svm_v12_75k_C0.0354", "rf_v12_75k_tw_rf_chi2k3000_n60_mln2000_none"
pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
fold_ids = np.zeros(len(dev_idx), dtype=int)
for k, (_, va) in enumerate(folds):
    fold_ids[va] = k
(s_oof, s_te), (r_oof, r_te) = L.load_proba(SVM), L.load_proba(RF)
s_oof, r_oof = s_oof[dev_idx], r_oof[dev_idx]
y_dev, y_te = y[dev_idx], y[test_idx]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mix(kind, w, a, b):
    if kind == "prob":
        return w * a + (1 - w) * b
    if kind == "logit":
        return 1 / (1 + np.exp(-(w * logit(a) + (1 - w) * logit(b))))
    if kind == "geo":
        return np.exp(w * np.log(np.clip(a, 1e-4, 1)) + (1 - w) * np.log(np.clip(b, 1e-4, 1)))
    raise ValueError(kind)


def report(name, oof, te, extra=None):
    thr = T.tune_grid(y_dev, oof)
    o = ev.evaluate(y_dev, oof, thr)
    t = ev.evaluate(y_te, te, thr)
    cv = T.cv_threshold_score(y_dev, oof, fold_ids, lambda a, b: T.tune_grid(a, b))
    rec = {"name": name, "oof_macro": o["f1_macro"], "cvthr_macro": cv["macro"], "oof_p3": o["precision_at_3"],
           "test_macro": t["f1_macro"], "test_micro": t["f1_micro"], "test_p3": t["precision_at_3"],
           "test_p1": t["precision_at_1"], **(extra or {})}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "blend_results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


svm_only = report("svm_saja", s_oof, s_te)
p3_floor = svm_only["oof_p3"]
for kind in ("prob", "logit", "geo"):
    rows = []
    for w in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        oof, te = mix(kind, w, s_oof, r_oof), mix(kind, w, s_te, r_te)
        thr = T.tune_grid(y_dev, oof)
        o = ev.evaluate(y_dev, oof, thr)
        rows.append((w, o["f1_macro"], o["precision_at_3"]))
    best = max(rows, key=lambda r: r[1])
    allowed = [r for r in rows if r[2] >= p3_floor]
    best_c = max(allowed, key=lambda r: r[1]) if allowed else None
    print(kind, "grid:", [(w, round(m, 4), round(p, 4)) for w, m, p in rows], flush=True)
    report(f"{kind}_bebas_w{best[0]}", mix(kind, best[0], s_oof, r_oof), mix(kind, best[0], s_te, r_te),
           {"w_svm": best[0]})
    if best_c is not None:
        report(f"{kind}_p3dijaga_w{best_c[0]}", mix(kind, best_c[0], s_oof, r_oof), mix(kind, best_c[0], s_te, r_te),
               {"w_svm": best_c[0], "syarat": "OOF P@3 >= SVM saja"})
