"""Konfigurasi hasil TPE: uji representasi alternatif + LogReg dengan bobot yang sama.

Pemakaian: BEST='{"C":..,"cw":..,"w_corpus":..,...}' python run_final_check.py rep1 rep2 ...
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json
sys.path.insert(0, os.path.dirname(__file__))

import exp_lib as L
import blocks_lib as B
import clf_lib as C
from run_batch1_specs import get_variant

best = json.loads(os.environ["BEST"])
MODEL = os.environ.get("MODEL", "svm")
pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)

for rep in sys.argv[1:]:
    specs, W = get_variant(rep)
    W.update({"corpus": best["w_corpus"], "synopsis": best["w_synopsis"], "tags": best["w_tags"], "meta": best["w_meta"]})
    for b in [k for k in W if k not in ("corpus", "synopsis", "tags", "meta")]:
        W[b] = best.get(f"w_{b}", W[b])
    blocks = B.build(txt, pdf, S, specs)
    cw = None if best["cw"] in (None, "none") else best["cw"]
    Cs = [best["C"]] if MODEL == "svm" else [float(c) for c in os.environ.get("LR_CS", "2,4,8").split(",")]
    for c in Cs:
        make = C.svm(C=c, cw=cw) if MODEL == "svm" else C.lr(C=c, cw=None)
        pf, pt, dt = L.oof_and_test(make, blocks, W, y, S)
        name = f"final_{MODEL}_{rep}_C{round(c, 4)}"
        L.save_proba(name, pf, pt)
        L.score(y, dev_idx, test_idx, labels, pf, pt, name, {"secs": round(dt, 1), "weights": W})
