import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, copy
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import exp_lib as L
import blocks_lib as B

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
txt = B.text_columns(pdf, stop, True)


def svm(C=0.3, cw="balanced"):
    return lambda: OneVsRestClassifier(CalibratedClassifierCV(
        LinearSVC(C=C, class_weight=cw, max_iter=5000, dual="auto", random_state=42),
        method="sigmoid", cv=3), n_jobs=int(os.environ.get("N_JOBS", "-1")))


def variant(**changes):
    specs = copy.deepcopy(B.BASE_SPECS)
    w = dict(B.BASE_W)
    for blk, upd in changes.items():
        if blk == "W":
            w.update(upd)
        elif upd is None:
            specs.pop(blk, None); w.pop(blk, None)
        else:
            specs.setdefault(blk, {}).update(upd)
    return specs, w


VARIANTS = {
    "v0_base": variant(),
    "v1_maxfeat_big": variant(corpus={"max_features": 150_000}, synopsis={"max_features": 150_000}, tags={"max_features": 20_000}),
    "v2_all_min2": variant(corpus={"max_features": None, "min_df": 2}, synopsis={"max_features": None, "min_df": 2}, tags={"max_features": None, "min_df": 2}),
    "v3_ngram13": variant(corpus={"max_features": 150_000, "ngram": (1, 3)}, synopsis={"max_features": 150_000, "ngram": (1, 3)}),
    "v4_tags_phrase": variant(tags={"col": "tags_phrase", "ngram": (1, 1), "stem": False, "max_features": None}),
    "v5_tags_plus_phrase": variant(tagp={"col": "tags_phrase", "ngram": (1, 1), "stem": False, "min_df": 2, "max_features": None}, W={"tagp": 1.0}),
    "v6_char": variant(char={"col": "synopsis_raw", "analyzer": "char_wb", "ngram": (3, 5), "min_df": 5, "max_features": 100_000, "stem": False}, W={"char": 1.0}),
    "v7_nostem": variant(corpus={"stem": False}, synopsis={"stem": False}, tags={"stem": False}),
    "v8_bm25": variant(corpus={"kind": "bm25"}, synopsis={"kind": "bm25"}),
    "v9_maxdf05": variant(corpus={"max_df": 0.5}, synopsis={"max_df": 0.5}, tags={"max_df": 0.5}),
}

only = sys.argv[1:] or list(VARIANTS)
for name in only:
    specs, w = VARIANTS[name]
    t0 = time.time()
    blocks = B.build(txt, pdf, S, specs)
    tb = time.time() - t0
    dims = {k: v["final"][0].shape[1] for k, v in blocks.items()}
    pf, pt, dt = L.oof_and_test(svm(), blocks, w, y, S)
    L.save_proba(f"b1_{name}", pf, pt)
    L.score(y, dev_idx, test_idx, labels, pf, pt, f"b1_{name}", {"secs": round(dt, 1), "build_secs": round(tb, 1), "dims": dims})
