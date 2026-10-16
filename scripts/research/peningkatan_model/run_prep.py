import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import exp_lib as L
from textprep_exp import rebuild

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)


def tfidf(min_df, max_features):
    return lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.85,
                                   max_features=max_features, sublinear_tf=True, dtype=np.float32)


def svm(C=0.3, cw="balanced"):
    return lambda: OneVsRestClassifier(CalibratedClassifierCV(
        LinearSVC(C=C, class_weight=cw, max_iter=5000, dual="auto", random_state=42),
        method="sigmoid", cv=3), n_jobs=-1)


W = {"corpus": 1.0, "synopsis": 1.0, "tags": 1.6, "meta": 0.6}
meta = L.meta_block(pdf, S)
for fix in (False, True):
    tag = f"rb_fix{int(fix)}"
    t0 = time.time()
    txt = rebuild(pdf, stop, fix)
    if not fix:
        same = (txt["corpus_clean"] == pdf["corpus_clean"].fillna("")).mean()
        same_s = (txt["synopsis_clean"] == pdf["synopsis_clean"].fillna("")).mean()
        print(f"rebuild identik dgn Fase 2: corpus {same:.4f} synopsis {same_s:.4f}")
    blocks = {
        "corpus": L.cached_block("corpus", txt["corpus_clean"], tfidf(3, 40_000), S, tag=tag),
        "synopsis": L.cached_block("synopsis", txt["synopsis_clean"], tfidf(3, 40_000), S, tag=tag),
        "tags": L.cached_block("tags", txt["tags_clean"], tfidf(2, 10_000), S, tag=tag),
        "meta": meta,
    }
    print("blocks %s %.1fs" % (tag, time.time() - t0))
    pf, pt, dt = L.oof_and_test(svm(), blocks, W, y, S)
    L.save_proba(f"{tag}_svm", pf, pt)
    L.score(y, dev_idx, test_idx, labels, pf, pt, f"{tag}_svm_C0.3", {"secs": round(dt, 1)})
