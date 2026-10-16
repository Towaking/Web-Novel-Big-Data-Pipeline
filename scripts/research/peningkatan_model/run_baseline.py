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

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
S = L.splits(dev_idx, test_idx, folds)
print("labels", len(labels), "dev", len(dev_idx), "test", len(test_idx))

def tfidf(min_df, max_features):
    return lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.85,
                                   max_features=max_features, sublinear_tf=True, dtype=np.float32)

t0 = time.time()
tags_clean = pdf["tags"].fillna("").str.lower().str.replace(",", " ", regex=False)
blocks = {
    "corpus": L.cached_block("corpus", pdf["corpus_clean"].fillna(""), tfidf(3, 40_000), S, tag="p2"),
    "synopsis": L.cached_block("synopsis", pdf["synopsis_clean"].fillna(""), tfidf(3, 40_000), S, tag="p2"),
    "tags": L.cached_block("tags", tags_clean, tfidf(2, 10_000), S, tag="p2"),
    "meta": L.meta_block(pdf, S),
}
print("blocks built %.1fs" % (time.time() - t0), {k: v["final"][0].shape for k, v in blocks.items()})
W = {"corpus": 1.0, "synopsis": 1.0, "tags": 1.6, "meta": 0.6}

def svm(C=0.3, cw="balanced"):
    return lambda: OneVsRestClassifier(CalibratedClassifierCV(
        LinearSVC(C=C, class_weight=cw, max_iter=5000, dual="auto", random_state=42),
        method="sigmoid", cv=3), n_jobs=-1)

def lr(C=4.0, cw=None):
    return lambda: OneVsRestClassifier(LogisticRegression(max_iter=3000, C=C, class_weight=cw), n_jobs=-1)

for name, mk in [("base_svm_C0.3_bal", svm()), ("base_lr_C4_none", lr())]:
    pf, pt, dt = L.oof_and_test(mk, blocks, W, y, S)
    L.save_proba(name, pf, pt)
    L.score(y, dev_idx, test_idx, labels, pf, pt, name, {"secs": round(dt, 1)})
