"""Varian classifier untuk eksperimen Head A."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import os
NJ = int(os.environ.get("N_JOBS", "-1"))


class NBScaler(BaseEstimator, TransformerMixin):
    """Skala fitur dengan rasio log-count Naive Bayes (Wang & Manning 2012)."""

    def __init__(self, alpha=1.0, binarize=True):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        y = np.asarray(y).ravel().astype(bool)
        X = sparse.csr_matrix(X)
        xb = X.copy()
        if self.binarize:
            xb.data = np.ones_like(xb.data)
        p = self.alpha + np.asarray(xb[y].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(xb[~y].sum(axis=0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        return self

    def transform(self, X):
        return sparse.csr_matrix(X) @ sparse.diags(self.r_)


def svm(C=0.3, cw="balanced", method="sigmoid", cv=3, n_jobs=NJ, ensemble=True):
    return lambda: OneVsRestClassifier(CalibratedClassifierCV(
        LinearSVC(C=C, class_weight=cw, max_iter=10000, dual="auto", random_state=42),
        method=method, cv=cv, ensemble=ensemble), n_jobs=n_jobs)


def lr(C=4.0, cw=None, solver="lbfgs", n_jobs=NJ):
    return lambda: OneVsRestClassifier(LogisticRegression(max_iter=5000, C=C, class_weight=cw, solver=solver), n_jobs=n_jobs)


def nb_svm(C=0.3, cw="balanced", alpha=1.0, binarize=True, method="sigmoid"):
    return lambda: OneVsRestClassifier(CalibratedClassifierCV(
        Pipeline([("nb", NBScaler(alpha, binarize)),
                  ("svc", LinearSVC(C=C, class_weight=cw, max_iter=10000, dual="auto", random_state=42))]),
        method=method, cv=3), n_jobs=NJ)


def nb_lr(C=4.0, cw=None, alpha=1.0, binarize=True):
    return lambda: OneVsRestClassifier(Pipeline([
        ("nb", NBScaler(alpha, binarize)),
        ("lr", LogisticRegression(max_iter=5000, C=C, class_weight=cw))]), n_jobs=NJ)


def cnb(alpha=0.3):
    return lambda: OneVsRestClassifier(ComplementNB(alpha=alpha), n_jobs=NJ)


def mnb(alpha=0.1):
    return lambda: OneVsRestClassifier(MultinomialNB(alpha=alpha), n_jobs=NJ)


def sgd_mh(alpha=1e-5, cw="balanced"):
    return lambda: OneVsRestClassifier(SGDClassifier(loss="modified_huber", alpha=alpha, class_weight=cw,
                                                     max_iter=50, tol=1e-4, random_state=42), n_jobs=NJ)


class Chi2TopK:
    """Pilih k fitur dengan skor chi2 maksimum lintas label (multi-label)."""

    def __init__(self, k):
        self.k = k

    def __call__(self, xtr, xev, ytr):
        scores = np.zeros(xtr.shape[1])
        for j in range(ytr.shape[1]):
            s, _ = chi2(xtr, ytr[:, j])
            scores = np.maximum(scores, np.nan_to_num(s))
        keep = np.sort(np.argsort(-scores)[: self.k])
        return xtr[:, keep], xev[:, keep]


def rf(n=300, max_features="sqrt", min_leaf=1, cw="balanced_subsample", max_depth=None,
       max_samples=None, extra=False, n_jobs=NJ, max_leaf_nodes=None):
    cls = ExtraTreesClassifier if extra else RandomForestClassifier
    return lambda: cls(n_estimators=n, max_features=max_features, min_samples_leaf=min_leaf,
                       class_weight=cw, max_depth=max_depth, max_samples=max_samples,
                       max_leaf_nodes=max_leaf_nodes, bootstrap=True, n_jobs=n_jobs, random_state=42)
