"""Blok fitur berbasis spesifikasi + cache per (spesifikasi, split)."""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize as l2norm

import exp_lib as L
from src.common.text_prep import uncensor
from textprep_exp import rebuild

_SLUG = re.compile(r"[^a-z0-9]+")


def tag_tokens(tags: pd.Series) -> pd.Series:
    """Setiap tag -> satu token utuh (frasa), mis. 'Weak to Strong' -> tag_weak_to_strong."""
    def conv(s):
        out = []
        for t in str(s).split(","):
            t = uncensor(t.strip().lower())
            slug = _SLUG.sub("_", t).strip("_")
            if slug:
                out.append("tag_" + slug)
        return " ".join(out)
    return tags.fillna("").map(conv)


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """Okapi BM25 (Robertson & Zaragoza 2009) sebagai pembobot fitur dokumen."""

    def __init__(self, k1=1.2, b=0.75, ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=None):
        self.k1, self.b = k1, b
        self.ngram_range, self.min_df, self.max_df, self.max_features = ngram_range, min_df, max_df, max_features

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
                                   max_features=self.max_features, dtype=np.float32)
        tf = self.cv_.fit_transform(X)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf_ = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.avgdl_ = float(tf.sum(axis=1).mean())
        return self

    def transform(self, X):
        tf = self.cv_.transform(X).tocsr().astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        denom_row = self.k1 * (1 - self.b + self.b * dl / self.avgdl_)
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + denom_row[rows])
        out = sparse.csr_matrix((data * self.idf_[tf.indices], tf.indices, tf.indptr), shape=tf.shape)
        return l2norm(out)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def text_columns(pdf, stop, fix_censor=True):
    txt = rebuild(pdf, stop, fix_censor)
    txt["tags_phrase"] = tag_tokens(pdf["tags"]).values
    txt["tags_mix"] = (txt["tags_clean"] + " " + txt["tags_phrase"]).values
    txt["synopsis_raw"] = pdf["synopsis"].fillna("").str.lower().map(uncensor).values
    return txt


def make_vec(spec):
    kind = spec.get("kind", "tfidf")
    if kind == "bm25":
        return lambda: BM25Vectorizer(k1=spec.get("k1", 1.2), b=spec.get("b", 0.75),
                                      ngram_range=spec.get("ngram", (1, 2)), min_df=spec["min_df"],
                                      max_df=spec.get("max_df", 0.85), max_features=spec.get("max_features"))
    return lambda: TfidfVectorizer(
        analyzer=spec.get("analyzer", "word"), ngram_range=spec.get("ngram", (1, 2)),
        min_df=spec["min_df"], max_df=spec.get("max_df", 0.85), max_features=spec.get("max_features"),
        sublinear_tf=spec.get("sublinear", True), binary=spec.get("binary", False),
        token_pattern=spec.get("token_pattern", r"(?u)\b\w\w+\b"), dtype=np.float32)


def build(txt, pdf, S, specs: dict, variant="fix1"):
    blocks = {}
    for name, spec in specs.items():
        if name == "meta":
            blocks[name] = L.meta_block(pdf, S)
            continue
        col = spec["col"]
        tag = f"{variant}|{sorted(spec.items())}"
        blocks[name] = L.cached_block(name, txt[col], make_vec(spec), S, stem=spec.get("stem", True), tag=tag)
    return blocks


BASE_SPECS = {
    "corpus": {"col": "corpus_clean", "min_df": 3, "max_features": 40_000},
    "synopsis": {"col": "synopsis_clean", "min_df": 3, "max_features": 40_000},
    "tags": {"col": "tags_clean", "min_df": 2, "max_features": 10_000},
    "meta": {},
}
BASE_W = {"corpus": 1.0, "synopsis": 1.0, "tags": 1.6, "meta": 0.6}
