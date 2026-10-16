"""Eksperimen Head D (estimasi rating) - fitur, model, protokol."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import normalize

import exp_lib as L

RS = 42


def shared_embeddings(corpus: pd.Series):
    """Representasi bersama Fase 4 (TF-IDF 30k -> SVD 200 -> L2), dilatih atas semua dokumen."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=30_000, sublinear_tf=True)
    tfidf = vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=200, random_state=RS)
    emb = normalize(svd.fit_transform(tfidf)).astype(np.float32)
    return emb


def numeric_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    origin = pd.get_dummies(pdf["type_origin"].fillna("UNKNOWN"), prefix="origin")
    typ = pd.get_dummies(pdf["type"].fillna("UNKNOWN"), prefix="type")
    frame = pd.DataFrame({
        "log_chapters": np.log1p(pdf["chapter_count"].fillna(pdf["chapter_count"].median())),
        "n_tags": pdf["n_tags"].fillna(0),
        "token_count": pdf["token_count"].fillna(0),
        "year": pdf["year"].fillna(pdf["year"].median()),
        "is_completed": pdf["is_completed"].fillna(0),
        "log_days_per_chapter": np.log1p(pdf["days_per_chapter"].fillna(pdf["days_per_chapter"].median())),
    })
    return pd.concat([frame, origin, typ], axis=1).astype(np.float32)


def split_reliable(pdf):
    mask = pdf["is_rated_reliable"].fillna(False).to_numpy(dtype=bool)
    rows = np.flatnonzero(mask)
    pos_tr, pos_te = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=RS)
    return rows, rows[pos_tr], rows[pos_te]


def reg_metrics(y, p, base_mean):
    base = np.full_like(y, base_mean)
    m = {
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "r2": round(float(r2_score(y, p)), 4),
        "baseline_mae": round(float(mean_absolute_error(y, base)), 4),
    }
    m["improvement_pct"] = round(100 * (m["baseline_mae"] - m["mae"]) / m["baseline_mae"], 2)
    return m


def ridge_text_oof(text_tr: pd.Series, y_tr, text_te: pd.Series, alpha=3.0, folds=5,
                   vec_kwargs=None, w_tr=None):
    """Stacked generalization (Wolpert 1992): prediksi ridge teks yang OOF untuk baris latih."""
    vec_kwargs = vec_kwargs or dict(ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=200_000,
                                    sublinear_tf=True, dtype=np.float32)
    oof = np.zeros(len(text_tr), dtype=np.float32)
    for a, b in KFold(folds, shuffle=True, random_state=RS).split(np.arange(len(text_tr))):
        vec = TfidfVectorizer(**vec_kwargs)
        xa = vec.fit_transform(text_tr.iloc[a])
        m = Ridge(alpha=alpha)
        m.fit(xa, y_tr[a] - y_tr[a].mean(), sample_weight=None if w_tr is None else w_tr[a])
        oof[b] = m.predict(vec.transform(text_tr.iloc[b])) + y_tr[a].mean()
    vec = TfidfVectorizer(**vec_kwargs)
    xa = vec.fit_transform(text_tr)
    m = Ridge(alpha=alpha)
    m.fit(xa, y_tr - y_tr.mean(), sample_weight=w_tr)
    te = m.predict(vec.transform(text_te)) + y_tr.mean()
    return oof, te.astype(np.float32)


def tag_multihot(tags: pd.Series, min_count=50):
    from sklearn.feature_extraction.text import CountVectorizer
    from blocks_lib import tag_tokens
    toks = tag_tokens(tags)
    cv = CountVectorizer(token_pattern=r"\S+", min_df=min_count, binary=True, dtype=np.float32)
    return cv, cv.fit_transform(toks)
