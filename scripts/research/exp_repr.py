"""
Eksperimen representasi — protokol IDENTIK dengan Model/pipeline/train.py
========================================================================
Pembagian, jumlah fold, penyetelan ambang, dan metrik semuanya diambil dari
modul produksi supaya angkanya langsung sebanding dengan model card.

Yang dibandingkan: ruang fitur, bukan klasifikatornya. Klasifikator dikunci
(OvR Logistic Regression / SVM) agar selisih yang muncul benar-benar berasal
dari representasi.
"""
from __future__ import annotations
import sys, os, json, time, argparse
sys.path.insert(0, '/opt/project')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np, pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from Model.pipeline import evaluate as ev
from Model.pipeline.config import MIN_GENRE_SUPPORT, FeatureConfig
from Model.pipeline.features import text_branch
from Model.pipeline.preprocessing import build_feature_frame, build_label_matrix
from Model.pipeline.train import out_of_fold_probabilities, split_indices

ART = '/opt/project/Model/artifacts/'


# --------------------------------------------------------------------------
# Blok fitur padat: embedding pralatih yang sudah dihitung di luar CV.
# Aman dari kebocoran — model encoder-nya beku, tidak pernah di-fit pada
# data kita, jadi tidak ada informasi fold validasi yang bisa merembes.
# --------------------------------------------------------------------------
class DenseCols(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.asarray(X, dtype=np.float32)


def load_frame():
    df = pd.read_parquet(ART + '_research_frame.parquet')
    frame = build_feature_frame(df)
    y, labels, _ = build_label_matrix(df['genre_list'], MIN_GENRE_SUPPORT)
    return df, frame, y, labels


def attach_emb(frame: pd.DataFrame, emb: np.ndarray, prefix: str) -> list[str]:
    cols = [f'{prefix}{i}' for i in range(emb.shape[1])]
    for i, c in enumerate(cols):
        frame[c] = emb[:, i]
    return cols


def build_union(fc: FeatureConfig, tfidf: bool, emb_cols: list[str] | None,
                emb_weight: float):
    """Rangkai ColumnTransformer sesuai blok yang diaktifkan."""
    tr, w = [], {}
    if tfidf:
        tr += [
            ('corpus',   text_branch(fc.corpus_min_df, fc.corpus_max_features, fc), 'corpus_clean'),
            ('synopsis', text_branch(fc.synopsis_min_df, fc.synopsis_max_features, fc), 'synopsis_clean'),
            ('tags',     text_branch(fc.tags_min_df, fc.tags_max_features, fc), 'tags_clean'),
            ('meta',     OneHotEncoder(handle_unknown='ignore', min_frequency=20), ['type_origin', 'type']),
        ]
        w.update({k: v for k, v in fc.weights.items()})
    if emb_cols:
        tr.append(('emb', DenseCols(), emb_cols))
        w['emb'] = emb_weight
    return ColumnTransformer(tr, transformer_weights=w, sparse_threshold=1.0 if tfidf else 0.0)


def head(kind: str, C: float, cw):
    if kind == 'logreg':
        base = LogisticRegression(max_iter=3000, C=C, class_weight=cw)
    else:
        base = CalibratedClassifierCV(
            LinearSVC(C=C, class_weight=cw, max_iter=5000, dual='auto'),
            method='sigmoid', cv=3)
    return OneVsRestClassifier(base, n_jobs=2)


def run(name, frame, y, labels, dev, test, *, tfidf=True, emb_cols=None,
        emb_weight=1.0, kind='logreg', C=4.0, cw='balanced', scale=False):
    """Satu konfigurasi: OOF -> ambang -> uji sekali."""
    steps = [('features', build_union(FeatureConfig(), tfidf, emb_cols, emb_weight))]
    if scale:
        from sklearn.preprocessing import StandardScaler
        steps.append(('scale', StandardScaler(with_mean=not tfidf)))
    steps.append(('clf', head(kind, C, cw)))
    model = Pipeline(steps)

    t0 = time.time()
    xd, yd = frame.iloc[dev], y[dev]
    prob_oof = out_of_fold_probabilities(model, xd, yd)
    th = ev.tune_thresholds(yd, prob_oof, labels)
    oof = ev.evaluate(yd, prob_oof, th)

    model.fit(xd, yd)
    from Model.pipeline.models import predict_proba_frame
    prob_te = predict_proba_frame(model, frame.iloc[test])
    te = ev.evaluate(y[test], prob_te, th)

    print('%-42s OOF makro %.4f | UJI makro %.4f mikro %.4f P@3 %.4f Hamming %.4f  (%.0fs)'
          % (name, oof['f1_macro'], te['f1_macro'], te['f1_micro'],
             te['precision_at_3'], te['hamming_loss'], time.time() - t0), flush=True)
    return {'name': name, 'oof_f1_macro': oof['f1_macro'], **te,
            'per_label': ev.per_label_scores(y[test], prob_te, th, labels)}
