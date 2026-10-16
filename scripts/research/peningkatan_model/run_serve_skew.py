"""Uji skew saat inferensi: pengguna mengetik kata utuh (tanpa sensor).

Model lama dilatih pada teks bersensor; di sini set uji diberi versi tanpa
sensor (seperti input pengguna), lalu dibandingkan dengan model yang dilatih
pada teks yang sudah dipulihkan.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder

import exp_lib as L
from Model.pipeline import evaluate as ev
from Model.pipeline.preprocessing import stem_texts
from textprep_exp import rebuild

pdf, stop, y, labels, dev_idx, test_idx, folds = L.setup()
cen = rebuild(pdf, stop, False)
unc = rebuild(pdf, stop, True)
affected = pdf["synopsis"].fillna("").str.contains(r"\*", regex=True) | pdf["tags"].fillna("").str.contains(r"\*", regex=True)
aff_test = affected.to_numpy()[test_idx]
print("dokumen uji terdampak sensor:", int(aff_test.sum()), "dari", len(test_idx))

cfg = {"corpus_clean": (3, 40_000, 1.0), "synopsis_clean": (3, 40_000, 1.0), "tags_clean": (2, 10_000, 1.6)}


def fit_eval(train_txt, test_txt):
    xs_tr, xs_te = [], []
    for col, (mdf, mf, w) in cfg.items():
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=mdf, max_df=0.85, max_features=mf,
                              sublinear_tf=True, dtype=np.float32)
        xs_tr.append(vec.fit_transform(stem_texts(train_txt[col].iloc[dev_idx])) * w)
        xs_te.append(vec.transform(stem_texts(test_txt[col].iloc[test_idx])) * w)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    meta = pdf[["type_origin", "type"]].fillna("UNKNOWN")
    xs_tr.append(enc.fit_transform(meta.iloc[dev_idx]) * 0.6)
    xs_te.append(enc.transform(meta.iloc[test_idx]) * 0.6)
    clf = OneVsRestClassifier(CalibratedClassifierCV(
        LinearSVC(C=0.3, class_weight="balanced", max_iter=5000, dual="auto", random_state=42),
        method="sigmoid", cv=3), n_jobs=-1)
    clf.fit(sparse.hstack(xs_tr).tocsr(), y[dev_idx])
    return L.proba_of(clf, sparse.hstack(xs_te).tocsr())


oof_cen, _ = L.load_proba("rb_fix0_svm")
oof_unc, _ = L.load_proba("rb_fix1_svm")
thr_cen = ev.tune_thresholds(y[dev_idx], oof_cen[dev_idx], labels)
thr_unc = ev.tune_thresholds(y[dev_idx], oof_unc[dev_idx], labels)

p_old_on_user = fit_eval(cen, unc)   # model lama, input pengguna tanpa sensor
p_new_on_user = fit_eval(unc, unc)   # model baru, input pengguna tanpa sensor
yt = y[test_idx]
for name, p, thr in [("lama (latih bersensor)", p_old_on_user, thr_cen), ("baru (latih dipulihkan)", p_new_on_user, thr_unc)]:
    all_m = ev.evaluate(yt, p, thr)
    sub = ev.evaluate(yt[aff_test], p[aff_test], thr)
    print(f"{name}: semua uji macro {all_m['f1_macro']} micro {all_m['f1_micro']} | terdampak macro {sub['f1_macro']} micro {sub['f1_micro']} P@3 {sub['precision_at_3']}")
    per = f1_score(yt[aff_test], (p[aff_test] >= thr).astype(int), average=None, zero_division=0)
    print("   ", {l: round(float(s), 3) for l, s in zip(labels, per) if l in ("Adult", "Mature", "Smut", "Ecchi", "Tragedy", "Psychological", "Horror", "Mystery", "Drama", "Yuri", "Yaoi")})
