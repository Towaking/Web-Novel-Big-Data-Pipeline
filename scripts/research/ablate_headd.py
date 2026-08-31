# -*- coding: utf-8 -*-
"""Ablasi Head D: apakah `log_days_per_chapter` benar-benar menyumbang?

Head D membaik (R² 0,147 -> 0,167) setelah fitur kecepatan rilis ditambahkan,
tetapi pada saat yang sama pembagian datanya juga berubah. Dua perubahan
sekaligus berarti sebab-akibatnya belum terbukti. Uji ini mengunci semuanya
kecuali satu fitur itu.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/opt/project")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ART = "/opt/project/Model/artifacts/"
RS = 42

cat = pd.read_parquet(ART + "catalog.parquet")
emb = np.load(ART + "embeddings.npy")
M = json.load(open(ART + "metrics.json"))
names = M["numeric_features"]
print("fitur numerik Head D (%d):" % len(names))
print("  ", names)
print("  days_per_chapter dipakai:", "log_days_per_chapter" in names)

# Bangun ulang matriks fitur seperti build_numeric_features
origin = pd.get_dummies(cat["type_origin"].fillna("UNKNOWN"), prefix="origin")
frame = pd.DataFrame({
    "log_chapters": np.log1p(cat["chapter_count"].fillna(cat["chapter_count"].median())),
    "n_tags": cat["n_tags"].fillna(0),
    "year": pd.to_numeric(cat["year"], errors="coerce").fillna(
        pd.to_numeric(cat["year"], errors="coerce").median()),
    "log_days_per_chapter": np.log1p(
        cat["days_per_chapter"].fillna(cat["days_per_chapter"].median())),
})
frame = pd.concat([frame, origin], axis=1)

mask = (cat["rating_votes"].fillna(0) >= 10) & cat["rating_score"].notna()
y = cat.loc[mask, "rating_score"].to_numpy(dtype=np.float32)
print("\npopulasi rating tepercaya: %d" % mask.sum())


def run(cols, label):
    x = np.hstack([emb[mask.to_numpy()], frame.loc[mask, cols].to_numpy(dtype=np.float32)])
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=RS)
    m = HistGradientBoostingRegressor(random_state=RS, max_iter=300,
                                      learning_rate=0.06, max_depth=6).fit(xtr, ytr)
    p = m.predict(xte)
    print("  %-28s MAE %.4f | R2 %.4f" % (label, mean_absolute_error(yte, p), r2_score(yte, p)))
    return mean_absolute_error(yte, p), r2_score(yte, p)


all_cols = list(frame.columns)
without = [c for c in all_cols if c != "log_days_per_chapter"]
print("\nPembagian & data IDENTIK, hanya fitur yang berbeda:")
mae_b, r2_b = run(without, "TANPA kecepatan rilis")
mae_a, r2_a = run(all_cols, "DENGAN kecepatan rilis")
print("\nselisih: MAE %+.4f | R2 %+.4f" % (mae_a - mae_b, r2_a - r2_b))
print("putusan:", "fitur menyumbang" if r2_a > r2_b else "fitur TIDAK menyumbang")
