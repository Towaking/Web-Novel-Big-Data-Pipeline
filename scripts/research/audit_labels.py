# -*- coding: utf-8 -*-
"""
Audit kualitas label genre
==========================
Model card menyatakan batas atas tugas ini ~0,653 makro-F1 dan menyebut
penyebabnya "label tidak konsisten". Klaim itu diukur lewat kesepakatan antar
novel bertetangga — proksi yang benar tetapi tidak memberi tahu BENTUK
kesalahannya: apakah label yang ada itu salah, atau label yang benar tidak
dituliskan?

Bedanya penting. Kalau dominan **label hilang** (missing label), maka setiap
"positif palsu" model sebetulnya sebagian benar, dan presisi yang dilaporkan
terlalu rendah secara sistematis.

Cara mengukurnya di sini: `tags` adalah kurasi manusia yang terpisah dari
`genres` dan jauh lebih rinci (759 tag unik vs 33 genre). Kalau sebuah tag
secara empiris hampir selalu berarti sebuah genre, lalu ada novel bertag itu
yang genrenya kosong, itu kandidat label hilang — bukan tebakan model yang keliru.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/opt/project")

import numpy as np
import pandas as pd

from Model.pipeline.config import MIN_GENRE_SUPPORT
from Model.pipeline.preprocessing import build_label_matrix

ART = "/opt/project/Model/artifacts/"
df = pd.read_parquet(ART + "_research_frame.parquet")
Y, LABELS, _ = build_label_matrix(df["genre_list"], MIN_GENRE_SUPPORT)
L = {g: i for i, g in enumerate(LABELS)}
print("n=%d | genre=%d" % (len(df), len(LABELS)))


def tagset(s):
    if not isinstance(s, str):
        return set()
    return {t.strip() for t in s.split(",") if t.strip() and t.strip() != "Unknown"}


tags = df["tags"].map(tagset)
all_tags = pd.Series([t for s in tags for t in s]).value_counts()
common = all_tags[all_tags >= 30].index.tolist()
print("tag dipakai (>=30 novel): %d dari %d\n" % (len(common), len(all_tags)))

# ---------------------------------------------------------------------------
# 1. Aturan implikasi tag -> genre
#    presisi = P(genre | tag). Lift = presisi / prevalensi genre.
#    Aturan disimpan hanya bila SANGAT kuat: presisi tinggi DAN lift tinggi,
#    supaya tidak sekadar menangkap genre yang memang umum (Romance 56%).
# ---------------------------------------------------------------------------
has = {t: tags.map(lambda s, t=t: t in s).values for t in common}
prev = Y.mean(axis=0)

rules = []
for t in common:
    m = has[t]
    n = int(m.sum())
    for g, gi in L.items():
        p = float(Y[m, gi].mean())
        if p >= 0.80 and prev[gi] > 0 and p / prev[gi] >= 1.8:
            rules.append({"tag": t, "genre": g, "precision": round(p, 3),
                          "lift": round(p / prev[gi], 2), "n_tag": n,
                          "n_missing": int(((m) & (Y[:, gi] == 0)).sum())})

rules = sorted(rules, key=lambda r: -r["n_missing"])
print("=== 1. Aturan tag->genre yang sangat kuat (presisi>=0,80 & lift>=1,8) ===")
print("%d aturan ditemukan. 18 teratas menurut jumlah label yang hilang:\n" % len(rules))
print("  %-32s %-14s %6s %6s %7s %9s" % ("TAG", "GENRE", "P", "LIFT", "n_tag", "HILANG"))
for r in rules[:18]:
    print("  %-32s %-14s %6.2f %6.2f %7d %9d"
          % (r["tag"][:32], r["genre"][:14], r["precision"], r["lift"], r["n_tag"], r["n_missing"]))

# ---------------------------------------------------------------------------
# 2. Berapa banyak label yang hilang secara keseluruhan?
#    Satu novel dihitung "kehilangan genre G" bila ia punya >=2 tag berbeda
#    yang masing-masing mengimplikasikan G. Syarat dua tag dipakai supaya satu
#    tag menyimpang tidak langsung dianggap bukti.
# ---------------------------------------------------------------------------
votes = np.zeros_like(Y, dtype=np.int16)
for r in rules:
    votes[has[r["tag"]], L[r["genre"]]] += 1

missing = (votes >= 2) & (Y == 0)
print("\n=== 2. Perkiraan label hilang (>=2 tag independen sepakat) ===")
print("sel label total          : %d" % Y.size)
print("positif tercatat         : %d  (kardinalitas %.2f/novel)" % (Y.sum(), Y.sum() / len(Y)))
print("positif HILANG (dugaan)  : %d  (+%.1f%% dari yang tercatat)"
      % (missing.sum(), 100 * missing.sum() / Y.sum()))
print("novel terdampak          : %d dari %d (%.1f%%)"
      % ((missing.sum(1) > 0).sum(), len(Y), 100 * (missing.sum(1) > 0).mean()))

per_g = pd.DataFrame({
    "genre": LABELS,
    "tercatat": Y.sum(0),
    "hilang": missing.sum(0),
}).assign(rasio=lambda d: (d.hilang / d.tercatat).round(3)).sort_values("hilang", ascending=False)
print("\ngenre dengan label paling banyak hilang:")
print(per_g.head(12).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Apakah "positif palsu" model sebetulnya label hilang?
#    Memakai probabilitas OOF dari model baseline pada set pengembangan.
# ---------------------------------------------------------------------------
print("\n=== 3. Positif palsu model vs bukti tag ===")
try:
    oof = np.load(ART + "_oof_baseline.npy")
    dev = np.load(ART + "_dev_idx.npy")
    th = np.load(ART + "_th_baseline.npy")
    pred = (oof >= th).astype(int)
    ydev, vdev = Y[dev], votes[dev]
    fp = (pred == 1) & (ydev == 0)
    tp = (pred == 1) & (ydev == 1)
    print("positif palsu (FP)       : %d" % fp.sum())
    print("  di antaranya didukung >=1 tag implikatif : %d (%.1f%%)"
          % ((fp & (vdev >= 1)).sum(), 100 * (fp & (vdev >= 1)).sum() / max(fp.sum(), 1)))
    print("  didukung >=2 tag implikatif              : %d (%.1f%%)"
          % ((fp & (vdev >= 2)).sum(), 100 * (fp & (vdev >= 2)).sum() / max(fp.sum(), 1)))
    print("pembanding — sel negatif yang TIDAK diprediksi model:")
    tn = (pred == 0) & (ydev == 0)
    print("  didukung >=2 tag implikatif              : %d (%.1f%%)"
          % ((tn & (vdev >= 2)).sum(), 100 * (tn & (vdev >= 2)).sum() / max(tn.sum(), 1)))
    print("\nBila FP jauh lebih sering didukung tag daripada sel negatif biasa,")
    print("maka sebagian 'kesalahan' model sebenarnya adalah label yang hilang.")
except FileNotFoundError:
    print("(lewati — jalankan run_exp2.py dulu untuk menyimpan probabilitas OOF)")

json.dump({"rules": rules,
           "missing_total": int(missing.sum()),
           "recorded_total": int(Y.sum()),
           "per_genre": per_g.to_dict("records")},
          open(ART + "_label_audit.json", "w"), indent=1)
np.save(ART + "_missing_mask.npy", missing)
print("\ntersimpan: _label_audit.json, _missing_mask.npy")
