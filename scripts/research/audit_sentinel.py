# -*- coding: utf-8 -*-
"""
Audit sentinel: apakah nilai yang "hilang" sebenarnya masih ada di kolom lain?
=============================================================================
Fase 2 mengubah sentinel (`chapter_count == 0`, `year == 0`) menjadi null dan
berhenti di situ. Tapi dua kolom yang saat ini TIDAK dipakai analitik masih
menyimpan informasi yang sama dalam bentuk teks:

    status_coo         "208 Chapters (Ongoing)"   <- memuat jumlah bab
    release_frequency  "Every 0.9 Day(s)"         <- kecepatan rilis

Kalau benar, 902 baris ber-`chapter_count = 0` tidak semuanya benar-benar tak
diketahui — sebagiannya bisa dipulihkan, dan itu mengubah setiap agregat
panjang cerita di Fase 3.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

df = pd.read_csv("/opt/project/Dataset/data_novel.csv")
print("n =", len(df))

# ---------------------------------------------------------------------------
# 1. status_coo -> jumlah bab
# ---------------------------------------------------------------------------
NUM = r"(\d[\d,]*)"
pat_ch = re.compile(NUM + r"\s*(?:chapters?|chs?)\b", re.I)
pat_vol = re.compile(NUM + r"\s*(?:volumes?|vols?)\b", re.I)


def parse_ch(s):
    m = pat_ch.search(str(s))
    return int(m.group(1).replace(",", "")) if m else np.nan


def parse_vol(s):
    m = pat_vol.search(str(s))
    return int(m.group(1).replace(",", "")) if m else np.nan


df["ch_from_status"] = df["status_coo"].map(parse_ch)
df["vol_from_status"] = df["status_coo"].map(parse_vol)

zero = df["chapter_count"] == 0
print("\n=== 1. chapter_count == 0 ===")
print("baris sentinel                    : %d (%.2f%%)" % (zero.sum(), 100 * zero.mean()))
print("  status_coo memuat jumlah bab    : %d  <- DAPAT DIPULIHKAN"
      % df.loc[zero, "ch_from_status"].notna().sum())
print("  status_coo memuat jumlah volume : %d" % df.loc[zero, "vol_from_status"].notna().sum())
rest = zero & df["ch_from_status"].isna() & df["vol_from_status"].isna()
print("  benar-benar tak diketahui       : %d" % rest.sum())
print("\ncontoh nilai status_coo pada baris sentinel yang bisa dipulihkan:")
sample = df.loc[zero & df["ch_from_status"].notna(), ["title", "status_coo", "ch_from_status"]].head(6)
for _, r in sample.iterrows():
    print("   %-40s %-28s -> %d" % (str(r["title"])[:40], str(r["status_coo"])[:28], r["ch_from_status"]))

# Validasi aturan pada baris yang chapter_count-nya TIDAK nol:
# kalau parser benar, angkanya harus cocok dengan kolom aslinya.
nz = (~zero) & df["ch_from_status"].notna()
agree = (df.loc[nz, "ch_from_status"] == df.loc[nz, "chapter_count"]).mean()
print("\nvalidasi parser pada %d baris ber-chapter_count>0: cocok %.1f%%" % (nz.sum(), 100 * agree))
diff = df.loc[nz & (df["ch_from_status"] != df["chapter_count"]),
              ["chapter_count", "status_coo", "ch_from_status"]].head(5)
if len(diff):
    print("contoh yang tidak cocok:")
    for _, r in diff.iterrows():
        print("   kolom=%-7s status=%-30s parse=%s"
              % (r["chapter_count"], str(r["status_coo"])[:30], r["ch_from_status"]))

# Dampak ke agregat
old = df.loc[~zero, "chapter_count"]
recovered = df["chapter_count"].where(~zero, df["ch_from_status"])
new = recovered.dropna()
print("\ndampak ke statistik panjang cerita:")
print("  sebelum : n=%d  mean=%.1f  median=%.1f" % (len(old), old.mean(), old.median()))
print("  sesudah : n=%d  mean=%.1f  median=%.1f" % (len(new), new.mean(), new.median()))
print("  selisih : +%d baris (%.1f%% lebih banyak), mean %+.1f bab"
      % (len(new) - len(old), 100 * (len(new) - len(old)) / len(old), new.mean() - old.mean()))

# ---------------------------------------------------------------------------
# 2. release_frequency -> hari per bab (fitur numerik, saat ini tak dipakai)
# ---------------------------------------------------------------------------
pat_freq = re.compile(r"every\s+([\d.]+)\s+day", re.I)


def parse_freq(s):
    m = pat_freq.search(str(s))
    return float(m.group(1)) if m else np.nan


df["days_per_chapter"] = df["release_frequency"].map(parse_freq)
print("\n=== 2. release_frequency ===")
print("nilai unik                : %d" % df["release_frequency"].nunique())
print("dapat diurai jadi angka   : %d (%.1f%%)"
      % (df["days_per_chapter"].notna().sum(), 100 * df["days_per_chapter"].notna().mean()))
print("tidak terurai, contoh     :", df.loc[df.days_per_chapter.isna(), "release_frequency"].value_counts().head(4).to_dict())
d = df["days_per_chapter"].dropna()
print("sebaran hari/bab          : median %.2f  p10 %.2f  p90 %.2f  max %.1f"
      % (d.median(), d.quantile(.1), d.quantile(.9), d.max()))

# Apakah kecepatan rilis berhubungan dengan rating? (populasi tervalidasi saja)
ok = df[(df.rating_votes >= 10) & df.days_per_chapter.notna()]
r = np.corrcoef(np.log1p(ok.days_per_chapter), ok.rating_score)[0, 1]
print("korelasi log(hari/bab) ~ rating (n=%d): r = %+.4f" % (len(ok), r))

# ---------------------------------------------------------------------------
# 3. year == 0
# ---------------------------------------------------------------------------
print("\n=== 3. year == 0 ===")
zy = df["year"] == 0
print("baris sentinel : %d (%.2f%%)" % (zy.sum(), 100 * zy.mean()))
print("apakah tahun muncul di kolom lain? cek status_coo & release_frequency:")
yr = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
found = df.loc[zy, "status_coo"].astype(str).str.contains(yr).sum()
print("  status_coo memuat pola tahun : %d" % found)
print("  -> tidak ada sumber pemulihan; sentinel ini memang tak terpulihkan")
