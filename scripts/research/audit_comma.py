# -*- coding: utf-8 -*-
"""Cari baris yang `chapter_count`-nya terpotong koma ribuan.

`status_coo` menuliskan jumlah bab dengan pemisah ribuan ("1,593 Chapters").
Bila pipeline pengambilan data pernah membaca angka itu setelah dipotong pada
koma, nilainya menyusut satu orde. Kolom `status_coo` masih menyimpan angka
utuhnya, jadi kesalahan seperti ini bisa dibuktikan, bukan diduga.
"""
import re
import numpy as np
import pandas as pd

df = pd.read_csv("/opt/project/Dataset/data_novel.csv")
pat = re.compile(r"(\d[\d,]*)\s*(?:chapters?|chs?)\b", re.I)


def parse(s):
    m = pat.search(str(s))
    return int(m.group(1).replace(",", "")) if m else np.nan


df["parsed"] = df["status_coo"].map(parse)
both = df[df.parsed.notna() & (df.chapter_count > 0)]
bad = both[both.parsed != both.chapter_count]

print("baris dengan status_coo terurai & chapter_count>0 : %d" % len(both))
print("tidak cocok                                       : %d" % len(bad))
print()
for _, r in bad.iterrows():
    tail = str(r["parsed"])[-len(str(r["chapter_count"])):]
    flag = "  <- terpotong koma" if tail == str(int(r["chapter_count"])) else ""
    print("  %-46s kolom=%-8d status_coo=%-30s utuh=%d%s"
          % (str(r["title"])[:46], r["chapter_count"], str(r["status_coo"])[:30],
             r["parsed"], flag))

print("\n--- kenapa 538 baris ber-chapter_count>0 tidak punya angka di status_coo ---")
no_parse = df[(df.chapter_count > 0) & df.parsed.isna()]
print("jumlah:", len(no_parse))
print(no_parse["status_coo"].value_counts().head(8).to_string())
