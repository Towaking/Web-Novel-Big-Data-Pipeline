# -*- coding: utf-8 -*-
"""
Uji berpasangan: apakah `hybrid` benar-benar unggul, atau cuma beruntung?
=========================================================================
Satu set uji berisi 892 baris dan 24 genre. Mengganti pembagian saja sudah
menggeser macro-F1 sekitar ±0,02 — lebih besar daripada selisih antar-model
yang diukur (+0,013). Karena itu perbandingan pada satu pembagian saja tidak
cukup untuk menyimpulkan apa pun.

Solusinya perbandingan **berpasangan**: pada setiap seed, kedua model dilatih
dan diuji pada pembagian yang sama persis, lalu yang dianalisis adalah
SELISIHNYA. Ragam yang berasal dari pembagian muncul di kedua model sekaligus
sehingga saling meniadakan — persis alasan uji berpasangan dipakai.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/opt/project")
sys.path.insert(0, "/opt/project/scripts/research")

import numpy as np
from sklearn.model_selection import train_test_split

from exp_repr import ART, load_frame, attach_emb, run

SEEDS = [42, 7, 123, 2024, 99]

df, frame, y, labels = load_frame()
emb = np.load(ART + "_emb_bge-small-en-v15_full.npy")
ecols = attach_emb(frame, emb, "e")
frame = frame.copy()

rows = []
for seed in SEEDS:
    dev, test = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=seed)
    print("\n--- seed %d ---" % seed, flush=True)
    base = run(f"[{seed}] baseline svm", frame, y, labels, dev, test,
               tfidf=True, kind="svm", C=0.3, cw="balanced")
    hyb = run(f"[{seed}] hybrid  svm", frame, y, labels, dev, test,
              tfidf=True, emb_cols=ecols, emb_weight=2.0, kind="svm", C=0.3, cw="balanced")
    rows.append({"seed": seed,
                 "base_oof": base["oof_f1_macro"], "hyb_oof": hyb["oof_f1_macro"],
                 "base_test": base["f1_macro"], "hyb_test": hyb["f1_macro"],
                 "base_micro": base["f1_micro"], "hyb_micro": hyb["f1_micro"],
                 "base_p3": base["precision_at_3"], "hyb_p3": hyb["precision_at_3"]})

json.dump(rows, open(ART + "_paired.json", "w"), indent=1)

print("\n" + "=" * 74)
print("%-6s %-17s %-17s %-17s" % ("SEED", "MAKRO uji", "MIKRO uji", "P@3 uji"))
print("%-6s %-17s %-17s %-17s" % ("", "base -> hybrid", "base -> hybrid", "base -> hybrid"))
print("-" * 74)
for r in rows:
    print("%-6d %.4f -> %.4f  %.4f -> %.4f  %.4f -> %.4f"
          % (r["seed"], r["base_test"], r["hyb_test"],
             r["base_micro"], r["hyb_micro"], r["base_p3"], r["hyb_p3"]))


def summarize(name, base_key, hyb_key):
    b = np.array([r[base_key] for r in rows])
    h = np.array([r[hyb_key] for r in rows])
    d = h - b
    # Uji-t berpasangan sederhana; n=5 jadi ini indikasi, bukan bukti kuat.
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float("inf")
    print("%-12s base %.4f ± %.4f | hybrid %.4f ± %.4f | selisih %+.4f ± %.4f | menang %d/%d | t=%.2f"
          % (name, b.mean(), b.std(ddof=1), h.mean(), h.std(ddof=1),
             d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d), t))


print("-" * 74)
summarize("OOF makro", "base_oof", "hyb_oof")
summarize("UJI makro", "base_test", "hyb_test")
summarize("UJI mikro", "base_micro", "hyb_micro")
summarize("UJI P@3", "base_p3", "hyb_p3")
