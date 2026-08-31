# -*- coding: utf-8 -*-
"""Verifikasi jalur inferensi produksi untuk keempat model genre.

Memakai `load_registry()` — jalur yang sama persis dipakai dashboard — supaya
yang diuji adalah artefak yang benar-benar tersimpan, bukan model in-memory
hasil training.
"""
import sys, time
sys.path.insert(0, "/opt/project")

from src.common.model_registry import load_registry

reg = load_registry()
print("model genre tersedia :", reg.genre_model_names)
print("model terbaik        :", reg.metrics["head_a_genre_tagger"]["best_model"])
print("kolom katalog        :", "days_per_chapter" in reg.catalog.columns)

SYN = ("A young swordsman is betrayed by his sect and left for dead, but wakes "
       "ten years in the past. Armed with knowledge of the future he begins "
       "cultivating a forbidden technique to take revenge on the elders.")
TAGS = "Time Travel, Cultivation, Revenge, Male Protagonist, Weak to Strong"

for name in reg.genre_model_names:
    t0 = time.time()
    out = reg.predict_genres(SYN, tags=TAGS, model=name, top_k=6)
    ms = 1000 * (time.time() - t0)
    hit = out[out["predicted"]]["genre"].tolist()
    print("\n[%s] %.0f ms" % (name, ms))
    print("   diprediksi :", hit or "(tidak ada)")
    print("   top-3      :", [(r.genre, round(r.probability, 3)) for r in out.head(3).itertuples()])

print("\n--- Head D dengan fitur kecepatan rilis ---")
for pace in (2.0, 30.0, 300.0):
    r = reg.estimate_rating(TAGS, chapter_count=400, year=2023,
                            type_origin="CN", days_per_chapter=pace)
    print("   %6.1f hari/bab -> rating %.3f" % (pace, r))

print("\n--- Head B & C ---")
rec = reg.recommend_by_text(SYN, k=3)
print("   rekomendasi:", rec["title"].tolist())
cl = reg.assign_cluster(TAGS)
print("   arketipe   : #%d %s" % (cl["cluster_id"], cl["profile"]["top_terms"][:4]))
print("\nVERIFIKASI INFERENSI LULUS")
