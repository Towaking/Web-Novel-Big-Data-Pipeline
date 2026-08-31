# -*- coding: utf-8 -*-
"""Render tabel hasil Head A / B / C / D langsung dari metrics.json.

Dipakai untuk menyusun tabel di README dan model card. Angkanya diambil dari
artefak, bukan disalin tangan, sehingga dokumen tidak bisa melenceng dari
model yang benar-benar tersimpan.
"""
import json

M = json.load(open("/opt/project/Model/artifacts/metrics.json"))
a = M["head_a_genre_tagger"]

print("dilatih         :", M.get("generated_at_utc"))
print("dokumen         :", M.get("n_documents"))
print("dev / uji       : %s / %s" % (a.get("n_train"), a.get("n_test")))
print("model terbaik   :", a.get("best_model"))
print()

hdr = "| Model | Konfigurasi | OOF makro | F1 makro | F1 mikro | P@3 | Hamming ↓ |"
print(hdr)
print("|---|---|---|---|---|---|---|")
rows = sorted(a["models"].items(), key=lambda kv: -kv[1]["f1_macro"])
for name, e in rows:
    params = ", ".join(f"{k}={v}" for k, v in e.get("params", {}).items())
    star = "**" if name == a.get("best_model") else ""
    print("| %s%s%s | `%s` | %.4f | %s%.4f%s | %.4f | %.4f | %.4f |"
          % (star, e.get("label", name), star, params, e.get("f1_macro_oof", 0),
             star, e["f1_macro"], star, e["f1_micro"],
             e.get("precision_at_3", 0), e.get("hamming_loss", 0)))

print("\n--- F1 per genre, model terbaik vs Linear SVM ---")
best = a["models"][a["best_model"]]["f1_per_label"]
svm = a["models"].get("svm", {}).get("f1_per_label", {})
print("| Genre | %s | Linear SVM | Selisih |" % a["models"][a["best_model"]].get("label"))
print("|---|---|---|---|")
for g in sorted(best, key=lambda g: -(best[g] - svm.get(g, 0))):
    d = best[g] - svm.get(g, 0)
    print("| %s | %.3f | %.3f | %+.3f |" % (g, best[g], svm.get(g, 0), d))

print("\n--- Head B / C / D ---")
for key in ("head_b_recommender", "head_c_clustering", "head_d_rating_estimator"):
    print(key, "->", {k: v for k, v in M[key].items()
                      if not isinstance(v, (list, dict))})
print("\nfitur numerik Head D:", M.get("numeric_features"))
