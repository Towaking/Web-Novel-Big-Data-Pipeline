# Model Card — NovelLens Multi-Head AI

Artefak: `Model/artifacts/` · Dilatih pada 4.460 judul (dari 4.461; 1 baris
tanpa sinyal teks) · Sumber: `s3a://clean-data/novels_clean`

---

## 1. Arsitektur dan alasan pemilihannya

Tiga sifat dataset menentukan pilihan arsitektur:

| Sifat data | Konsekuensi desain |
|---|---|
| Tidak ada `user_id` / histori baca | *Collaborative filtering* (ALS) mustahil → rekomendasi harus berbasis konten |
| Teks campuran prosa + label (sinopsis median 118 kata + 12 tag) | Terlalu pendek untuk fine-tuning transformer; TF-IDF + LSA justru optimal |
| Hanya 4,5k baris | Model besar akan *overfit* dan memperlambat inferensi dari milidetik ke detik |

```
corpus (title + synopsis + tags)
        │
   ┌────┴──────────────────────────────────┐
   │ HEAD A — Genre Tagger (3 pilihan)     │  fitur khusus: stemming +
   │  ├─ Logistic Regression               │  blok TF-IDF terpisah +
   │  ├─ Linear SVM (kalibrasi Platt)      │  metadata + ambang per genre
   │  └─ Neural Network MLP (LSA 240-d)    │
   └───────────────────────────────────────┘
        │
   TF-IDF 1–2 gram → TruncatedSVD 200-d (representasi bersama)
        │
   ┌────┼───────────────┬──────────────────┐
 HEAD B              HEAD C            HEAD D
 Recommender         Arketipe          Estimasi rating
 (cosine kNN)        (KMeans k=9)      (HistGradientBoosting)
```

---

## 2. Head A — pelabelan genre (3 model yang dapat dipilih)

### Protokol evaluasi

- Data dibagi **80% pengembangan (3.568) / 20% uji (892)**.
- Hyperparameter **dan** ambang per genre dipilih memakai probabilitas
  *out-of-fold* dari 3-fold CV **di dalam** set pengembangan.
- Set uji disentuh **sekali**, di akhir. Angka di bawah ini berasal dari sana.
- Ketiga model memakai pembagian baris yang identik → perbandingan adil.

### Hasil (set uji, 892 judul, 24 genre)

| Model | Konfigurasi terpilih | F1 makro | F1 mikro | F1 makro bila ambang 0,5 |
|---|---|---|---|---|
| **Linear SVM** | `C=0.3`, class_weight balanced | **0,5377** | **0,6374** | 0,4104 |
| Logistic Regression | `C=4.0`, tanpa class_weight | 0,5321 | 0,6334 | 0,4501 |
| Neural Network (MLP) | LSA 240-d, lapisan 384→192 | 0,5068 | 0,6183 | 0,4288 |

### Peningkatan dibanding versi awal

Versi awal: satu Logistic Regression, TF-IDF polos atas korpus gabungan,
tanpa stemming, tanpa metadata, ambang seragam 0,5.

| Metrik (set uji) | Sebelum | Sesudah | Selisih |
|---|---|---|---|
| F1 makro | 0,5121 | **0,5377** | **+5,0%** |
| F1 mikro | 0,6346 | **0,6374** | +0,4% |

Yang berubah dan seberapa besar sumbangannya:

1. **Ambang keputusan per genre** — penyumbang terbesar. Pada SVM,
   macro-F1 naik dari 0,4104 (ambang seragam 0,5) menjadi 0,5377, yaitu
   **+31%**. Sebabnya: Romance hadir di 56% katalog sementara Mecha di bawah
   2%; dengan ambang seragam, genre langka nyaris tak pernah diprediksi.
2. **Stemming Snowball** — "cultivation/cultivating/cultivator" menyatu jadi
   satu fitur, bukan tiga fitur lemah.
3. **Blok fitur terpisah berbobot** — tag (label kurasi, presisi tinggi) diberi
   bobot 1,6× dibanding korpus prosa yang lebih berderau.
4. **Metadata sebagai fitur** — asal negara sangat menentukan genre; Xianxia
   praktis hanya ada pada novel Tiongkok.
5. **Pencarian hyperparameter** — 9 kandidat dievaluasi lewat OOF CV.
   Temuan yang tidak terduga: Logistic Regression justru lebih baik **tanpa**
   `class_weight="balanced"` begitu ambang disetel — memakai keduanya adalah
   koreksi ganda terhadap ketimpangan kelas.

### Membaca angka ini secara jujur

F1 mikro nyaris tidak bergerak (+0,4%); yang benar-benar naik adalah F1 makro,
yaitu kemampuan mengenali genre langka. Itu memang target perbaikannya, tetapi
perlu dikatakan apa adanya: **peningkatan keseluruhannya moderat, bukan
lompatan.** Batasnya bersifat intrinsik — pelabelan genre di sumber data
sebagian subjektif (novel yang sama bisa diberi label berbeda oleh kurator
berbeda), sehingga ada langit-langit yang tidak bisa ditembus model mana pun
yang belajar dari label tersebut.

---

## 3. Head B, C, D

| Head | Tugas | Metrik | Nilai |
|---|---|---|---|
| **B** | Rekomendasi berbasis konten | Genre overlap@10 | **0,9188** |
| **C** | Klasterisasi arketipe | Silhouette (k=9) | 0,0185 |
| **D** | Estimasi rating dari konten | MAE | **0,4106** (baseline 0,4434) |
| | | RMSE / R² | 0,5181 / 0,1473 |

- **Head B (overlap 0,92)** — 9 dari 10 rekomendasi berbagi minimal satu genre
  dengan judul acuan. Ini **proksi**, bukan pengganti uji A/B; tanpa data
  perilaku pembaca, relevansi sesungguhnya tidak terukur.
- **Head C (silhouette 0,02)** rendah, dan itu khas data teks berdimensi
  tinggi: klaster saling bersinggungan, bukan terpisah rapi. Yang membuatnya
  tetap berguna adalah koherensi semantiknya (lihat tabel di bawah).
- **Head D (R² 0,15)** — konten hanya menjelaskan ~15% variasi rating. Model
  tetap 7,4% lebih baik daripada menebak rata-rata, tetapi keluarannya
  **indikasi arah**, bukan prediksi presisi. Dashboard menyatakan ini eksplisit.

### Arketipe hasil klasterisasi (k=9)

| # | Ukuran | Istilah dominan |
|---|---|---|
| 0 | 942 | male protagonist, world, day, life |
| 1 | 292 | school, childhood, high school, friends |
| 2 | 628 | female protagonist, marriage, love |
| 3 | 394 | game, game elements, system, level |
| 4 | 214 | european ambience, duke, female protagonist |
| 5 | 620 | family, marriage, ancient |
| 6 | 533 | magic, fantasy world, another world |
| 7 | 437 | love interests, first love, male lead |
| 8 | 400 | cultivation, sect, immortal, martial |

Kelompok-kelompok ini terbaca sebagai arketipe naratif yang dikenali pembaca
(xianxia kultivasi, isekai, LitRPG, romansa keluarga) — bukti kualitatif bahwa
ruang embedding menangkap makna, bukan derau.

---

## 4. Keputusan yang mencegah kebocoran & bias

1. **Fitur popularitas dikeluarkan dari Head D.** `rating_votes` masuk ke dalam
   rumus `weighted_rating`, sehingga memakainya sebagai fitur = membocorkan
   target. `reading_list_count` juga tidak dipakai agar model menilai isi
   cerita, bukan keramaiannya.
2. **Head D hanya dilatih pada rating tepercaya (≥10 suara).** Rating dari 1–2
   suara adalah derau; melatih di atasnya mengajarkan model meniru kebisingan.
3. **Genre dengan <50 novel dibuang** (24 dari 33 genre dipakai). Kelas sekecil
   itu hanya menghasilkan klasifikator yang selalu memprediksi "tidak".
4. **Ambang disetel pada OOF, bukan set uji.** Menyetel lalu melaporkan pada
   data yang sama akan melebih-lebihkan performa.
5. **Daftar stopword Fase 2 disimpan sebagai artefak** dan dibaca ulang oleh
   jalur inferensi, sehingga tidak terjadi *train/serve skew*.
6. **`k` klaster dipilih lewat silhouette**, bukan ditebak.

---

## 5. Batasan yang harus disebutkan saat presentasi

- Model membaca **deskripsi**, bukan naskah. Novel dengan sinopsis buruk akan
  diprediksi buruk meski isinya bagus.
- Rekomendasi murni berbasis konten: dua judul bertema sama dianggap mirip
  walau mutu penulisannya jauh berbeda.
- Katalog didominasi novel Tiongkok (2.455 dari 4.461); performa untuk judul
  Barat/EN kemungkinan lebih lemah karena representasinya sedikit.
- Rating berasal dari satu platform (NovelUpdates) dengan basis pembaca
  terjemahan — bukan populasi pembaca umum.
- Arketipe adalah kelompok yang bersinggungan, bukan sekat tegas.

---

## 6. Artefak yang dihasilkan

| File | Isi | Ukuran |
|---|---|---|
| `genre_models.joblib` | 3 pipeline genre lengkap + ambang per genre | 65,0 MB |
| `svd.joblib` | Proyeksi LSA 200 dimensi | 35,7 MB |
| `vectorizer.joblib` | TF-IDF representasi bersama | 0,27 MB |
| `genre_binarizer.joblib` | Pemetaan 24 label | <0,01 MB |
| `kmeans.joblib` | KMeans k=9 | 0,01 MB |
| `rating_regressor.joblib` | HistGradientBoosting | 0,28 MB |
| `embeddings.npy` | Matriks embedding 4460×200 (float32) | 3,4 MB |
| `catalog.parquet` | Metadata + cuplikan sinopsis + cluster_id | 1,8 MB |
| `metrics.json` | Seluruh metrik, profil klaster, stopword | 0,02 MB |

Semua artefak juga disalin ke `s3a://curated-data/models/` sebagai cadangan
yang dapat diaudit.

---

## 7. Cara melatih ulang

```bash
docker compose exec pipeline python -m src.phases.phase4_model
```

Waktu training penuh ≈ 3 menit (tanpa GPU), termasuk pencarian 9 konfigurasi
hyperparameter lewat validasi silang. Dashboard memuat artefak baru setelah
container Streamlit di-restart.
