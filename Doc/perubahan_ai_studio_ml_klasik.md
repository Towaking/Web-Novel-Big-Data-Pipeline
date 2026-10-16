# Perubahan: AI Studio memakai ML klasik saja, dan notebook sklearn vs MLlib

Dokumen ini mencatat dua perubahan yang dikerjakan pada 2026-10-11, apa isinya, bagaimana diperiksa, dan apa yang belum selesai.

> **Pembaruan 2026-10-15.** Model bawaan AI Studio kini ensemble Linear SVM + Random Forest (probabilitas digabung di ruang log-odds), dan Random Forest tersedia sebagai pilihan keempat. Angka di bawah adalah kondisi 2026-10-11; angka terbaru dan metodenya ada di [peningkatan_model.md](peningkatan_model.md).

## 1. AI Studio: hanya model machine learning klasik

### Sebelum dan sesudah

Tab "Analyse a story" di AI Studio memberi pilihan model genre dari `Model/artifacts/genre_models.joblib`.

| Model | Jenis | Sebelum | Sesudah |
|---|---|---|---|
| Logistic Regression | ML klasik, TF-IDF | ada | ada |
| Linear SVM | ML klasik, TF-IDF | ada | ada, jadi default |
| Hybrid (TF-IDF + BGE) | encoder neural pralatih | ada, default | dihapus |
| BiLSTM + Attention | deep learning (PyTorch) | ada | dihapus |

"ML klasik" di sini berarti tanpa jaringan saraf, tanpa embedding pralatih, dan tanpa PyTorch.

Angka dari `metrics.json` (35 genre, 7.932 novel uji):

| Model | Macro-F1 cross-validation | Macro-F1 uji |
|---|---|---|
| Hybrid (dihapus) | 0,5912 | 0,5971 |
| Linear SVM (default baru) | 0,5834 | 0,5913 |
| Logistic Regression | 0,5756 | 0,5840 |
| BiLSTM (dihapus) | 0,5417 | 0,5504 |

Default berpindah dari Hybrid ke SVM. Selisih macro-F1 uji sekitar 0,006. Ini kecil dibanding batas atas empiris tugas ini di README (sekitar 0,65), yang berasal dari label yang subjektif.

Bagian lain AI Studio sudah klasik sejak awal: vektor teks memakai TF-IDF lalu SVD, rekomendasi memakai cosine pada vektor itu, archetype memakai KMeans, dan estimasi rating memakai gradient boosting. Hanya pemilih model genre yang menyentuh model neural.

### Berkas yang berubah

| Berkas | Perubahan |
|---|---|
| `Model/artifacts/genre_models.joblib` | Hanya berisi `logreg` dan `svm`. |
| `Model/artifacts/metrics.json` | `head_a_genre_tagger.models` hanya berisi `logreg` dan `svm`. `best_model` menjadi `svm` (dipilih dari skor cross-validation). Hasil Hybrid dan BiLSTM disimpan di kunci baru `excluded_models` supaya angkanya tidak hilang dari laporan. |
| `app/streamlit_app.py` | Teks `hybrid` dan `bilstm` dihapus dari `MODEL_TEXT`. Komentar tentang pemuatan encoder dihapus. Catatan di halaman Methodology diganti: menjelaskan bahwa model neural pernah dilatih dan sengaja tidak dipakai di aplikasi. |

Cadangan artefak lama ada di `Model/artifacts/_backup_pre_classical/` (`genre_models.joblib` dan `metrics.json`). Artefak model di-gitignore, jadi folder ini satu-satunya salinan.

### Cara memeriksa

- Registry dimuat ulang dan daftar modelnya `['svm', 'logreg']`.
- Setelah registry dimuat, paket `torch`, `sentence_transformers`, dan `transformers` tidak terimpor.
- Prediksi contoh sinopsis (reinkarnasi dan kultivasi) berjalan di kedua model dan menghasilkan genre yang masuk akal (Xianxia, Fantasy, Xuanhuan, Romance, Action).
- Container `streamlit` dijalankan ulang, lalu `scripts/verify_app.py` melaporkan 15 dari 15 halaman LULUS, termasuk AI Studio dan Methodology.

Batas pemeriksaan: smoke test hanya merender halaman dengan isian bawaan dan tidak menekan tombol "Analyse story". Jalur prediksi diperiksa terpisah lewat pemanggilan `predict_genres` di atas. Tampilan di browser belum saya lihat langsung.

### Yang sengaja tidak diubah

- `src/common/model_registry.py`, `src/phases/phase4_model.py`, `Model/train.py`: kodenya tidak diubah.
- `README.md` dan `Doc/model_card.md` masih menyebut empat model genre.
- `docker/Dockerfile` masih memasang torch dan bobot BGE. Image bisa diperkecil nanti, tapi itu perubahan terpisah.

Konsekuensi penting: melatih ulang Fase 4 tanpa batasan akan membuat Hybrid dan BiLSTM muncul lagi. Untuk menjaga hanya model klasik, jalankan:

```bash
docker compose exec -e PHASE4_MODELS=logreg,svm pipeline python -m src.phases.phase4_model
```

### Cara mengembalikan

1. Salin `genre_models.joblib` dan `metrics.json` dari `Model/artifacts/_backup_pre_classical/` ke `Model/artifacts/`.
2. Kembalikan `app/streamlit_app.py` dengan `git checkout app/streamlit_app.py` (atau `git diff` dulu untuk melihat perubahannya).
3. `docker compose restart streamlit`.

## 2. Notebook baru: scikit-learn vs Spark MLlib

### Lokasi dan keputusan

`scripts/notebook/sklearn_vs_mllib.ipynb`, 45 sel, satu file yang dirancang untuk Run All.

Notebook ini dibuat sebagai file terpisah, bukan menimpa `model_training.ipynb`. Alasannya: `model_training.ipynb` melatih ulang Fase 4 (hybrid sekitar 1 sampai 1,5 jam, BiLSTM sekitar 5 jam menurut tabel di notebook itu), jadi menyatukannya akan membuat Run All ikut menjalankan itu. `model_training.ipynb` tidak disentuh.

### Isinya

Tugas: menebak `type_origin` (CN, JP, KR) dari korpus teks, genre, tipe novel, dan beberapa kolom angka, pada 38.983 novel. Kata penunjuk negara (china, korean, japan, dan sejenisnya) dibuang dari teks supaya model tidak sekadar mencocokkan kata.

Alurnya mengikuti permintaan:

1. Setup dan dataset
2. EDA singkat (EDA lengkap tetap di `eda_visualisasi.ipynb`)
3. Konfigurasi: split 80/20 berstrata, 3 fold yang sama untuk kedua kerangka, preprocessing yang dibuat setara
4. Feature selection ANOVA F: perbandingan fitur terpilih antar-kerangka, kurva jumlah fitur, pemilihan `k`
5. Model A, B, C (scikit-learn) dan Evaluasi 1
6. Model D, E, F (MLlib) dan Evaluasi 2
7. Komparasi Evaluasi 1 vs 2
8. Tuning A, B, C dan Evaluasi 3
9. Tuning D, E, F dan Evaluasi 4
10. Komparasi Evaluasi 3 vs 4
11. Komparasi keseluruhan: performa, kecepatan per tahap, eksperimen skala data, matriks kebingungan, ringkasan otomatis

Pasangan model: Logistic Regression (A dan D), Naive Bayes (B dan E), Random Forest (C dan F).

Hasil disimpan ke `Doc/generated/sklearn_vs_mllib.json`. Variabel lingkungan `NB_SMOKE=1` menjalankan versi kecil (sekitar 6 ribu baris) dan menulis ke `sklearn_vs_mllib_smoke.json`.

### Status verifikasi

Jujur soal apa yang sudah dan belum terbukti:

- Semua sel kode lolos pemeriksaan sintaks, dan teks bebas dari karakter em dash.
- Satu run mode uji cepat sudah berjalan dan menghasilkan `.tmp_probe/smoke_out.ipynb`. Saya belum membuka hasilnya, jadi belum tahu apakah semua sel lolos atau ada yang error.
- Run penuh (seluruh data) belum dijalankan, jadi belum ada angka hasil dan belum ada perkiraan waktu yang bisa dipercaya. Run uji cepat sendiri sudah memakan belasan menit, sebagian besar di Random Forest MLlib yang menjalankan ratusan job Spark kecil, jadi run penuh kemungkinan lama.
- Catatan interpretasi dengan angka nyata belum ditulis. Ringkasan otomatis di akhir notebook menyusun kalimat dari hasil yang dihitung, tapi analisis yang lebih panjang menunggu run penuh.

### Batasan desain yang perlu diketahui

- Peringkat fitur ANOVA dihitung dari seluruh data latih, bukan di dalam tiap fold, jadi skor cross-validation sedikit optimistis. Berlaku sama untuk kedua kerangka dan tidak menyentuh set uji.
- Nilai `k` dipilih dari kurva Logistic Regression scikit-learn saja, lalu dipakai kedua kerangka.
- Random Forest di-tuning hanya pada satu nilai `k` karena biayanya.
- Waktu prediksi MLlib sudah termasuk menarik hasil ke driver.
- Worker Spark hanya punya 2 core, jadi hasil kecepatan berlaku untuk kondisi ini dan tidak menggambarkan Spark pada data yang jauh lebih besar.
- Tiap pengukuran waktu dijalankan sekali, jadi angkanya punya noise.
- Notebook perlu sesi Spark yang bebas. Tutup kernel notebook lain yang memegang core sebelum Run All.

## 3. Lanjutan 2026-10-14: perbaikan, optimasi, dan run penuh

### AI Studio dan pelatihan ulang

- Diperiksa ulang: registry memuat `['logreg', 'svm']`, prediksi sinopsis kultivasi menghasilkan Xianxia di kedua model, dan `torch`, `sentence_transformers`, `transformers` tidak terimpor.
- Risiko "melatih ulang memunculkan Hybrid dan BiLSTM" ditutup. `DEFAULT_MODELS = ("logreg", "svm")` ditambahkan di `Model/pipeline/models.py` dan dipakai Fase 4 (`PHASE4_MODELS` kosong) serta `Model/train.py` (tanpa `--models`). Model neural tetap bisa dilatih bila namanya diminta.
- `README.md` dan `Doc/model_card.md` disesuaikan: tabel Head A memakai angka katalog 39.660 judul, Linear SVM ditandai bawaan, Hybrid dan BiLSTM ditandai tidak disajikan.
- Tidak diubah: `docker/Dockerfile` (masih memasang torch dan BGE; memperkecil image butuh rebuild).

### Notebook sklearn vs MLlib

Uji cepat lama lolos tanpa error, tetapi probe pada data penuh menemukan dua masalah yang akan menggagalkan run penuh:

- Random Forest MLlib pada 10.000 fitur kehabisan memori executor (`OutOfMemoryError`), karena MLlib menyimpan tiap baris sebagai larik bin sepanjang jumlah fitur.
- `localCheckpoint()` membuat kegagalan itu fatal: begitu executor mati, blok checkpoint ikut hilang ("Checkpoint block not found").

Perbaikan dan optimasi yang diterapkan:

| Perubahan | Alasan |
|---|---|
| `checkpoint()` ke MinIO menggantikan `localCheckpoint()` | Tahan terhadap executor yang mati. |
| Random Forest mendapat kurva dan `k` sendiri (maks. 3.000) | Menghindari OOM; skor RF juga lebih baik dengan fitur lebih sedikit. |
| Tuning memakai pipeline seleksi + model di dalam CV (`SelectKBest` / `UnivariateFeatureSelector`) | Seleksi ikut dilatih ulang per fold, jadi batasan "skor CV optimistis" hilang. |
| Refit MLlib tidak diulang; durasinya dicatat pipeline bertimer | Sebelumnya model terbaik di-refit dua kali. |
| Grid diperlebar (C sampai 256, alpha sampai 0,001, depth 10/20/30) | Di uji awal nilai terbaik selalu jatuh di ujung grid. |
| 2 partisi Spark (= core) dan `parallelism=2` di CrossValidator untuk LR dan NB | Setara `n_jobs=2` di scikit-learn; LR MLlib 24% lebih cepat per fit. |
| Jumlah iterasi L-BFGS dicatat di kedua kerangka, plus rasio waktu per iterasi | scikit-learn dan MLlib berhenti di jumlah iterasi berbeda, jadi rasio waktu total menyesatkan. |

Hasil: uji cepat turun dari 17,3 menjadi 8,7 menit walau grid lebih besar. Run penuh (38.983 novel) selesai dalam 19,4 menit tanpa error. Output tersimpan di notebook dan di `Doc/generated/sklearn_vs_mllib.json`, dan catatan interpretasi ada di bagian 12.6 notebook.

Ringkasan angka run penuh:

- Model terbaik: Logistic Regression scikit-learn tuned, macro-F1 0,9520 (MLlib 0,9509).
- Logistic Regression dan Naive Bayes setara di kedua kerangka (selisih di bawah 0,002). Random Forest MLlib tertinggal jauh (0,742 vs 0,857 tuned).
- Seluruh pipeline MLlib 7,6 kali lebih lama. Rasio waktu Logistic Regression turun dari 12,2x (7.923 baris) ke 1,4x (124.744 baris).

Berkas sementara `.tmp_probe/` sudah dihapus.

### Yang masih terbuka

- Gap Random Forest MLlib baru punya dugaan penyebab (binning 32 bin kuantil); belum diuji dengan `maxBins` lebih besar.
- `alpha` Naive Bayes terbaik masih di ujung grid (0,001), walau selisihnya dengan 0,01 di bawah 0,002.
