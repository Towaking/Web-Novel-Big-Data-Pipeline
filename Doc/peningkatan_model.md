# Peningkatan model: metode dari literatur, hasil uji, dan yang diterapkan

Tanggal: 2026-10-15. Data: katalog 39.660 judul dari zona clean, 35 genre (Head A) dan
23.256 judul ber-rating tepercaya (Head D). Semua model tetap ML klasik: tanpa jaringan
saraf, tanpa embedding pralatih, tanpa LLM.

## Ringkasan

Angka set uji yang sama (7.932 judul untuk Head A, 4.652 judul untuk Head D):

| Head | Ukuran | Sebelum | Sesudah |
|---|---|---|---|
| A, model bawaan | makro-F1 | 0,5913 (Linear SVM) | **0,6121** (ensemble SVM + Random Forest) |
| A, model bawaan | mikro-F1 | 0,6540 | 0,6738 |
| A, model bawaan | Precision@3 | 0,7877 | 0,8007 |
| A, Linear SVM | makro-F1 | 0,5913 | 0,6093 |
| A, Logistic Regression | makro-F1 | 0,5840 | 0,6027 |
| A, Random Forest (baru) | makro-F1 | - | 0,5522 |
| D, estimasi rating | MAE / R² | 0,3395 / 0,307 | **0,3250 / 0,354** |

- Penyumbang terbesar adalah penyetelan ulang regularisasi (C) dan bobot tiap blok fitur dengan
  TPE. Nilai lama dipilih saat katalog baru 4.460 judul dan tidak pernah disetel ulang.
- Random Forest ditambahkan sebagai model ketiga. Sendirian ia paling lemah, tetapi kesalahannya
  berbeda dari SVM, sehingga gabungan keduanya menjadi model terbaik dan kini model bawaan.
- Ensemble menggabungkan log-odds (logarithmic opinion pool), bukan probabilitas. Rata-rata
  probabilitas menurunkan Precision@3 secara bermakna (-0,0040, p = 0,02); log-odds tidak (-0,0014,
  p = 0,14) dengan makro-F1 yang sama.
- Kata yang disensor NovelUpdates (`mu*der`, `r*pe`) dipulihkan sebelum tokenisasi. Skor uji hampir
  tidak berubah, tetapi model tidak lagi buta terhadap kata utuh yang diketik pengguna.
- Head D kini memakai skor ridge teks yang ditumpuk (stacked generalization), tag multi-hot, bobot
  sampel menurut jumlah suara, dan parameter HGB hasil TPE. Evaluasinya memakai TF-IDF dan SVD yang
  di-fit tanpa baris uji.
- Kenaikannya diuji pada lima pembagian data: SVM +0,0197 dan ensemble +0,0231 makro-F1 dibanding
  SVM lama, menang di kelima pembagian (p < 0,0001, t terkoreksi Nadeau & Bengio). Pada pembagian
  utama, F1 naik di 34 dari 35 genre; Sports turun 0,004.
- Belasan metode lain dari literatur diuji dan ditolak dengan angka: BM25, n-gram 1-3, NB-SVM,
  Naive Bayes, SGD, kalibrasi isotonic, ambang eksak, stacking antar genre, classifier chains, dan
  lain-lain.
- Biayanya di dashboard: registry dimuat dalam 3,6 detik (sebelum revisi 7,5) dengan RAM puncak sekitar
  757 MB (sebelumnya 674 MB), setelah koefisien float32, vectorizer TF-IDF yang dipakai bersama, dan
  artefak pickle biasa yang dimuat unpickler C.

## 1. Cara mengukur

- **Pembagian data sama dengan artefak lama.** Baris diurutkan `series_id`, lalu 80% pengembangan
  (31.728) / 20% uji (7.932) dengan `random_state=42`. Angka "sebelum" direproduksi ulang di frame
  yang sama dan cocok dengan `metrics.json` lama (SVM uji 0,5913).
- **Pemilihan hanya dari set pengembangan.** Tiap kandidat dinilai dengan makro-F1 out-of-fold
  (OOF, 3-fold) setelah ambang per genre disetel di OOF itu. Set uji disentuh sekali per
  konfigurasi akhir dan tidak dipakai memilih apa pun.
- **Eksperimen lokal, model final di container.** Ratusan konfigurasi dijalankan di env lokal
  (scikit-learn 1.7.2, cache matriks TF-IDF per fold). Model yang disajikan dilatih ulang lewat
  jalur resmi `python -m src.phases.phase4_model` di container (scikit-learn 1.5.2); selisih
  angka antar versi di bawah 0,001.
- **Uji berpasangan lima pembagian.** Konfigurasi lama dan baru dilatih pada lima seed pembagian
  yang sama, lalu selisihnya diuji dengan t berpasangan dan t terkoreksi Nadeau & Bengio (2003),
  yang memperhitungkan tumpang tindih data latih antar pembagian.

## 2. Metode yang dikaji

Status: **diterapkan** (masuk kode dan artefak), **ditolak** (diuji, tidak lebih baik), atau
**ditambahkan** (diminta, disajikan walau tidak menjadi model terbaik).

### 2.1 Pra-pemrosesan teks (Head A, SVM C=0,3 balanced)

| Metode | Rujukan | OOF makro | Uji makro | Status |
|---|---|---|---|---|
| Baseline (kolom Fase 2) | - | 0,5833 | 0,5913 | pembanding |
| Pemulihan kata bersensor NovelUpdates | temuan data proyek ini | 0,5835 | 0,5911 | diterapkan |
| Stemming dimatikan | Porter (1980) | 0,5839 | 0,5899 | ditolak: stemming dipertahankan |
| Tag sebagai satu token frasa | - | 0,5824 | 0,5915 | ditolak |
| Tag frasa sebagai blok tambahan | - | 0,5820 | 0,5905 | ditolak |

**Kata bersensor.** NovelUpdates mengganti satu huruf per `*` pada kosakata tertentu: 15% sinopsis
dan 15% daftar tag terdampak (`mu*der`, `su*cide`, `r*pe`, `s*x`, `S*aves`, `Interc**rse`). Tokenizer
Fase 2 memecahnya menjadi `der`, `cide`, atau membuangnya karena kurang dari 3 huruf. Situs juga
ikut menyensor kata yang memuat substring terlarang (`gr*pe` = grape, `peac*ck` = peacock), jadi
pemulihan memakai daftar kata target berprioritas, bukan pencocokan otomatis ke kosakata korpus
(yang memetakan `r*pe` ke "rope" dan `s*x` ke "six").

Di set uji skornya nyaris tidak bergeser karena token rusak itu konsisten di data latih dan uji.
Manfaatnya terlihat saat input berupa kata utuh, seperti yang diketik pengguna AI Studio. Diukur
pada 2.352 judul uji yang terdampak sensor, dengan sinopsis versi tanpa sensor sebagai input:

| Model | Makro-F1 | Mikro-F1 | Adult | Smut | Ecchi | Tragedy |
|---|---|---|---|---|---|---|
| Dilatih dari teks bersensor (lama) | 0,5946 | 0,6688 | 0,709 | 0,682 | 0,568 | 0,360 |
| Dilatih dari teks yang dipulihkan | 0,6014 | 0,6779 | 0,753 | 0,725 | 0,603 | 0,384 |

### 2.2 Representasi fitur (Head A, SVM C=0,3 balanced)

| Metode | Rujukan | OOF makro | Uji makro | Status |
|---|---|---|---|---|
| Kosakata 40k/40k/10k (baseline) | - | 0,5835 | 0,5911 | pembanding |
| Kosakata 150k/150k/20k | Joachims (1998) | 0,5887 | 0,5965 | membantu di C lama; lihat 2.5 |
| Kosakata penuh, min_df 2 (±740k fitur) | Joachims (1998) | 0,5898 | 0,5980 | idem, 2,3x lebih besar dari 150k |
| N-gram 1-3 | - | 0,5871 | 0,5959 | ditolak |
| Okapi BM25 menggantikan TF-IDF | Robertson & Zaragoza (2009) | 0,5837 | 0,5907 | ditolak |
| max_df 0,5 | - | 0,5831 | 0,5912 | ditolak |
| Blok char n-gram 3-5 | - | 0,5872 | 0,5933 | ditolak: setara setelah tuning, 5x lebih lambat |

Di C=0,3 lama, kosakata lebih besar membantu, sejalan dengan argumen Joachims bahwa teks punya
banyak fitur relevan. Tetapi efek itu hilang setelah C dan bobot blok disetel ulang (2.5): yang
sebenarnya kurang tepat adalah regularisasinya, bukan jumlah fiturnya.

### 2.3 Algoritma klasifikasi (representasi 150k, sebelum tuning)

| Metode | Rujukan | OOF makro | Uji makro | Status |
|---|---|---|---|---|
| Linear SVM C=0,3 balanced | Joachims (1998); Fan dkk. (2008) | 0,5887 | 0,5965 | pembanding |
| Linear SVM, C disapu 0,1-1,0 | - | 0,5831-0,5919 | 0,5905-0,6006 | C kecil lebih baik |
| NB-SVM (fitur diskalakan rasio log-count NB) | Wang & Manning (2012) | 0,5929 | 0,5998 | ditolak: setara SVM C=0,1 |
| NB-LR | Wang & Manning (2012) | 0,5821 | 0,5880 | ditolak |
| Complement Naive Bayes | Rennie dkk. (2003) | 0,4457 | 0,4764 | ditolak |
| Multinomial Naive Bayes | McCallum & Nigam (1998) | 0,5448 | 0,5469 | ditolak |
| SGD modified Huber | Zhang (2004) | 0,5498 | 0,5571 | ditolak |
| Random Forest multi-output (200 pohon, tanpa class_weight) | Breiman (2001); Madjarov dkk. (2012) | 0,5607 | 0,5675 | ditambahkan (versi ringan, lihat bawah) |
| ExtraTrees | Geurts dkk. (2006) | 0,5670 | 0,5688 | diuji, tidak disajikan |

**Random Forest.** Satu hutan multi-output untuk 35 genre sekaligus, sesuai temuan Madjarov dkk.
(2012) bahwa random forest pohon multi-target termasuk metode multi-label terbaik di 11 dataset.
Penyetelan di set pengembangan (fitur 150k, bobot lama):

| Variasi | OOF makro | Catatan |
|---|---|---|
| class_weight balanced_subsample | 0,5275 | bobot 35 genre dikalikan per sampel, jadi ekstrem |
| tanpa class_weight | 0,5607 | ketimpangan sudah ditangani ambang per genre |
| di atas SVD 300 dimensi | 0,4399 | ditolak |
| fitur chi-kuadrat 1.500 / 3.000 / 10.000 | 0,5602 / 0,5607 / 0,5601 | 3.000 dipakai |
| min_samples_leaf 1 / 2 / 5 | 0,5605 / 0,5607 / 0,5504 | |
| max_features 5% kolom (vs sqrt) | 0,5593 | |
| 500 pohon | 0,5627 | |

Tiap simpul menyimpan probabilitas 35 genre x 2 kelas (560 byte), jadi 200 pohon tanpa batas daun
mencapai 2,6 juta simpul (228 MB terkompresi, ±1,4 GB di RAM). Ukuran hutan lalu diuji terhadap
perannya sebagai anggota ensemble (fitur final 75k, bobot hasil tuning):

| Hutan | Simpul | RAM | RF OOF | Ensemble OOF | Ensemble uji |
|---|---|---|---|---|---|
| 150 pohon x 3.000 daun | ±900 rb | ±535 MB | 0,5590 | 0,6061 | 0,6137 |
| 50 pohon x 4.000 daun | ±400 rb | ±238 MB | 0,5504 | 0,6061 | 0,6147 |
| **60 pohon x 2.000 daun** | **±240 rb** | **±143 MB** | **0,5492** | **0,6057** | **0,6141** |
| 100 pohon x 1.000 daun | ±200 rb | ±119 MB | 0,5476 | 0,6051 | 0,6118 |

Hutan kecil lebih lemah sendirian, tetapi hampir sama bergunanya di dalam ensemble, jadi yang
disajikan 60 pohon x 2.000 daun.

### 2.4 Kalibrasi dan ambang keputusan

| Metode | Rujukan | Hasil | Status |
|---|---|---|---|
| Kalibrasi isotonic menggantikan sigmoid | Zadrozny & Elkan (2002) | OOF 0,5875 vs 0,5887 | ditolak |
| Satu SVM terkalibrasi (`ensemble=False`) | Platt (1999) | OOF 0,5788 vs 0,5887 | ditolak |
| Ambang per genre, grid 0,05-0,80 langkah 0,025 | Yang (2001), SCut; Hoang (2018) | cv-ambang 0,5815 | dipertahankan |
| Grid lebih halus (0,01 dan 0,005) | Fan & Lin (2007) | cv-ambang 0,5813-0,5815 | ditolak |
| Sapuan eksak semua skor | Lipton dkk. (2014) | cv-ambang 0,5814 | ditolak |
| Minimal satu genre per judul (RCut) | Yang (2001) | tidak berubah | ditolak |

"cv-ambang" adalah makro-F1 dengan ambang disetel di dua fold OOF dan dinilai di fold ketiga,
supaya strategi ambang dibandingkan tanpa menilai di data tempat ia disetel. Strategi ambang yang
ada sudah di batas yang bisa dicapai; peningkatan harus datang dari representasi dan model.

### 2.5 Penyetelan hiperparameter

TPE (Tree-structured Parzen Estimator; Bergstra dkk. 2011) lewat Optuna (Akiba dkk. 2019), 40 trial
untuk SVM atas C, class_weight, dan bobot empat blok fitur (korpus, sinopsis, tag, metadata).
Konfigurasi lama dimasukkan sebagai trial pertama.

| | Lama | Hasil TPE |
|---|---|---|
| Bobot blok korpus / sinopsis / tag / meta | 1,0 / 1,0 / 1,6 / 0,6 | 2,31 / 0,95 / 1,48 / 0,48 |
| SVM C, class_weight | 0,3, balanced | 0,0354, None |
| LogReg C, class_weight (disapu di bobot yang sama) | 4,0, None | 0,25, None (0,5 setara) |
| Makro-F1 OOF SVM (kosakata 150k) | 0,5887 | 0,6027 |

Sepuluh trial teratas berselisih hanya 0,0006, jadi optimumnya landai dan tidak bergantung pada
satu titik beruntung. class_weight=None menang di semua trial teratas: ambang per genre sudah
mengoreksi ketimpangan kelas, sehingga "balanced" menjadi koreksi ganda.

Dengan bobot dan C baru, pilihan representasi diuji ulang:

| Kosakata per blok (korpus / sinopsis / tag) | OOF makro | Uji makro | Uji mikro |
|---|---|---|---|
| 40k / 40k / 10k (asli) | 0,6020 | 0,6101 | 0,6688 |
| **75k / 75k / 15k** | **0,6022** | **0,6101** | **0,6712** |
| 100k / 100k / 20k | 0,6022 | 0,6095 | 0,6718 |
| 150k / 150k / 20k | 0,6027 | 0,6106 | 0,6721 |
| penuh, min_df 2 (±740k) | 0,6024 | 0,6104 | 0,6736 |
| 150k + char n-gram | 0,6032 | 0,6104 | 0,6729 |

Hampir seluruh kenaikan datang dari C dan bobot blok: kosakata asli 40k pun mencapai 0,6101 di
set uji. Kosakata 75k dipakai sebagai kompromi: mikro-F1 setara 150k dengan separuh jumlah fitur.
Uji berpasangan (bagian 5) menaksir harganya -0,0008 makro-F1 dibanding 150k.

### 2.6 Ensemble dan korelasi antar genre (fitur 150k, bobot hasil tuning)

| Metode | Rujukan | OOF makro | cv-ambang | Uji makro | Status |
|---|---|---|---|---|---|
| Linear SVM hasil tuning | - | 0,6027 | 0,6006 | 0,6106 | anggota |
| Stacking antar label (meta-LR di atas logit 35 genre) | Godbole & Sarawagi (2004) | 0,6014 | - | 0,6066 | ditolak |
| Stacking + fitur asli | Godbole & Sarawagi (2004) | 0,5926 | - | 0,6002 | ditolak |
| Rata-rata SVM + LogReg | Dietterich (2000) | 0,6027 | 0,6005 | 0,6094 | ditolak |
| Rata-rata SVM + NB-SVM | Dietterich (2000) | 0,6041 | 0,6015 | 0,6113 | ditolak |
| Rata-rata probabilitas SVM 0,6 + Random Forest 0,4 | Wolpert (1992); Dietterich (2000) | 0,6064 | 0,6053 | 0,6128 | diganti log-odds |

Dua model linear di fitur yang sama membuat kesalahan yang mirip, jadi rata-ratanya tidak menolong.
Random Forest lebih lemah sendirian, tetapi kesalahannya berbeda, sehingga campurannya mengungguli
SVM di makro- dan mikro-F1. Stacking antar genre tidak menolong, sama seperti temuan pada katalog
4.460 judul.

**Ruang penggabungan.** Rata-rata probabilitas menurunkan Precision@3 dibanding SVM saja di kelima
pembagian. Penggabungan di ruang log-odds (logarithmic opinion pool, Genest dkk. 1984), yaitu
`sigmoid(w * logit(p_svm) + (1 - w) * logit(p_rf))`, mempertahankan makro-F1 dan memperkecil
selisih P@3 sampai tidak bermakna (fitur final 75k, lima seed, t NB):

| Penggabungan | Bobot SVM | Makro-F1 uji | Selisih P@3 vs SVM | p (P@3) | P@1 uji (seed 42) |
|---|---|---|---|---|---|
| Rata-rata probabilitas | 0,6 | 0,6105 | -0,0040 | 0,023 | 0,8971 |
| **Rata-rata log-odds (diterapkan)** | **0,7** | **0,6105** | **-0,0014** | **0,143** | **0,9038** |

**Dua metode tambahan (fitur final 75k, pembagian utama):**

| Metode | Rujukan | OOF makro | Uji makro | Uji P@3 | Status |
|---|---|---|---|---|---|
| LogReg binary relevance C=0,25 (pembanding) | - | 0,5969 | 0,6022 | 0,7953 | pembanding |
| Ensemble classifier chains, 3 rantai acak, LogReg C=0,25 | Read dkk. (2011) | 0,5920 | 0,6003 | 0,7947 | ditolak, 6x lebih lambat |
| SVM final dengan fold OOF stratifikasi iteratif | Sechidis dkk. (2011) | 0,6035 | 0,6109 | 0,8017 | tidak diterapkan |

Stratifikasi iteratif memang meratakan sebaran positif per fold (koefisien variasi 0,012 vs 0,023
pada KFold acak), tetapi skor ujinya hanya +0,0008, di bawah simpangan baku antar pembagian.

### 2.7 Head D: estimasi rating

| Metode | Rujukan | MAE uji | RMSE | R² | CV-MAE |
|---|---|---|---|---|---|
| Baseline: HGB di embedding SVD + numerik | Ke dkk. (2017) | 0,3371 | 0,4293 | 0,3116 | 0,3397 |
| + skor ridge teks out-of-fold (alpha 3) | Wolpert (1992) | 0,3302 | 0,4189 | 0,3444 | 0,3323 |
| + bobot sampel log(1 + jumlah suara) | weighted least squares | 0,3285 | 0,4193 | 0,3431 | 0,3311 |
| + tag multi-hot (708 tag) | - | 0,3288 | 0,4173 | 0,3494 | 0,3311 |
| Gabungan ketiganya | - | 0,3279 | 0,4189 | 0,3445 | 0,3311 |
| **Gabungan + HGB hasil TPE (25 trial)** | Bergstra dkk. (2011) | **0,3247** | **0,4150** | **0,3566** | **0,3284** |
| Loss absolute_error | - | 0,3321 | 0,4241 | 0,3282 | 0,3321 |
| Random Forest regressor (300 pohon) | Breiman (2001) | 0,3308 | 0,4197 | 0,3418 | 0,3330 |
| ExtraTrees regressor | Geurts dkk. (2006) | 0,3329 | 0,4215 | 0,3362 | 0,3334 |

CV-MAE dihitung di set latih saja (5-fold, kecuali baris TPE yang 3-fold). Random Forest regressor
mendekati HGB tetapi 40 kali lebih lambat dilatih, jadi tidak dipakai. Skor ridge mengembalikan
informasi frasa jarang yang hilang saat teks dipadatkan ke SVD 200 dimensi; karena untuk baris
latih nilainya out-of-fold, HGB tidak pernah melihat skor yang dihitung dari baris itu sendiri.

## 3. Yang diterapkan ke kode

| Berkas | Perubahan |
|---|---|
| `src/common/text_prep.py` | `uncensor()` dan `tags_text()`; `normalize()` memulihkan sensor sebelum tokenisasi |
| `Model/pipeline/preprocessing.py` | `rebuild_clean_text()`; `build_feature_frame(pdf, stopwords)` membangun teks latih lewat fungsi inferensi yang sama |
| `Model/pipeline/config.py` | kosakata 75k/75k/15k, bobot blok hasil TPE, `PHASE4_N_JOBS` |
| `Model/pipeline/features.py` | `MultiLabelChi2Selector` untuk Random Forest |
| `Model/pipeline/models.py` | arsitektur `rf` dan `ensemble` (`ENSEMBLE_SPACE = "logit"`), grid SVM/LogReg baru, `DEFAULT_MODELS`, `shrink_coefficients()` (float32), `share_feature_steps()` |
| `Model/pipeline/evaluate.py` | `blend_probabilities(parts, space)`: rata-rata probabilitas atau log-odds |
| `Model/pipeline/train.py` | `blend_members()`: bobot ensemble dipilih di OOF, tanpa fit tambahan; koefisien dirampingkan sebelum diuji |
| `Model/pipeline/rating.py` (baru) | `RatingModel`: skor ridge OOF, tag multi-hot, bobot sampel, parameter HGB hasil TPE |
| `src/phases/phase4_model.py` | satu tabel teks untuk keempat head; Head D memakai `RatingModel`, dievaluasi dengan TF-IDF+SVD tanpa baris uji, dan mencatat skor konfigurasi lama; `save_artifacts` memakai satu vectorizer bersama untuk ketiga pipeline |
| `src/phases/phase5_mllib.py` | fitur dan C scikit-learn dikunci ke konfigurasi pertama agar Tabel 5.3 laporan tetap dapat diulang |
| `src/phases/similarity_graph.py` | homofili dibandingkan dengan F1 model Head A terbaik (kini ensemble) |
| `src/common/model_registry.py` | inferensi ensemble (log-odds) dan `RatingModel`; tag dibersihkan dengan fungsi latih yang sama |
| `Model/train.py` | memakai teks yang dipulihkan; ensemble lama dibuang bila anggotanya dilatih ulang sendiri |
| `app/streamlit_app.py` | deskripsi model baru di AI Studio dan halaman Methodology |
| `scripts/verify_app.py` | memilih tiap model genre di AI Studio dan menekan "Analyse story" |
| `scripts/research/asg2_figures.py` | gambar Laporan Assignment II untuk empat model Head A dan graf baru |
| `scripts/notebook/*.ipynb` | `model_training`, `check`, dan `eda_visualisasi` dieksekusi ulang, 0 error |

## 4. Hasil artefak baru

Dari `Model/artifacts/metrics.json` hasil `python -m src.phases.phase4_model` di container
(scikit-learn 1.5.2, `PHASE4_N_JOBS=10`, 15 menit):

| Model genre | Konfigurasi | OOF makro | Uji makro | Uji mikro | P@3 | Hamming |
|---|---|---|---|---|---|---|
| **Ensemble SVM + RF (bawaan)** | log-odds, bobot SVM 0,7 / RF 0,3 | **0,6059** | **0,6121** | **0,6738** | 0,8007 | 0,1058 |
| Linear SVM | C=0,0354, tanpa class_weight | 0,6022 | 0,6093 | 0,6718 | **0,8015** | **0,1057** |
| Logistic Regression | C=0,25, tanpa class_weight | 0,5970 | 0,6027 | 0,6670 | 0,7953 | 0,1090 |
| Random Forest | 60 pohon x 2.000 daun, 3.000 fitur chi-kuadrat | 0,5492 | 0,5522 | 0,6296 | 0,7410 | 0,1239 |
| Linear SVM lama | kosakata 40k, C=0,3 balanced | 0,5834 | 0,5913 | 0,6540 | 0,7877 | 0,1143 |
| Logistic Regression lama | kosakata 40k, C=4 | 0,5756 | 0,5840 | 0,6438 | 0,7747 | 0,1196 |
| Hybrid BGE (riset, fitur lama) | tidak disajikan | 0,5912 | 0,5971 | 0,6574 | 0,7923 | 0,1128 |

Ensemble juga memiliki Precision@1 tertinggi (0,9043, SVM 0,9032). Model ML klasik yang disetel
ulang kini melampaui model Hybrid dengan encoder BGE yang dulu menjadi model terbaik. F1 naik di 34
dari 35 genre (ensemble dibanding SVM lama), dan kenaikan terbesar justru di genre yang sulit:

| Genre | SVM lama | Ensemble | Selisih |
|---|---|---|---|
| Tragedy | 0,363 | 0,408 | +0,045 |
| Psychological | 0,424 | 0,469 | +0,045 |
| Mecha | 0,505 | 0,549 | +0,043 |
| Josei | 0,554 | 0,597 | +0,042 |
| Shounen Ai | 0,446 | 0,487 | +0,041 |
| Mature | 0,457 | 0,492 | +0,035 |
| Shoujo Ai (paling sulit) | 0,169 | 0,187 | +0,018 |
| Romance (paling mudah) | 0,866 | 0,873 | +0,007 |
| Sports (satu-satunya yang turun) | 0,750 | 0,746 | -0,004 |

Kepala lain:

| Head | Ukuran | Sebelum | Sesudah |
|---|---|---|---|
| B, rekomendasi | genre overlap@10 | 0,9498 | 0,949 |
| C, arketipe | k / silhouette | 7 / 0,0285 | 7 / 0,0285 |
| D, rating | MAE | 0,3395 | 0,3250 |
| D, rating | RMSE | 0,4307 | 0,4157 |
| D, rating | R² | 0,3071 | 0,3544 |
| D, rating | lebih baik dari tebakan rata-rata | 18,2% | 21,7% |

Angka "sebelum" Head D adalah konfigurasi lama yang dilatih ulang pada baris latih yang sama di run
ini (`previous_config` di `metrics.json`). Keduanya dievaluasi dengan TF-IDF dan SVD yang di-fit
ulang tanpa 4.652 baris uji (`evaluation_representation`), sehingga teks baris uji tidak ikut
membentuk ruang fitur. Model Head D yang disajikan tetap memakai representasi bersama dari seluruh
katalog. Head B dan C tidak diubah; pergeserannya berasal dari korpus yang kini dipulihkan sensornya.

Biaya di dashboard (container Streamlit, proses baru, tanpa beban lain):

| | Sebelum revisi | Versi pertama revisi | Disajikan |
|---|---|---|---|
| `genre_models.joblib` | 92 MB (2 model) | 202 MB | 238 MB tanpa kompresi (3 model + ensemble) |
| `rating_regressor.joblib` | 0,5 MB | 3,0 MB | 7,3 MB tanpa kompresi |
| Waktu muat registry (sekali per server, lalu di-cache) | 7,5 dtk | 12,5 dtk | 3,6 dtk |
| RAM puncak saat memuat | 674 MB | 1,24 GB | 757 MB |

Tiga perampingan tanpa mengubah prediksi:

- Koefisien model linear disimpan sebagai float32. Pada 3.000 judul, probabilitas bergeser paling
  banyak 3e-8 dan 0 dari 105.000 keputusan genre berubah, untuk LogReg maupun SVM.
- Ketiga pipeline memuat langkah TF-IDF yang identik byte demi byte. `share_feature_steps()`
  menyimpannya sekali, sehingga pemuatan `genre_models.joblib` turun dari 7,2 ke 3,1 detik dan RSS
  puncaknya dari 849 ke 671 MB. Probabilitas ketiga model pada 2.000 judul identik (selisih maks.
  3e-16 untuk RF, berasal dari urutan penjumlahan paralel).
- `joblib.load` memakai unpickler Python murni. Artefak kini ditulis sebagai pickle protokol 5 tanpa
  kompresi (`src/common/artifact_io.py`) dan dimuat dengan unpickler C: `genre_models` 2,0 detik dari
  bind mount, dibanding 3,1 sampai 4,6 detik sebelumnya; `svd` dari 1,1 detik menjadi 0,01 detik. Berkasnya
  tetap terbaca `joblib.load` (notebook, `Model/train.py`), dan berkas joblib lama tetap dimuat lewat
  fallback. Cadangan format lama: `Model/artifacts/_backup_joblib_2026-10-15/`.

Kompresi joblib tidak berpengaruh pada RAM puncak (849 MB pada kompresi 0, 1, maupun 3); yang membuatnya
lambat adalah unpicklernya. Kosakata 150k dengan Random Forest 150 pohon sempat dilatih lebih dulu:
berkasnya 421 MB dan RAM puncaknya 2,4 GB.

Verifikasi:

- `scripts/notebook/model_training.ipynb`, `check.ipynb`, dan `eda_visualisasi.ipynb` dieksekusi
  ulang di container (0 sel error). Metrik keempat head identik dengan run `phase4_model`
  sebelumnya, termasuk F1 per genre, jadi pelatihannya deterministik dan dapat diulang.
- `scripts/verify_app.py` setelah Streamlit di-restart: 15 halaman dan 4 model AI Studio (ensemble,
  logreg, svm, rf) lulus tanpa exception. Tampilan di browser belum dilihat langsung.
- Salinan `models/` di bucket curated-data sama byte demi byte dengan `Model/artifacts`.

## 5. Uji berpasangan lima pembagian

Konfigurasi yang disajikan, lima seed pembagian (42, 7, 123, 2024, 99), masing-masing 31.728 latih
dan 7.932 uji. Rata-rata ± simpangan baku antar seed; t NB adalah t terkoreksi Nadeau & Bengio
(2003) dengan faktor varians (1/5 + 7.932/31.728), derajat bebas 4.

| Perbandingan | Metrik | Lama | Baru | Selisih | Menang | t NB | p |
|---|---|---|---|---|---|---|---|
| SVM baru vs lama | makro-F1 uji | 0,5874 ± 0,0022 | 0,6071 ± 0,0017 | +0,0197 ± 0,0007 | 5/5 | 43,4 | < 0,0001 |
| SVM baru vs lama | mikro-F1 uji | 0,6513 ± 0,0018 | 0,6711 ± 0,0006 | +0,0198 ± 0,0014 | 5/5 | 21,7 | < 0,0001 |
| SVM baru vs lama | P@3 uji | 0,7866 ± 0,0019 | 0,8022 ± 0,0012 | +0,0156 ± 0,0018 | 5/5 | 12,8 | 0,0002 |
| LogReg baru vs lama | makro-F1 uji | 0,5798 ± 0,0022 | 0,5997 ± 0,0023 | +0,0199 ± 0,0009 | 5/5 | 33,3 | < 0,0001 |
| **Ensemble log-odds vs SVM baru** | makro-F1 uji | 0,6071 ± 0,0017 | 0,6105 ± 0,0019 | +0,0034 ± 0,0012 | 5/5 | 4,1 | 0,016 |
| **Ensemble log-odds vs SVM baru** | mikro-F1 uji | 0,6711 ± 0,0006 | 0,6739 ± 0,0010 | +0,0028 ± 0,0005 | 5/5 | 7,6 | 0,002 |
| **Ensemble log-odds vs SVM baru** | P@3 uji | 0,8022 ± 0,0012 | 0,8008 ± 0,0018 | -0,0014 ± 0,0011 | 0/5 | -1,8 | 0,143 |
| **Ensemble log-odds vs SVM lama** | makro-F1 uji | 0,5874 ± 0,0022 | 0,6105 ± 0,0019 | +0,0231 ± 0,0015 | 5/5 | 23,4 | < 0,0001 |
| **Ensemble log-odds vs SVM lama** | P@3 uji | 0,7866 ± 0,0019 | 0,8008 ± 0,0018 | +0,0142 ± 0,0019 | 5/5 | 11,0 | 0,0004 |
| Ensemble rata-rata probabilitas vs SVM baru | makro-F1 uji | 0,6071 ± 0,0017 | 0,6105 ± 0,0023 | +0,0034 ± 0,0011 | 5/5 | 4,6 | 0,010 |
| Ensemble rata-rata probabilitas vs SVM baru | P@3 uji | 0,8022 ± 0,0012 | 0,7982 ± 0,0019 | -0,0040 ± 0,0016 | 0/5 | -3,6 | 0,023 |
| SVM kosakata 75k vs 150k | makro-F1 uji | 0,6079 ± 0,0016 | 0,6071 ± 0,0017 | -0,0008 ± 0,0004 | 0/5 | -2,9 | 0,043 |
| Random Forest vs SVM baru | makro-F1 uji | 0,6071 ± 0,0017 | 0,5525 ± 0,0024 | -0,0546 ± 0,0027 | 0/5 | -30,0 | < 0,0001 |

Ragam antar pembagian jauh lebih kecil daripada di katalog 4.460 judul (simpangan baku makro-F1
0,002 vs 0,011), sehingga selisih +0,02 sudah sekitar 25 kali simpangan bakunya. Ensemble menang
tipis tetapi konsisten di F1. Dengan rata-rata probabilitas, P@3-nya kalah secara bermakna karena
probabilitas Random Forest mengacak sedikit urutan tiga genre teratas; di ruang log-odds selisih
itu tinggal -0,0014 dan tidak bermakna, sementara P@1 justru tertinggi. Bila yang dipentingkan
peringkat tiga teratas saja, Linear SVM tetap tersedia di AI Studio.

## 6. Batasan dan yang belum dikerjakan

- **Skor OOF hasil TPE optimistis.** 40 trial memilih yang terbaik di OOF yang sama, jadi OOF
  0,6059 sedikit terlalu tinggi. Angka yang jujur adalah set uji (0,6121) dan rata-rata lima
  pembagian (0,6105).
- **Sensor hanya dipulihkan di lapisan model.** Kolom `corpus_clean` di zona clean, analitik teks
  Fase 3, dan MapReduce word count masih memakai teks Fase 2. Fase 5 (perbandingan MLlib) sengaja
  tetap memakai kolom Fase 2, fitur 40k/40k/10k, dan C lama supaya angkanya dapat diulang.
- **Hybrid dan BiLSTM tidak dilatih ulang.** Angka di `excluded_models` berasal dari fitur lama dan
  keduanya di luar batas ML klasik proyek ini.
- **Representasi bersama yang disajikan di-fit pada seluruh katalog.** Ini wajar untuk Head B dan C
  (tanpa label). Untuk angka Head D, evaluasinya kini memakai TF-IDF dan SVD yang di-fit tanpa baris
  uji, jadi metrik di atas tidak terpengaruh.
- **Katalog ~76k.** Setelah scrape selesai, parameter hasil tuning perlu dicek ulang dengan skrip
  di `scripts/research/peningkatan_model/`.
- **Di luar cakupan:** fine-tuning transformer, karena proyek ini dibatasi ke ML klasik.

## 7. Analisis dan dokumen yang ikut diperbarui

- **Content Graph.** `python -m src.phases.similarity_graph` dijalankan ulang di atas embedding
  baru: 39.660 simpul, 314.786 sisi, 27 komunitas Louvain, modularity 0,7192. Homofili kini
  dibandingkan dengan F1 ensemble (`homophily_reference_model`), Spearman 0,844 (parsial 0,857
  setelah prevalensi dikendalikan). Lima genre berhomofili terendah: Shoujo Ai, Shounen Ai, Tragedy,
  Seinen, Psychological.
- **Laporan Assignment II** (`Doc/Laporan/Laporan_Assignment_II_Big_Data.docx` dan `.pdf`, 55
  halaman). Bab 4.3, 5, dan 6 disunting langsung di .docx: pemulihan sensor (5.1), bobot blok TPE dan
  fitur Head D baru (5.2), Random Forest dan ensemble log-odds (5.3), Tabel 5.4 empat model, Tabel
  5.5 sampai 5.6 dan 6.1 sampai 6.2 dengan angka baru, subbab analisis tentang sumber kenaikan,
  delapan referensi baru, dan enam gambar dari `scripts/research/asg2_figures.py`. Isi Fase 5 tidak
  diubah. Versi sebelumnya disimpan di `Doc/Laporan/_arsip_asg2_2026-10-10/`.
- **Optimasi dashboard** (diukur per halaman dengan `AppTest`, proses baru lalu rerun hangat):
  AI Studio dingin 9,2 ke 4,1 detik (format artefak di atas); Methodology 9,2 ke 1,9 detik dan 1.074 ke
  323 MB karena hanya membaca `metrics.json`, bukan seluruh registry; Dataset menggabungkan kolom list
  di Arrow dan memakai memory pool sistem (jemalloc bawaan Arrow menahan sekitar 400 MB setelah
  konversi), sel yang ditampilkan identik di 39.668 x 50 sel; Title Lookup menyimpan frekuensi tag di
  cache, rerun 0,66 ke 0,49 detik. `scripts/verify_app.py`: 15 halaman dan 4 model lulus.
- **Pemeriksaan ulang Laporan Assignment II, kata demi kata** (2026-10-15): angka EDA dihitung ulang dari
  `Dataset/variety` dan cocok dalam batas drift snapshot, kecuali kata tersering (yang benar "one",
  "world", "life", "even", "like"). Diperbaiki pula: deskripsi sistem yang sudah berubah sejak Assignment I
  (katalog gabungan sebagai satu-satunya sumber raw, TLS dan kredensial khusus, retensi versi, satu worker
  Spark, enam halaman data platform), satuan total lake (MB, bukan MiB), persentase `artists` dan
  `english_publisher` yang berasal dari dataset 4.461 judul, alasan kebocoran fitur Head D, enam referensi
  yang belum dikutip, spasi sebelum titik dua, dan format paragraf yang tidak seragam. HDFS dimuat ulang
  dari ekspor terbaru, dan verifikasi MapReduce kembali 4 dari 4 identik dengan Spark.

## Referensi

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD 2019. <https://arxiv.org/abs/1907.10902>
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. NIPS 24. <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. Multiple Classifier Systems, LNCS 1857, 1-15.
- Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A Library for Large Linear Classification. JMLR, 9, 1871-1874.
- Fan, R.-E., & Lin, C.-J. (2007). A Study on Threshold Selection for Multi-label Classification. Laporan teknis, National Taiwan University. <https://www.csie.ntu.edu.tw/~cjlin/papers/threshold.pdf>
- Genest, C., Weerahandi, S., & Zidek, J. V. (1984). Aggregating Opinions through Logarithmic Pooling. Theory and Decision, 17(1), 61-70.
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely Randomized Trees. Machine Learning, 63(1), 3-42.
- Godbole, S., & Sarawagi, S. (2004). Discriminative Methods for Multi-labeled Classification. PAKDD 2004, LNCS 3056, 22-30.
- Hoang, Q. (2018). Predicting Movie Genres Based on Plot Summaries. arXiv:1801.04813. <https://arxiv.org/abs/1801.04813>
- Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. ECML-98, LNAI 1398, 137-142.
- Ke, G., dkk. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 30.
- Lipton, Z. C., Elkan, C., & Narayanaswamy, B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD 2014, LNCS 8725, 225-239.
- Madjarov, G., Kocev, D., Gjorgjevikj, D., & Džeroski, S. (2012). An Extensive Experimental Comparison of Methods for Multi-label Learning. Pattern Recognition, 45(9), 3084-3104.
- McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for Naive Bayes Text Classification. AAAI-98 Workshop on Learning for Text Categorization.
- Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. Machine Learning, 52(3), 239-281.
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. Advances in Large Margin Classifiers.
- Porter, M. F. (1980). An Algorithm for Suffix Stripping. Program, 14(3), 130-137.
- Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier Chains for Multi-label Classification. Machine Learning, 85(3), 333-359.
- Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. ICML 2003, 616-623.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.
- Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the Stratification of Multi-label Data. ECML PKDD 2011, LNCS 6913, 145-158.
- Wang, S., & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL 2012, 90-94. <https://aclanthology.org/P12-2018/>
- Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241-259.
- Yang, Y. (2001). A Study of Thresholding Strategies for Text Categorization. SIGIR 2001, 137-145.
- Zadrozny, B., & Elkan, C. (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. KDD 2002, 694-699.
- Zhang, T. (2004). Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms. ICML 2004.
