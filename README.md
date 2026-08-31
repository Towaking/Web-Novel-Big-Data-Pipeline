# Novel Analytics — Big Data & Hybrid AI Pipeline

Pipeline Big Data lokal berbasis Docker untuk dataset `Dataset/data_novel.csv`
(4.461 judul novel dari NovelUpdates). Menggabungkan ekstraksi *insight* skala
besar (Spark SQL) dengan lapis pemodelan AI, lalu disajikan lewat dashboard
Streamlit interaktif.

## Arsitektur

```
                    ┌──────────────────────────────────────┐
  Dataset/*.csv ──▶ │  MinIO (S3-compatible Data Lake)     │
                    │  raw-data → clean-data → curated-data│
                    └──────────────┬───────────────────────┘
                                   │ s3a://
                    ┌──────────────▼───────────────────────┐
                    │  Spark Standalone Cluster            │
                    │  spark-master + spark-worker (N)     │
                    └──────────────┬───────────────────────┘
                                   │ submit
                    ┌──────────────▼───────────────────────┐
                    │  pipeline (driver) · Fase 1–4        │
                    └──────────────┬───────────────────────┘
                                   │ artefak
                    ┌──────────────▼───────────────────────┐
                    │  streamlit · BI + AI Playground      │
                    └──────────────────────────────────────┘
```

| Service | Peran | Port host |
|---|---|---|
| `minio` | Data lake S3-compatible | 9000 (API), 9001 (Console) |
| `minio-init` | Bootstrap bucket (sekali jalan, idempotent) | — |
| `spark-master` | Koordinator cluster | 8080 (UI), 7077 |
| `spark-worker` | Executor, bisa di-scale | 8081–8085 (UI, dialokasikan otomatis dari rentang — cek `docker compose ps`) |
| `pipeline` | Driver container tempat job Fase 1–4 disubmit | — |
| `notebook` | JupyterLab dengan driver Spark sendiri | 8888 |
| `streamlit` | Dashboard produk (BI + AI Studio) | 8501 |

Seluruh compute service memakai **satu image yang sama** (`bigdata-novel:base`)
agar versi Python & PySpark identik antara driver dan executor — syarat wajib
supaya serialisasi UDF PySpark tidak gagal.

## Quickstart

```bash
cp .env.example .env      # sesuaikan kredensial bila perlu
docker compose build      # ± 3–6 menit pada build pertama
docker compose up -d
docker compose exec pipeline python scripts/verify_env.py
```

Bila `verify_env.py` melaporkan **LULUS** untuk ketiga pemeriksaan, jalankan
seluruh pipeline (Fase 1 → 4):

```bash
docker compose exec pipeline python scripts/run_pipeline.py
```

Lalu buka dashboard di <http://localhost:8501>.

Menjalankan sebagian fase saja:

```bash
docker compose exec pipeline python scripts/run_pipeline.py --only 3    # satu fase
docker compose exec pipeline python scripts/run_pipeline.py --from 2    # fase 2 s/d 4
docker compose exec pipeline python -m src.phases.phase2_clean          # langsung per modul
```

Dengan `make` (Git Bash / WSL):

```bash
make build && make up && make verify && make pipeline
```

Antarmuka:

- Dashboard NovelLens → <http://localhost:8501>
- JupyterLab (eksplorasi PySpark) → <http://localhost:8888> — notebook siap pakai di [scripts/check.ipynb](scripts/check.ipynb)
- Spark Master UI → <http://localhost:8080>
- MinIO Console → <http://localhost:9001> (user/pass dari `.env`)

## Melatih Ulang Model

Skrip [Model/train.py](Model/train.py) melatih ulang model tanpa menjalankan
seluruh pipeline. Arsitektur yang tidak diminta tetap dipertahankan dari
artefak lama, jadi melatih satu model tidak menghapus yang lain.

```bash
docker compose exec pipeline python Model/train.py                        # semua head
docker compose exec pipeline python Model/train.py --only genre           # Head A saja
docker compose exec pipeline python Model/train.py --models hybrid        # satu arsitektur
docker compose exec pipeline python Model/train.py --models bilstm --epochs 5   # uji cepat
docker compose restart streamlit                                          # muat artefak baru
```

Arsitektur yang tersedia: `logreg`, `svm`, `hybrid`, `bilstm`.

Menambah kapasitas worker:

```bash
docker compose up -d --scale spark-worker=3
```

## Struktur Proyek

```text
.
├── app/
│   └── streamlit_app.py       # Fase 5 — dashboard produk (4 halaman)
├── .streamlit/config.toml     # Tema & identitas visual aplikasi
├── src/
│   ├── common/                # Layer bersama lintas fase
│   │   ├── config.py          # Konfigurasi 12-factor (frozen dataclass)
│   │   ├── logging_utils.py   # Format log seragam
│   │   ├── spark_session.py   # Factory SparkSession + konfigurasi S3A
│   │   ├── storage.py         # Helper boto3 + pembaca Parquet untuk dashboard
│   │   ├── text_prep.py       # Pra-pemrosesan teks jalur inferensi
│   │   └── model_registry.py  # Pemuat artefak + API inferensi 4 head
│   └── phases/
│       ├── phase1_collect.py  # Ingestion + manifest checksum
│       ├── phase2_clean.py    # Sentinel handling, dedup, text pipeline
│       ├── phase3_explore.py  # 5 blok analitik Spark SQL mentah
│       ├── phase4_model.py    # Orkestrasi Fase 4 (Head A–D)
│       ├── genre_models.py    # Lapisan kompatibilitas -> Model/pipeline
│       └── dl_models.py       # Lapisan kompatibilitas -> Model/pipeline
├── scripts/
│   ├── run_pipeline.py        # Orchestrator Fase 1→4
│   ├── check.ipynb            # Notebook eksplorasi PySpark
│   ├── verify_env.py          # Smoke test infrastruktur
│   ├── verify_app.py          # Smoke test 4 halaman dashboard
│   └── research/              # Skrip eksperimen pendukung klaim di Doc/
│       ├── audit_labels.py    #   aturan tag→genre, deteksi label hilang
│       ├── audit_sentinel.py  #   pemulihan sentinel & release_frequency
│       ├── run_paired.py      #   uji berpasangan hybrid vs baseline (5 seed)
│       ├── ablate_headd.py    #   ablasi fitur kecepatan rilis
│       └── run_finetune.py    #   fine-tuning transformer (BELUM dijalankan)
├── docker/
│   ├── Dockerfile             # JRE 17 + PySpark 3.5.3 + hadoop-aws + BGE
│   ├── fetch_encoder.py       # Unduh bobot encoder saat build (retry)
│   └── requirements.txt
├── Dataset/data_novel.csv     # Sumber data
├── Model/                     # Paket pipeline machine learning
│   ├── train.py               # CLI pelatihan ulang model
│   ├── pipeline/
│   │   ├── config.py          # Seluruh hyperparameter & konstanta
│   │   ├── preprocessing.py   # Stemming, korpus, matriks label
│   │   ├── augmentation.py    # MLSMOTE + EDA random deletion
│   │   ├── features.py        # TF-IDF multi-blok + LIFT + blok embedding
│   │   ├── embeddings.py      # Encoder kalimat pralatih (BGE) + memo
│   │   ├── networks.py        # BiLSTM + Attention (PyTorch)
│   │   ├── losses.py          # Asymmetric Loss (Ridnik 2021)
│   │   ├── models.py          # Katalog arsitektur + pembungkus augmentasi
│   │   ├── evaluate.py        # F1 makro/mikro, P@k, Hamming, ambang
│   │   └── train.py           # Protokol OOF CV + perbandingan model
│   └── artifacts/             # Artefak terlatih (binary gitignored)
├── Doc/
│   ├── data_contract.md       # Skema, sentinel, aturan analitik
│   └── model_card.md          # Arsitektur model, metrik, batasan
└── docker-compose.yml
```

## Kontrak Data

Pipeline menggabungkan **dua sumber**:

| Berkas | Bentuk | Peran |
|---|---|---|
| `Dataset/data_novel.csv` | 4.461 × 27 | Metadata, rating, jumlah bab, tag |
| `Dataset/novel_dataset_scrapped.csv` | 5.000 × 4 | Sinopsis, digabung lewat `url` |

Join-nya mencakup 100% katalog (98,45% setelah menyaring sinopsis <10 kata).
Sebelum digabung, 538 baris `url` duplikat di berkas sinopsis harus dibuang —
tanpa itu tabel fakta menggelembung jadi 4.999 baris dan semua agregat salah.

Korpus pemodelan = `title` + `synopsis` + `tags`; rating dibaca dari
`rating_score` berbobot `rating_votes`. Detail skema, sentinel value, dan
aturan turunannya ada di [Doc/data_contract.md](Doc/data_contract.md) — wajib
dibaca sebelum menulis kode fase mana pun.

## Status Fase

| Fase | Nama | Keluaran | Status |
|---|---|---|---|
| 0 | Infrastruktur (Docker, MinIO, Spark, Jupyter, Streamlit) | 7 service | ✅ |
| 1 | Collect — ingest CSV ke `raw-data` | 4.461 baris + manifest checksum | ✅ |
| 2 | Clean — join sinopsis, sentinel handling, dedup, text preprocessing | 4.461 baris Parquet, 98,45% bersinopsis | ✅ |
| 3 | Explore — 5 blok analitik Spark SQL | 16 tabel di `curated-data/analytics/` | ✅ |
| 4 | Model — 4 model genre + 3 head lain | artefak di `Model/artifacts/` | ✅ |
| 5 | Deploy — dashboard produk NovelLens | 4 halaman Streamlit | ✅ |

### Hasil kunci

**Analitik (Fase 3)**

- Novel panjang **tidak** dinilai lebih tinggi: r(jumlah bab, rating tertimbang) = **−0,0875**.
  Rata-rata rating turun konsisten dari 3,55 (<50 bab) ke 3,45 (1000+ bab).
- Genre rating tertinggi: **Horror** (3,718 dari 75 novel); genre terpanjang: **Sci-fi** (rata-rata 643 bab).
- Deskripsi kaya berkorelasi dengan rating lebih baik: r(token, rating) = **+0,19**;
  kelompok deskripsi terkaya rata-rata 3,68 vs 3,37 pada kelompok termiskin.
- 9,61% novel adalah outlier panjang (>933 bab, metode IQR); terpanjang 10.850 bab.

**Model (Fase 4)** — detail di [Doc/model_card.md](Doc/model_card.md)

Head A menyediakan **empat model yang bisa dipilih pengguna** di dashboard —
dua baseline berbasis fitur, satu arsitektur hybrid, dan satu model deep
learning yang direplikasi dari literatur. Semuanya dilatih dan diuji pada
pembagian data identik (80% pengembangan / 20% uji; ambang dan hyperparameter
dipilih lewat OOF cross-validation):

| Model genre | F1 makro | F1 mikro | P@3 | Hamming ↓ |
|---|---|---|---|---|
| **Hybrid TF-IDF + Embedding (terpilih)** | **0,536** | **0,650** | **0,781** | **0,134** |
| Linear SVM | 0,527 | 0,648 | 0,765 | 0,137 |
| Logistic Regression | 0,517 | 0,638 | 0,755 | 0,141 |
| BiLSTM + Attention (deep learning) | 0,456 | 0,591 | 0,663 | 0,173 |

**Apa yang baru: blok embedding pralatih.** Versi sebelumnya menutup jalur
transformer dengan alasan "lingkungan ini offline". Premis itu diuji ulang dan
ternyata **salah** — model pralatih dapat diunduh, dan sekarang bobotnya
di-*bake* ke dalam image sehingga inferensi tetap tidak butuh jaringan.
Hasilnya, blok padat **BGE-small 384-d** disandingkan dengan TF-IDF.

Yang menarik: **sendirian, embedding itu kalah dari TF-IDF** (OOF makro 0,514
vs 0,547). Yang menolong bukan model yang lebih canggih, melainkan keragaman
representasi — TF-IDF unggul pada frasa khas ("cultivation"), embedding pada
parafrasa. Digabung, keduanya mencapai **0,566**.

Karena selisihnya kecil, keunggulannya diuji **berpasangan pada 5 pembagian
data** (sama data, sama protokol, hanya seed pembagian yang berbeda):

| Metrik | Linear SVM | Hybrid | Selisih | Menang |
|---|---|---|---|---|
| OOF makro | 0,5424 | **0,5588** | **+0,0164** | **5/5** |
| F1 makro uji | 0,5523 | **0,5642** | **+0,0119** | **5/5** |
| F1 mikro uji | 0,6430 | **0,6491** | **+0,0060** | **5/5** |
| P@3 uji | 0,7705 | **0,7811** | **+0,0106** | **5/5** |

Peningkatannya **konsisten tapi moderat** — 20 dari 20 perbandingan dimenangkan
hybrid, namun besarannya ~+0,012 makro-F1, bukan lompatan.

**Kenapa F1-nya tidak lebih tinggi?** Bukan modelnya, melainkan labelnya yang
tidak konsisten. Dua novel yang isinya nyaris identik (kemiripan konten 0,80–0,90)
hanya sepakat 0,73 mikro-F1 / 0,58 makro-F1 soal genrenya. Batas atas empiris
tugas ini ≈ **0,653 makro-F1**.

Hipotesis bahwa sebagian label sekadar *hilang* (tidak dituliskan kurator) juga
diuji memakai 62 aturan implikasi tag→genre: hasilnya hanya **+0,6%** label
tambahan pada 2,4% novel — terlalu kecil untuk menjelaskan jaraknya. Jadi
batas atas itu berasal dari label yang memang **subjektif**, bukan tidak
lengkap; menembusnya butuh pelabelan ulang, bukan pembersihan otomatis.
Rinciannya di [Doc/data_contract.md](Doc/data_contract.md) §5.

Belasan teknik dari literatur diuji — augmentasi (MLSMOTE, EDA), asymmetric
loss, stacking, fitur khusus per label — semuanya didokumentasikan lengkap
dengan alasan diterima/ditolak di [Doc/model_card.md](Doc/model_card.md).

> **Catatan reproduktibilitas.** Angka di atas lebih rendah daripada versi
> sebelumnya (SVM 0,545 → 0,527) **bukan karena modelnya memburuk**: pembagian
> data dulu bergantung pada urutan baris hasil Spark yang tidak deterministik,
> sehingga skor bergoyang sampai ±0,02 antar-eksekusi tanpa perubahan kode.
> Urutannya kini dikunci (`sort_values("series_id")`), dan angka lama kebetulan
> berasal dari pembagian yang menguntungkan.

| Head | Tugas | Metrik utama |
|---|---|---|
| B | Content-based recommender | Genre overlap@10 **0,939** |
| C | Klasterisasi arketipe | k=5, silhouette 0,017 |
| D | Estimasi rating dari konten | MAE **0,416** (baseline 0,454), R² **0,167** |

Head D naik dari R² 0,147 → **0,167** setelah fitur kecepatan rilis
ditambahkan. Diuji terpisah pada pembagian yang dikunci, sumbangan fitur itu
**campuran**: R² +0,016 tetapi MAE justru +0,005. Dipertahankan karena R²
paling relevan untuk pertanyaan "apa yang menjelaskan penerimaan pembaca",
tetapi keunggulannya tidak seragam.

Head C berpindah dari k=9 ke k=5 — bukan karena datanya berubah, melainkan
karena silhouette-nya rata (0,0153–0,0169 untuk k=5..10). **Pemilihan `k` pada
data ini praktis tidak informatif**; arketipe layak dipakai sebagai alat
eksplorasi, bukan segmentasi yang stabil.

## Verifikasi

```bash
docker compose exec pipeline python scripts/verify_env.py   # infrastruktur
docker compose exec streamlit python scripts/verify_app.py  # 4 halaman dashboard
```

## Troubleshooting

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `verify_env.py` gagal di cek MinIO | `minio-init` belum jalan | `docker compose up -d minio-init` |
| Job Spark menggantung di stage 0 | `spark.driver.host` tidak sesuai hostname container | Pastikan `SPARK_DRIVER_HOST` diset di service driver |
| App Spark berstatus `WAITING` selamanya | Tidak ada worker hidup — cek `curl -s localhost:8080/json/ \| grep aliveworkers` | Lihat `docker compose logs spark-worker`; worker butuh `SPARK_WORKER_DIR` di path yang writable (`/tmp/spark-work`) |
| `403 Forbidden` saat baca `s3a://` | Kredensial `.env` tidak sinkron dengan volume MinIO lama | `docker compose down -v` lalu `up -d` |
| Port bentrok saat `up` | Port 9000/8080/8501 terpakai proses lain | Ubah `*_PORT` di `.env` |
| Fase 4 menggantung tanpa error | Deadlock thread OpenBLAS/OpenMP di dalam container | Sudah dicegah lewat `ENV *_NUM_THREADS=2` di Dockerfile — pastikan container dibuat ulang (`docker compose up -d --force-recreate`) setelah rebuild |
| Dashboard bilang tabel tidak ditemukan | Fase 3 belum dijalankan | `docker compose exec pipeline python scripts/run_pipeline.py --from 2` |
| AI Playground kosong | Artefak Fase 4 belum ada | `docker compose exec pipeline python -m src.phases.phase4_model` |
| Model `hybrid` gagal dimuat / `OSError` saat inferensi | Bobot encoder BGE belum ada di image (image lama) | `docker compose build && docker compose up -d --force-recreate` — Dockerfile mem-*bake* bobotnya ke `/opt/hf-cache` |
| Fase 4 lama sekali di tahap `[hybrid]` | Encoding 4.460 dokumen dibatasi `OMP_NUM_THREADS=2` | Naikkan `ENCODER_THREADS` di `.env` (mis. 8) — hanya PyTorch yang terpengaruh |
