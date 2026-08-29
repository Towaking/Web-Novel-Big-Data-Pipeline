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
| `streamlit` | Dashboard BI & AI Playground | 8501 |

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

- MinIO Console → <http://localhost:9001> (user/pass dari `.env`)
- Spark Master UI → <http://localhost:8080>
- Streamlit → <http://localhost:8501>

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
│       ├── phase4_model.py    # Training model multi-head
│       └── genre_models.py    # 3 arsitektur genre + penyetelan ambang
├── scripts/
│   ├── run_pipeline.py        # Orchestrator Fase 1→4
│   ├── verify_env.py          # Smoke test infrastruktur
│   └── verify_app.py          # Smoke test 4 halaman dashboard
├── docker/
│   ├── Dockerfile             # JRE 17 + PySpark 3.5.3 + hadoop-aws
│   └── requirements.txt
├── Dataset/data_novel.csv     # Sumber data
├── Model/artifacts/           # Artefak model Fase 4 (gitignored)
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
| 0 | Infrastruktur (Docker, MinIO, Spark, Streamlit) | 6 service | ✅ |
| 1 | Collect — ingest CSV ke `raw-data` | 4.461 baris + manifest checksum | ✅ |
| 2 | Clean — sentinel handling, dedup, text preprocessing | 4.461 baris Parquet, 4.439 bersinyal teks | ✅ |
| 3 | Explore — 5 blok analitik Spark SQL | 15 tabel di `curated-data/analytics/` | ✅ |
| 4 | Model — arsitektur AI multi-head | 9 artefak di `Model/artifacts/` | ✅ |
| 5 | Deploy — dashboard BI + AI Playground | 4 halaman Streamlit | ✅ |

### Hasil kunci

**Analitik (Fase 3)**

- Novel panjang **tidak** dinilai lebih tinggi: r(jumlah bab, rating tertimbang) = **−0,0875**.
  Rata-rata rating turun konsisten dari 3,55 (<50 bab) ke 3,45 (1000+ bab).
- Genre rating tertinggi: **Horror** (3,718 dari 75 novel); genre terpanjang: **Sci-fi** (rata-rata 643 bab).
- Deskripsi kaya berkorelasi dengan rating lebih baik: r(token, rating) = **+0,19**;
  kelompok deskripsi terkaya rata-rata 3,68 vs 3,37 pada kelompok termiskin.
- 9,61% novel adalah outlier panjang (>933 bab, metode IQR); terpanjang 10.850 bab.

**Model (Fase 4)** — detail di [Doc/model_card.md](Doc/model_card.md)

Head A menyediakan **tiga model yang bisa dipilih pengguna** di dashboard,
dilatih dan diuji pada pembagian data identik (80% pengembangan / 20% uji,
ambang & hyperparameter dipilih lewat OOF cross-validation):

| Model genre | F1 makro | F1 mikro | F1 makro bila ambang 0,5 |
|---|---|---|---|
| Linear SVM (terpilih) | **0,538** | **0,637** | 0,410 |
| Logistic Regression | 0,532 | 0,633 | 0,450 |
| Neural Network (MLP) | 0,507 | 0,618 | 0,429 |

| Head | Tugas | Metrik utama |
|---|---|---|
| B | Content-based recommender | Genre overlap@10 **0,919** |
| C | Klasterisasi arketipe | k=9, silhouette 0,019 |
| D | Estimasi rating dari konten | MAE **0,411** (baseline 0,443), R² 0,147 |

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
