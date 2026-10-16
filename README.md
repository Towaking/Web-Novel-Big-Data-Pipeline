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

| Service          | Peran                                          | Port host                                                                       |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| `minio`        | Data lake S3-compatible                        | 9000 (API), 9001 (Console)                                                      |
| `minio-init`   | Bootstrap bucket (sekali jalan, idempotent)    | —                                                                              |
| `spark-master` | Koordinator cluster                            | 7080 (UI), 7077                                                                 |
| `spark-worker` | Executor, bisa di-scale                        | 7081–7085 (UI, dialokasikan otomatis dari rentang — cek`docker compose ps`) |
| `pipeline`     | Driver container tempat job Fase 1–4 disubmit | —                                                                              |
| `notebook`     | JupyterLab dengan driver Spark sendiri         | 8888                                                                            |
| `streamlit`    | Dashboard produk (BI + AI Studio)              | 8501                                                                            |
| `namenode`     | HDFS — metadata & koordinator blok             | 9870 (UI)                                                                       |
| `worker1/2`    | HDFS DataNode + YARN NodeManager (satu hostname — syarat data locality) | — |
| `resourcemanager` | YARN — penjadwal job MapReduce              | 18088 (UI)                                                                      |
| `hadoop-client`   | Klien `hdfs`/`hadoop jar` (repo di-bind-mount) | —                                                                            |

Seluruh compute service Spark memakai **satu image yang sama** (`bigdata-novel:base`)
agar versi Python & PySpark identik antara driver dan executor — syarat wajib
supaya serialisasi UDF PySpark tidak gagal. Node Hadoop memakai `bigdata-novel:hadoop`
(image resmi `apache/hadoop:3.3.6` + Python 3, lihat [docker/Dockerfile.hadoop](docker/Dockerfile.hadoop) —
basisnya CentOS 7 EOL tanpa Python 3 dan mirror `yum` bawaannya sudah mati).

> **Port non-standar (7080/7081-7085/18088), bukan salah ketik.** Di Windows,
> Hyper-V/WSL mereservasi rentang port dinamis yang **berputar setiap boot**
> (`netsh interface ipv4 show excludedportrange protocol=tcp`). Rentang itu
> pernah menelan 8080-8096 sekaligus, membuat Spark UI dan port worker gagal
> bind dengan pesan yang tidak jelas ("access forbidden by its permissions").
> Bila port ini bentrok lagi di mesin Anda, cek ulang perintah itu dan
> sesuaikan `SPARK_MASTER_UI_PORT` / `SPARK_WORKER_UI_PORT(_END)` di `.env`.

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

Lalu buka dashboard di [http://localhost:8501](http://localhost:8501).

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

- Dashboard NovelLens → [http://localhost:8501](http://localhost:8501)
- JupyterLab (eksplorasi PySpark) → [http://localhost:8888](http://localhost:8888) — notebook siap pakai di [scripts/notebook/check.ipynb](scripts/notebook/check.ipynb) (eksplorasi Spark SQL bebas) dan [scripts/notebook/eda_visualisasi.ipynb](scripts/notebook/eda_visualisasi.ipynb) (EDA sistematis + visualisasi)
- Spark Master UI → [http://localhost:8080](http://localhost:8080)
- MinIO Console → [http://localhost:9001](http://localhost:9001) (user/pass dari `.env`)

## Melatih Ulang Model

Skrip [Model/train.py](Model/train.py) melatih ulang model tanpa menjalankan
seluruh pipeline. Tanpa `--models`, Head A melatih model ML klasik yang disajikan
AI Studio: `logreg`, `svm`, `rf`, dan `ensemble` (rata-rata log-odds berbobot svm + rf).
Arsitektur yang diminta lewat `--models` digabung dengan artefak lama, jadi
melatih satu model tidak menghapus yang lain; ensemble lama dibuang bila salah
satu anggotanya dilatih ulang sendirian.

```bash
docker compose exec -e PHASE4_N_JOBS=10 pipeline python Model/train.py   # semua head, 10 worker
docker compose exec pipeline python Model/train.py --only genre           # Head A saja
docker compose exec pipeline python Model/train.py --models svm           # satu arsitektur
docker compose restart streamlit                                          # muat artefak baru
```

`PHASE4_N_JOBS` (bawaan 2) mengatur worker latih paralel; dengan 10 worker Fase 4
penuh selesai sekitar 15 menit.

Arsitektur di katalog: `logreg`, `svm`, `rf`, `ensemble`, serta `hybrid` dan
`bilstm` (model neural untuk riset; hanya dilatih bila diminta lewat `--models`
atau `PHASE4_MODELS`, dan tidak dipakai aplikasi).

Menambah kapasitas worker:

```bash
docker compose up -d --scale spark-worker=3
```

## Struktur Proyek

```text
.
├── app/
│   └── streamlit_app.py       # Fase 5 — dashboard: 4 halaman produk + 3 halaman data platform
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
│       ├── data_catalog.py    # Inventaris elemen + klasifikasi format
│       ├── data_model.py      # Star schema + bridge + proyeksi graf
│       ├── governance_audit.py# 21 cek security/compliance/privacy/quality
│       ├── bigdata_audit.py   # Audit konsep Big Data (5V, DFS, MapReduce)
│       ├── data_variety.py    # Pisahkan zona clean -> CSV/JSONL/TXT
│       ├── streaming_ingest.py# Velocity - Structured Streaming inkremental
│       ├── graph_analysis.py  # PageRank, komunitas, traversal (NetworkX)
│       ├── genre_models.py    # Lapisan kompatibilitas -> Model/pipeline
│       └── dl_models.py       # Lapisan kompatibilitas -> Model/pipeline
│   └── mapreduce/              # Mapper/reducer Hadoop Streaming (Python 3, stdlib saja)
│       ├── wordcount_mapper.py #   Job A: frekuensi kata (TXT tidak terstruktur)
│       ├── genre_mapper.py     #   Job B: frekuensi + ko-okurensi genre (JSONL)
│       ├── origin_mapper.py    #   Job C: statistik per negara asal (CSV)
│       ├── join_mapper.py / join_reducer.py  # Job D: reduce-side join CSV x JSONL
│       └── sum_reducer.py      #   reducer/combiner generik (penjumlah vektor)
├── scripts/
│   ├── run_pipeline.py        # Orchestrator Fase 1→4 + variety + tata kelola
│   ├── notebook/
│   │   ├── check.ipynb            # Notebook eksplorasi Spark SQL bebas
│   │   ├── eda_visualisasi.ipynb  # EDA sistematis + visualisasi (matplotlib/wordcloud)
│   │   ├── model_training.ipynb   # Latih Fase 4 head demi head, interaktif
│   │   ├── web_scrap_1.ipynb      # Eksperimen scraping (draft)
│   │   └── web_scrap_2.ipynb      # Eksperimen scraping (draft)
│   ├── scraper/                # Scraper NovelUpdates (lihat Dataset/scraped/LANJUTKAN.md)
│   ├── hdfs/load_hdfs.sh      # Muat Dataset/variety/ ke HDFS + fsck
│   ├── mapreduce/
│   │   ├── run_jobs.sh            # Serahkan 4 job Hadoop Streaming ke YARN
│   │   ├── local_smoke.sh         # Job yang sama tanpa Hadoop (pipa Unix)
│   │   └── verify_mapreduce.py    # Cocokkan hasil MapReduce vs Spark SQL
│   ├── streaming/simulate_landing.py  # Harness demo: simulasikan batch baru untuk streaming_ingest.py
│   ├── verify_env.py          # Smoke test infrastruktur
│   ├── verify_app.py          # Smoke test 7 halaman dashboard
│   └── research/              # Skrip eksperimen pendukung klaim di Doc/
│       ├── audit_labels.py    #   aturan tag→genre, deteksi label hilang
│       ├── audit_sentinel.py  #   pemulihan sentinel & release_frequency
│       ├── run_paired.py      #   uji berpasangan hybrid vs baseline (5 seed)
│       ├── ablate_headd.py    #   ablasi fitur kecepatan rilis
│       └── run_finetune.py    #   fine-tuning transformer (BELUM dijalankan)
├── docker/
│   ├── Dockerfile             # JRE 17 + PySpark 3.5.3 + hadoop-aws + BGE
│   ├── Dockerfile.hadoop      # apache/hadoop:3.3.6 + Python 3 (untuk mapper/reducer)
│   ├── hadoop.env             # Konfigurasi HDFS/YARN (core/hdfs/mapred/yarn-site.xml)
│   ├── fetch_encoder.py       # Unduh bobot encoder saat build (retry)
│   └── requirements.txt
├── Dataset/data_novel.csv     # Sumber data
├── Dataset/variety/            # Hasil pemisahan format (CSV/JSONL/TXT), lihat manifest.json
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
│   ├── model_card.md          # Arsitektur model, metrik, batasan
│   └── generated/             # Dihasilkan mesin — jangan diedit manual
│       ├── data_elements.md   #   inventaris 28 elemen + kelas format
│       ├── governance_audit.md#   hasil 21 pemeriksaan tata kelola
│       ├── bigdata_concept_audit.md#  status tiap konsep Big Data (5V, DFS, MapReduce)
│       └── mapreduce_verification.md# hasil 4 job MapReduce vs jawaban Spark
└── docker-compose.yml
```

## Variety, HDFS & MapReduce

Zona clean (39.668 novel, `s3a://clean-data/novels_clean`) sudah menyimpan
ketiga kelas format tata kelola (`data_catalog.py`) dalam satu Parquet — untuk
benar-benar mendapatkan *variety*, ketiganya juga diwujudkan sebagai **berkas
fisik terpisah dengan format yang cocok untuk sifatnya**, lalu dimuat ke HDFS
dan diproses dengan MapReduce murni (bukan Spark) sebagai jalur kedua yang
independen:

| Kelas            | Berkas                                     | Format | Baris/berkas |
| ---------------- | ------------------------------------------- | ------ | ------------- |
| Terstruktur      | `Dataset/variety/structured/novel_metrics.csv`      | CSV    | 39.668 |
| Semi-terstruktur | `Dataset/variety/semistructured/novel_labels.jsonl` | JSONL  | 39.668 (genre/tag/author sebagai array, bukan kolom datar) |
| Tidak terstruktur| `Dataset/variety/unstructured/synopsis/synopsis-*.txt` | TXT (4 shard) | 39.342 (hanya yang bersinopsis) |

```bash
docker compose exec pipeline python -m src.phases.data_variety   # ekspor + verifikasi baca-ulang
docker compose exec hadoop-client bash scripts/hdfs/load_hdfs.sh # muat ke HDFS (blok 4 MB, replikasi 2) + fsck
docker compose exec hadoop-client bash scripts/mapreduce/run_jobs.sh # 4 job Hadoop Streaming di YARN
docker compose exec pipeline python scripts/mapreduce/verify_mapreduce.py # cocokkan dgn Spark SQL
```

Empat job MapReduce (mapper/reducer Python murni di [src/mapreduce/](src/mapreduce/)):

| Job | Input | Pertanyaan |
| --- | --- | --- |
| `wordcount` | TXT | Frekuensi kata di korpus sinopsis |
| `genre_stats` | JSONL | Frekuensi genre + ko-okurensi genre berpasangan |
| `origin_stats` | CSV | Cacah/rating/bab per negara asal (`type_origin`) |
| `genre_rating` | CSV × JSONL | Rata-rata weighted rating per genre — **reduce-side join** dua format berbeda, dua tahap (`join_stage1` lalu `genre_rating`) |

Tiap job diverifikasi dengan menjawab pertanyaan yang sama lewat Spark SQL dari
sumber Parquet-nya dan membandingkan baris demi baris — hasil terbaru: **4 dari
4 job identik** dengan Spark (0 selisih baris/nilai). Detail lengkap termasuk
counter Hadoop (task, combiner, shuffle) ada di
[Doc/generated/mapreduce_verification.md](Doc/generated/mapreduce_verification.md).

HDFS berjalan sebagai klaster nyata (NameNode + 2 worker, replikasi 2) yang
**berdampingan** dengan data lake MinIO, bukan menggantikannya — MinIO tetap
menjadi *system of record* Fase 1–4 (versioning, dibaca Spark & dashboard),
sementara HDFS menyimpan turunan zona clean yang diproses MapReduce/YARN.
Data-nya bertahan lintas `docker compose up --force-recreate` lewat named
volume (`novel-namenode-data`, `novel-datanode{1,2}-data`).

**Data locality nyata, bukan cuma sebaran replika.** `worker1`/`worker2`
menjalankan DataNode **dan** NodeManager dalam satu container/hostname yang
sama (`docker/Dockerfile.hadoop` + `docker-compose.yml`), sehingga YARN bisa
menjadwalkan task map persis di NodeManager yang disk-nya memegang blok
tersebut. Dibuktikan dari counter Hadoop asli di log job, bukan diklaim dari
topologi: **100% task map data-local** (0% rack-local) pada keempat job.

### Velocity — Structured Streaming inkremental

Seluruh Fase 1–4 sengaja batch (determinisme > kecepatan) — tapi itu berarti
proyek ini tidak punya jalur *velocity* sama sekali, padahal sumbernya
(scraper NovelUpdates) sebenarnya berdatangan bertahap. `src/phases/streaming_ingest.py`
menutup celah itu: Spark Structured Streaming (file source, trigger
`availableNow`, checkpoint di MinIO) yang memproses berkas baru di folder
*landing* satu micro-batch per berkas, lalu berhenti sendiri.

```bash
docker compose exec pipeline python scripts/streaming/simulate_landing.py --offset 0 --rows 500  # simulasikan batch baru
docker compose exec pipeline python -m src.phases.streaming_ingest                                # proses berkas baru
docker compose exec pipeline python -m src.phases.streaming_ingest --reset                        # ulangi demo dari nol
```

Sifat *exactly-once*-nya **diverifikasi langsung**, bukan diasumsikan dari
dokumentasi Spark: batch A (500 baris) diproses jadi 500 baris di tabel;
+batch B (500 baris) diproses jadi total 1.000 baris; mengulang job tanpa
batch baru menghasilkan **0 baris baru** — checkpoint mencegah pemrosesan
ulang, persis definisi ingest inkremental.

### Graph Analysis — PageRank, komunitas, traversal

`data_model.py` memproyeksikan katalog jadi property graph (67.807 simpul,
860.109 sisi: Novel–Genre–Tag–Author) dan dua "operasi graf" berbasis
self-join (`op_graph_genre_cooccurrence`, `op_graph_author_reach`) — tapi
self-join cuma menghitung ko-kemunculan, bukan menjawab pertanyaan yang
BUTUH algoritma graf sungguhan. `src/phases/graph_analysis.py` mengisi celah
itu dengan NetworkX di atas dua proyeksi:

```bash
docker compose exec pipeline python -m src.phases.graph_analysis
```

| Graf | Simpul/Sisi | Temuan |
| --- | --- | --- |
| Ko-okurensi genre | 35 / 579 | 1 komponen (menyatu penuh), 3 komunitas, **modularity rendah (0,086)** — genre saling berbaur, tidak terbelah tajam |
| Kolaborasi penulis (co-authorship nyata, bukan proksi genre) | 2.578 / 2.699 | **931 komponen terpisah**, terbesar 22 penulis, **modularity 0,995** — tim penulis nyaris tidak pernah lintas-kolaborasi |

PageRank membedakan pengaruh genre dari sekadar frekuensi (Romance/Fantasy
teratas karena bertetangga dengan genre lain yang juga berpengaruh, bukan
cuma karena sering muncul). Traversal BFS **mengukur, bukan mengasumsikan**,
klaim "<=2 hop" yang jadi alasan `data_model.py` tidak memakai database graf
terpisah — contoh nyata: `Yaoi -> Romance -> Yuri` (dua genre yang tak pernah
disematkan langsung, tapi terhubung lewat satu genre perantara). Visualisasi
lengkap (graf berwarna komunitas, distribusi komponen, bar PageRank) ada di
`scripts/notebook/eda_visualisasi.ipynb` §9.

### Audit konsep Big Data

```bash
docker compose exec pipeline python -m src.phases.bigdata_audit
```

Memeriksa stack yang **sedang hidup** (bukan daftar klaim) untuk tiap konsep —
5V, penyimpanan/pemrosesan terdistribusi, graf, dan keandalan — lalu
melaporkan `TERPENUHI` / `SEBAGIAN` / `BELUM` dengan bukti terukur. Hasil
terbaru: **24 terpenuhi, 2 sebagian, 0 belum, 2 informasi**. Dua yang tersisa
**SEBAGIAN** memang tidak bisa ditutup dengan kode lagi:

- **Volume (skala data)** — 39.668 baris novel masih muat di satu mesin;
  arsitekturnya terdistribusi (Spark 2 worker, HDFS 2 DataNode), tapi datanya
  sendiri belum "big" secara ukuran. Satu-satunya V yang jujur tidak bisa
  dipalsukan lewat kode.
- **Variety (sumber heterogen saat ingest)** — seluruh sumber luar (scraper,
  CSV pendamping) berakhir sebagai CSV sebelum masuk Fase 1; format
  JSON/TXT baru lahir SESUDAH Fase 2. Menunggu keputusan dataset kedua yang
  betul-betul non-tabular (graf) yang belum diambil.

Rincian lengkap tiap butir (termasuk yang sudah TERPENUHI beserta buktinya):
[Doc/generated/bigdata_concept_audit.md](Doc/generated/bigdata_concept_audit.md).

## Kontrak Data

Pipeline menggabungkan **dua sumber**:

| Berkas                                 | Bentuk      | Peran                             |
| -------------------------------------- | ----------- | --------------------------------- |
| `Dataset/data_novel.csv`             | 4.461 × 27 | Metadata, rating, jumlah bab, tag |
| `Dataset/novel_dataset_scrapped.csv` | 5.000 × 4  | Sinopsis, digabung lewat`url`   |

Join-nya mencakup 100% katalog (98,45% setelah menyaring sinopsis <10 kata).
Sebelum digabung, 538 baris `url` duplikat di berkas sinopsis harus dibuang —
tanpa itu tabel fakta menggelembung jadi 4.999 baris dan semua agregat salah.

Korpus pemodelan = `title` + `synopsis` + `tags`; rating dibaca dari
`rating_score` berbobot `rating_votes`. Detail skema, sentinel value, dan
aturan turunannya ada di [Doc/data_contract.md](Doc/data_contract.md) — wajib
dibaca sebelum menulis kode fase mana pun.

## Status Fase

| Fase | Nama                                                                 | Keluaran                                | Status |
| ---- | -------------------------------------------------------------------- | --------------------------------------- | ------ |
| 0    | Infrastruktur (Docker, MinIO, Spark, Jupyter, Streamlit)             | 7 service                               | ✅     |
| 1    | Collect — ingest CSV ke`raw-data`                                 | 4.461 baris + manifest checksum         | ✅     |
| 2    | Clean — join sinopsis, sentinel handling, dedup, text preprocessing | 4.461 baris Parquet, 98,45% bersinopsis | ✅     |
| 3    | Explore — 5 blok analitik Spark SQL                                 | 16 tabel di`curated-data/analytics/`  | ✅     |
| 4    | Model: 2 model genre ML klasik + 3 head lain                         | artefak di`Model/artifacts/`          | ✅     |
| 5    | Deploy — dashboard NovelLens (produk + data platform: warehouse OLAP, graf, tata kelola) | 7 halaman Streamlit | ✅     |
| —   | Tata kelola — inventaris elemen, model data, audit                  | 17 tabel + 2 dokumen                    | ✅     |
| —   | Big Data — variety, HDFS, MapReduce, Structured Streaming, analisis graf, audit konsep | 4 job MapReduce (4/4 cocok Spark, 100% data-local), 24/26 konsep terpenuhi | ✅     |

Lapisan tata kelola bukan fase berurutan: ia memotong seluruh fase (memprofil
zona raw, memproyeksikan zona clean, memeriksa ketiganya). Karena itu tidak
diberi nomor — `Fase 5` sudah dipakai dashboard.

```bash
docker compose exec pipeline python scripts/run_pipeline.py --only-governance
docker compose exec pipeline python -m src.phases.data_catalog      # satu langkah saja
```

### Hasil kunci

**Analitik (Fase 3)**

- Novel panjang **tidak** dinilai lebih tinggi: r(jumlah bab, rating tertimbang) = **−0,0875**.
  Rata-rata rating turun konsisten dari 3,55 (<50 bab) ke 3,45 (1000+ bab).
- Genre rating tertinggi: **Horror** (3,718 dari 75 novel); genre terpanjang: **Sci-fi** (rata-rata 643 bab).
- Deskripsi kaya berkorelasi dengan rating lebih baik: r(token, rating) = **+0,19**;
  kelompok deskripsi terkaya rata-rata 3,68 vs 3,37 pada kelompok termiskin.
- 9,61% novel adalah outlier panjang (>933 bab, metode IQR); terpanjang 10.850 bab.

**Tata kelola data**

- **Inventaris elemen** — 28 elemen terklasifikasi: **20 terstruktur, 6
  semi-terstruktur, 2 tidak terstruktur**. Kelasnya dideklarasikan eksplisit
  (dengan alasannya) lalu statistiknya diukur dari data; tipe fisik tidak
  dipakai menebak, karena `genres` (daftar berkoma) dan `release_frequency`
  (`"Every 0.9 Day(s)"`) sama-sama `string` tapi berbeda kelas.
- **Model data** — star schema (`fact_novel` grain per novel + 6 dimensi +
  3 bridge) berdampingan dengan proyeksi graf (9.334 simpul, 77.493 sisi).
  Dua model karena dua kelas pertanyaan: roll-up OLAP vs traversal relasi.
- **Resolusi entitas penulis** — `authors` ternyata bukan selalu daftar
  beberapa penulis: pada **1.494 novel (34,6%)** isinya satu orang dalam dua
  aksara (`"Mizume, ミズメ"`). Tanpa penanganan, `dim_author` menggelembung
  **5.439** entitas untuk 4.461 novel dan setiap analitik penulis berhitung
  ganda. Setelah resolusi: **4.081** entitas (2.624 nama → 1.266 orang);
  36 komponen ambigu sengaja **tidak** digabung dan ditandai.
- **Audit tata kelola** — 21 pemeriksaan berjalan: **14 lulus, 2 peringatan,
  0 gagal**. Peringatan yang tersisa (kredensial bawaan, TLS) memang diterima
  untuk lingkungan lokal dan tercatat sebagai tindak lanjut, bukan disembunyikan.
  Kontrol yang benar-benar dipasang: versioning aktif di ketiga bucket, akses
  anonim ditutup, dan akun read-only untuk dashboard — **diuji nyata**, tulis
  dari service streamlit ditolak `AccessDenied`.
- **Privasi** — 0 kecocokan PII langsung di seluruh kolom teks, dan tidak ada
  satu pun kolom ber-*grain* pengguna: sinyal pembaca hanya tersedia sebagai
  agregat (`rating_votes`, `reading_list_count`).

Detail terukur: [Doc/generated/data_elements.md](Doc/generated/data_elements.md)
dan [Doc/generated/governance_audit.md](Doc/generated/governance_audit.md)
(keduanya dihasilkan mesin — jangan diedit manual).

**Model (Fase 4)** — detail di [Doc/model_card.md](Doc/model_card.md)

Head A di dashboard menyediakan **tiga model ML klasik dan satu ensemble** yang
bisa dipilih pengguna, semuanya di atas fitur TF-IDF tanpa jaringan saraf. Model
bawaan dipilih dari F1 makro OOF. Angka berikut dari katalog 39.660 judul, 35
genre, 7.932 judul uji (80% pengembangan / 20% uji; ambang dan hyperparameter
dipilih lewat OOF cross-validation), setelah revisi 2026-10-15
([Doc/peningkatan_model.md](Doc/peningkatan_model.md)):

| Model genre                                    | F1 makro OOF    | F1 makro uji    | F1 mikro        | P@3             | Hamming ↓      |
| ---------------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **Ensemble SVM + Random Forest (bawaan)**      | **0,606**       | **0,612**       | **0,674**       | 0,801           | 0,106           |
| Linear SVM                                     | 0,602           | 0,609           | 0,672           | **0,802**       | **0,106**       |
| Logistic Regression                            | 0,597           | 0,603           | 0,667           | 0,795           | 0,109           |
| Random Forest                                  | 0,549           | 0,552           | 0,630           | 0,741           | 0,124           |
| Linear SVM sebelum revisi                      | 0,583           | 0,591           | 0,654           | 0,788           | 0,114           |
| Hybrid BGE (fitur lama, tidak disajikan)       | 0,591           | 0,597           | 0,657           | 0,792           | 0,113           |
| BiLSTM (fitur lama, tidak disajikan)           | 0,542           | 0,550           | 0,629           | 0,682           | 0,125           |

Kenaikan terbesar datang dari penyetelan ulang regularisasi dan bobot blok fitur
(TPE), bukan dari model yang lebih rumit: model ML klasik kini melampaui Hybrid
dengan encoder BGE. Diuji pada lima pembagian data, ensemble unggul +0,023 F1
makro atas SVM lama di kelima pembagian. Random Forest lebih lemah sendirian,
tetapi kesalahannya berbeda dari SVM sehingga gabungan keduanya menjadi yang
terbaik. Penggabungannya di ruang log-odds: rata-rata probabilitas biasa
menurunkan P@3 secara bermakna, sedangkan log-odds hanya -0,001 (tidak
bermakna) dengan P@1 tertinggi (0,904).

**Riset yang tidak dipakai aplikasi: blok embedding pralatih.** Uji berikut
dijalankan pada katalog lama 4.460 judul. Versi sebelumnya menutup jalur
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

| Metrik       | Linear SVM | Hybrid           | Selisih           | Menang        |
| ------------ | ---------- | ---------------- | ----------------- | ------------- |
| OOF makro    | 0,5424     | **0,5588** | **+0,0164** | **5/5** |
| F1 makro uji | 0,5523     | **0,5642** | **+0,0119** | **5/5** |
| F1 mikro uji | 0,6430     | **0,6491** | **+0,0060** | **5/5** |
| P@3 uji      | 0,7705     | **0,7811** | **+0,0106** | **5/5** |

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

| Head | Tugas                       | Metrik utama                                             |
| ---- | --------------------------- | -------------------------------------------------------- |
| B    | Content-based recommender   | Genre overlap@10 **0,949**                               |
| C    | Klasterisasi arketipe       | k=7, silhouette 0,029                                    |
| D    | Estimasi rating dari konten | MAE **0,325** (tebak rata-rata 0,415), R² **0,354**      |

Head D (revisi 2026-10-15) menumpuk skor ridge teks out-of-fold, tag multi-hot,
dan bobot sampel menurut jumlah suara di atas HistGradientBoosting yang disetel
dengan TPE. Konfigurasi lama pada baris latih yang sama: MAE 0,340, R² 0,307.
Keduanya dievaluasi dengan TF-IDF dan SVD yang di-fit tanpa baris uji.

Head C berpindah dari k=9 ke k=5 — bukan karena datanya berubah, melainkan
karena silhouette-nya rata (0,0153–0,0169 untuk k=5..10). **Pemilihan `k` pada
data ini praktis tidak informatif**; arketipe layak dipakai sebagai alat
eksplorasi, bukan segmentasi yang stabil.

## Verifikasi

```bash
docker compose exec pipeline python scripts/verify_env.py   # infrastruktur
docker compose exec streamlit python scripts/verify_app.py  # 7 halaman dashboard
```

## Troubleshooting

| Gejala                                                    | Penyebab umum                                                                     | Solusi                                                                                                                                                  |
| --------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verify_env.py` gagal di cek MinIO                      | `minio-init` belum jalan                                                        | `docker compose up -d minio-init`                                                                                                                     |
| Job Spark menggantung di stage 0                          | `spark.driver.host` tidak sesuai hostname container                             | Pastikan`SPARK_DRIVER_HOST` diset di service driver                                                                                                   |
| App Spark berstatus`WAITING` selamanya                  | Tidak ada worker hidup — cek`curl -s localhost:8080/json/ \| grep aliveworkers` | Lihat`docker compose logs spark-worker`; worker butuh `SPARK_WORKER_DIR` di path yang writable (`/tmp/spark-work`)                                |
| `403 Forbidden` saat baca `s3a://`                    | Kredensial`.env` tidak sinkron dengan volume MinIO lama                         | `docker compose down -v` lalu `up -d`                                                                                                               |
| Port bentrok saat`up`                                   | Port 9000/8080/8501 terpakai proses lain                                          | Ubah`*_PORT` di `.env`                                                                                                                              |
| Fase 4 menggantung tanpa error                            | Deadlock thread OpenBLAS/OpenMP di dalam container                                | Sudah dicegah lewat`ENV *_NUM_THREADS=2` di Dockerfile — pastikan container dibuat ulang (`docker compose up -d --force-recreate`) setelah rebuild |
| Dashboard bilang tabel tidak ditemukan                    | Fase 3 belum dijalankan                                                           | `docker compose exec pipeline python scripts/run_pipeline.py --from 2`                                                                                |
| AI Playground kosong                                      | Artefak Fase 4 belum ada                                                          | `docker compose exec pipeline python -m src.phases.phase4_model`                                                                                      |
| Model`hybrid` gagal dimuat / `OSError` saat inferensi | Bobot encoder BGE belum ada di image (image lama)                                 | `docker compose build && docker compose up -d --force-recreate` — Dockerfile mem-*bake* bobotnya ke `/opt/hf-cache`                              |
| Fase 4 lama sekali di tahap`[hybrid]`                   | Encoding 4.460 dokumen dibatasi`OMP_NUM_THREADS=2`                              | Naikkan`ENCODER_THREADS` di `.env` (mis. 8) — hanya PyTorch yang terpengaruh                                                                       |
| Port `docker compose up` gagal bind dengan "access forbidden by its permissions" (khusus Windows) | Hyper-V/WSL mereservasi rentang port dinamis yang **berputar tiap boot** | `netsh interface ipv4 show excludedportrange protocol=tcp`, pilih port di luar rentang yang tampil, set di `.env` |
| `run_jobs.sh`: "Name node is in safe mode" | NameNode baru start, belum menerima cukup laporan blok dari DataNode | Transien (~beberapa detik) — skrip sudah menunggu otomatis; ulangi manual dengan `hdfs dfsadmin -safemode get` bila masih lama |
| NameNode `InconsistentFSStateException` / `Cannot create directory .../current` setelah container dibuat ulang | Named volume HDFS dibuat root:root, atau berisi sisa format yang gagal | `docker compose up -d hdfs-permissions` dulu (chown), atau `docker volume rm novel-namenode-data novel-datanode{1,2}-data` untuk mulai bersih |
| `hadoop jar` gagal `Cannot run program "python3"` | NodeManager memakai `apache/hadoop:3.3.6` resmi (CentOS 7 EOL, tanpa Python 3, mirror `yum` bawaan sudah mati) | Pastikan compose memakai `bigdata-novel:hadoop` ([docker/Dockerfile.hadoop](docker/Dockerfile.hadoop)), bukan `apache/hadoop:3.3.6` langsung — `docker compose up -d --build` |
| `docker exec`/`compose exec` ke path Unix (`/opt/project`) berubah jadi `C:/Program Files/Git/...` (Git Bash Windows) | MSYS/Git Bash mengonversi argumen mirip path Unix ke path Windows | `export MSYS_NO_PATHCONV=1` sebelum perintah `docker`/`docker compose exec` |
| NodeManager `worker1/2` berstatus `UNHEALTHY` di YARN, `healthReport` menyebut `nm-local-dir: Permission denied` | `yarn.nodemanager.local-dirs` default ada di bawah `/tmp/hadoop-hadoop` - induk mount volume DataNode, dibuat root oleh Docker; user `hadoop` tak bisa buat direktori baru di situ | Sudah diset ke `/tmp/nm-local-dir` (sticky-bit, world-writable) di [docker/hadoop.env](docker/hadoop.env) - jangan pindahkan kembali ke bawah `/tmp/hadoop-hadoop` |
| Job MapReduce `Rack-local map tasks` tinggi, `Data-local` 0 | NodeManager dan DataNode di container/hostname berbeda - YARN tidak bisa mencocokkan lokasi blok | Pastikan pakai service `worker1`/`worker2` (DataNode+NodeManager sehostname), bukan container DataNode/NodeManager terpisah |
