# Audit Konsep Big Data (dihasilkan otomatis)

> Dihasilkan `src/phases/bigdata_audit.py` pada 2026-10-15 12:39 UTC terhadap stack yang sedang hidup. **Jangan diedit manual.**

**Ringkasan:** 22 terpenuhi · 4 sebagian · 0 belum · 2 informasi

## Volume

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `VOL-01` | Skala data | **SEBAGIAN** | 477.2 MB di 123 objek lake (raw+clean+curated), 39,668 novel. Muat di satu mesin: skala 'big' belum dicapai oleh data, hanya oleh arsitektur. |
| `VOL-02` | Komputasi terdistribusi (Spark) | **SEBAGIAN** | 1 worker hidup, 2 core, master spark://spark-master:7077 |
| `VOL-03` | Penyimpanan kolumnar (Parquet) | **TERPENUHI** | 52 berkas part Parquet (snappy) di curated-data; zona clean dan curated seluruhnya Parquet |
| `VOL-04` | Partisi fisik untuk partition pruning | **TERPENUHI** | novels_clean di-partitionBy('metadata_tier') -> direktori Hive-style ['full', 'listing']; kueri terfilter metadata_tier melompati direktori lain (dibuktikan PartitionFilters di physical plan Spark, bukan cuma DataFilters pasca-baca) |

**Celah:**

- `VOL-01` — Volume adalah satu-satunya V yang tidak bisa dipenuhi lewat kode — tidak dipalsukan. Klaim skalabilitas bertumpu pada VOL-02 dan STO-02.
- `VOL-02` — Hanya satu worker: shuffle antar-node belum diperagakan. `docker compose up -d --scale spark-worker=2`.

## Variety

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `VAR-01` | Tiga kelas format, tiga berkas fisik | **TERPENUHI** | terstruktur=CSV (1), semi-terstruktur=JSONL (1), tidak terstruktur=TXT (4 shard); 39,668 novel, sumber s3a://clean-data/novels_clean |
| `VAR-02` | Sumber data heterogen saat ingest | **SEBAGIAN** | Fase 1 menelan CSV (katalog, sinopsis, hasil scrape) dan meng-scrape HTML; JSON/TXT baru lahir SESUDAH Fase 2 lewat pemisahan format |

**Celah:**

- `VAR-02` — Semua sumber luar berakhir sebagai CSV. Dataset kedua non-tabular (lihat memori 'dataset kedua graf') belum diputuskan.

## Velocity

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `VEL-01` | Pemrosesan aliran (stream) | **TERPENUHI** | src/phases/streaming_ingest.py - Structured Streaming (file source, trigger availableNow, checkpoint di MinIO); TERBUKTI berjalan: 2 micro-batch tercatat, 2 berkas Parquet keluaran |
| `VEL-02` | Ingest inkremental | **TERPENUHI** | checkpoint melacak 2 offset / 2 commit micro-batch; berkas landing yang sudah ter-commit dilewati pada run berikutnya (file source + checkpoint) - exactly-once, bukan proses ulang seluruh sumber |

## Veracity

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `VER-01` | Kualitas & keterpercayaan data | **TERPENUHI** | audit tata kelola: 17 lulus · 0 peringatan · 0 gagal · 6 informasi; sentinel→NULL, dedup by series_id, weighted rating Bayesian, uji batas atas label (data_contract §5) |

## Value

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `VAL-01` | Insight, model, dan produk | **TERPENUHI** | 31 berkas tabel analitik Fase 3, artefak model 4 head (ada), dashboard Streamlit |

## Penyimpanan terdistribusi

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `STO-01` | Data lake (object storage) | **TERPENUHI** | MinIO S3-compatible, tiga zona; versioning: raw-data=Enabled, clean-data=Enabled, curated-data=Enabled |
| `STO-02` | Distributed File System (HDFS) | **TERPENUHI** | NameNode aktif, 2 DataNode hidup, 75 blok tercatat |
| `STO-03` | Data terdistribusi dalam blok berreplika | **TERPENUHI** | 6 berkas → 21 blok (ukuran blok 4.0 MB), replikasi 2, tersebar di 2 DataNode; 6 berkas terpecah >1 blok; blok kurang-replika: 0 |
| `STO-04` | Dua tier penyimpanan dengan peran berbeda | **INFO** | MinIO = system of record Fase 1-4 (versioning, dibaca Spark & dashboard); HDFS = penyimpanan yang berdampingan dengan compute MapReduce |
| `STO-05` | Spark membaca langsung dari HDFS | **TERPENUHI** | verify_mapreduce.py membaca hdfs://namenode:8020/novel/output lewat Spark, berdampingan dengan s3a:// (laporan verifikasinya ada) |

**Celah:**

- `STO-04` — Isi HDFS adalah turunan zona clean (dapat dibangun ulang), bukan sumber kebenaran kedua.

## Pemrosesan terdistribusi

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `PRC-01` | Klaster MapReduce (YARN) | **TERPENUHI** | ResourceManager STARTED, 2 NodeManager |
| `PRC-02` | Job MapReduce terbukti benar | **TERPENUHI** | 4 dari 4 job identik dengan jawaban Spark SQL (word count TXT, genre JSONL, statistik CSV, join CSV×JSONL) |
| `PRC-03` | Optimasi combiner | **TERPENUHI** | 3,415,461 → 236,337 rekaman (93.1%); 717,320 → 4,207 rekaman (99.4%); 39,668 → 19 rekaman (100.0%); 125,636 → 70 rekaman (99.9%) |
| `PRC-04` | Batch analytics (Spark SQL/ML) | **TERPENUHI** | Fase 2-4 memakai DataFrame & Spark SQL pada cluster standalone; keluarannya ada: 2 part novels_clean, 31 part analytics/ |
| `PRC-05` | Data locality | **TERPENUHI** | 30/30 task map data-local (100%), 0 rack-local - diukur dari counter Hadoop di log job terakhir (bukan klaim topologi); worker1/worker2 menjalankan DataNode + NodeManager sehostname |

## Keandalan & skala

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `REL-01` | Toleransi kegagalan penyimpanan | **TERPENUHI** | HDFS replikasi 2 pada 2 DataNode hidup; MinIO versioning: raw-data=Enabled, clean-data=Enabled, curated-data=Enabled |
| `REL-02` | Skalabilitas horizontal | **SEBAGIAN** | 1 worker Spark dan 2 DataNode/NodeManager hidup; Spark: `--scale spark-worker=N`; HDFS/YARN: tambah workerN (DataNode+NodeManager sehostname) pada compose |
| `REL-03` | Orkestrasi & penjadwalan alur | **TERPENUHI** | scripts/run_pipeline.py: sekuensial (bukan DAG), berhenti di kegagalan pertama, + retry per langkah (`--retry N`, jeda 5 detik) dan resume dari langkah gagal terakhir (`--resume`, state di logs/pipeline_state.json, diuji 3 skenario: retry sukses, gagal permanen tanpa resume palsu, resume melewati langkah sukses) |
| `REL-04` | Basis data NoSQL | **INFO** | Tidak dipakai; kebutuhan relasional/graf dijawab star schema + tabel simpul/sisi (data_model.py) |

**Celah:**

- `REL-01` — Hanya satu NameNode: itu single point of failure (HA NameNode butuh JournalNode + ZooKeeper).
- `REL-02` — Skala horizontal baru terbukti dengan >= 2 node di tiap lapisan.
- `REL-04` — Bukan kekurangan: alasannya terdokumentasi. Menjadi relevan bila traversal graf melewati 2 hop.

## Analisis Graf

| ID | Konsep | Status | Bukti |
|---|---|---|---|
| `GRF-01` | PageRank & sentralitas dihitung (bukan self-join) | **TERPENUHI** | genre_cooccurrence: 35 simpul/579 sisi; author_collaboration: 2574 simpul/2693 sisi (NetworkX: pagerank, degree/betweenness centrality) |
| `GRF-02` | Deteksi komunitas (modularity) | **TERPENUHI** | genre: 3 komunitas (modularity=0.0858); author: 930 komunitas (modularity=0.9947) - 930 komponen terhubung terpisah, terbesar 22 penulis |
| `GRF-03` | Komponen terhubung (fragmentasi jaringan) | **TERPENUHI** | graf genre 1 komponen (menyatu penuh); graf kolaborasi penulis 930 komponen, terbesar 22 penulis; 24,750 penulis lain tanpa kolaborator sama sekali |
| `GRF-04` | Traversal multi-hop terverifikasi | **TERPENUHI** | shortest-path BFS atas graf genre menemukan jalur 2-hop Yaoi -> School Life -> Yuri antar genre yang tidak bertetangga langsung - membuktikan (bukan mengasumsikan) klaim '<=2 hop' yang jadi alasan data_model.py tidak memakai database graf terpisah |
