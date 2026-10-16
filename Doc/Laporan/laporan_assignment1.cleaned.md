# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang Masalah

Web novel dari Tiongkok, Korea Selatan, dan Jepang kini dibaca jauh di luar negara asalnya. Karya-karya ini lahir di platform serial seperti Qidian, JJWXC, Munpia, atau Kakao Page, lalu diterjemahkan ke bahasa Inggris oleh platform berlisensi maupun kelompok penerjemah komunitas. Sektor yang dibahas dalam laporan ini adalah media dan hiburan digital, tepatnya penerbitan fiksi daring lintas bahasa. Berbeda dengan buku cetak yang terbit sekali jadi, web novel terbit bab demi bab, sering setiap hari, dan bisa berjalan bertahun-tahun. Satu judul dengan seribu bab bukan hal aneh.

Pertumbuhannya terlihat jelas di katalog NovelUpdates, direktori komunitas terbesar untuk web novel terjemahan. Dari data yang dikumpulkan untuk tugas ini, judul dengan tahun rilis 2015 berjumlah 844, sedangkan judul dengan tahun rilis 2025 mencapai 5.626. Artinya, dalam sepuluh tahun, jumlah judul baru per tahun naik hampir tujuh kali lipat (Gambar 1.1). Hingga data diambil pada September 2026, sudah tercatat 2.525 judul untuk tahun 2026, padahal tahun itu belum berakhir.

[[FIG:g1_1_pertumbuhan.png|Gambar 1.1 Jumlah judul per tahun rilis di katalog NovelUpdates, 2010 sampai 2025|11]]

Banjir judul ini menimbulkan masalah bagi pihak yang harus memilih. Platform penerjemahan perlu memutuskan judul mana yang layak dilisensi dan diterjemahkan, padahal menerjemahkan satu judul berarti membayar ratusan bab kerja penerjemah dan editor. Median panjang judul di katalog adalah 135 bab, dengan rata-rata 291 bab, dan 3.822 judul melewati 595 bab. Pembaca menghadapi masalah yang sama dari sisi lain: dari puluhan ribu judul, mana yang cocok dengan selera mereka, dan mana yang layak diikuti sampai tamat?

Ada tiga hambatan yang membuat keputusan itu sulit diambil hanya dengan membaca halaman katalog satu per satu.

- **Sinyal kualitas tidak merata.** Sebanyak 41,4% judul punya kurang dari 10 suara rating, dan 2.126 judul belum dinilai sama sekali (Gambar 1.2). Rata-rata rating mentah dari dua atau tiga suara tidak bisa dibandingkan dengan rating dari ratusan suara. Sebanyak 1.587 judul tercatat bernilai sempurna 5,0, tetapi hampir semuanya berasal dari kurang dari 10 suara; di antara judul dengan minimal 10 suara, hanya 2 judul yang bernilai 5,0. Inilah masalah *cold start* yang klasik: judul baru justru yang paling perlu dinilai, tetapi paling sedikit datanya.
- **Data tersebar dan setengah terstruktur.** Informasi setiap judul ada di halaman HTML yang berbeda. Genre, tag, dan penulis disimpan sebagai daftar berpemisah koma. Status rilis bercampur dengan jumlah bab dalam satu teks bebas, misalnya "208 Chapters (Ongoing)", dan sinopsis berupa prosa bebas. Nilai kosong juga tidak konsisten: ada yang ditulis 0, "Unknown", atau "N/A".
- **Label genre bersifat subjektif dan berbaur.** Satu judul rata-rata menyandang 5,19 genre, dengan median 11 tag. Genre besar seperti Romance dan Fantasy muncul bersama hampir semua genre lain, sehingga label genre saja tidak cukup untuk membedakan judul.

[[FIG:g1_2_suara.png|Gambar 1.2 Sebaran jumlah suara rating per judul; batang oranye menandai judul dengan kurang dari 10 suara|11]]

Urgensinya datang dari dua arah. Pertama, volume katalog terus tumbuh lebih cepat daripada kapasitas kurator manusia untuk menilainya. Kedua, keputusan lisensi yang keliru mahal biayanya, sementara judul yang tepat bisa menjadi sumber kekayaan intelektual bernilai tinggi (lihat Subbab 1.2). Tanpa sistem yang mengumpulkan, membersihkan, dan menyatukan data katalog secara otomatis, keputusan tetap bertumpu pada intuisi dan popularitas sesaat.

Karena itu, rumusan masalah tugas ini adalah: bagaimana merancang solusi Big Data yang mampu mengumpulkan katalog web novel terjemahan dari sumber web secara berkelanjutan, membersihkan dan memodelkannya sesuai karakter tiap jenis data, lalu menyajikannya sebagai dasar keputusan lisensi, kurasi, dan rekomendasi?

Dari rumusan itu diturunkan tiga tujuan yang menjadi ukuran keberhasilan rancangan. Pertama, seluruh katalog, bukan sampel, dapat dikumpulkan ulang secara berkala tanpa membebani sumber data dan tanpa menggandakan data. Kedua, setiap angka yang dipakai untuk keputusan dapat ditelusuri ke berkas sumbernya, dan aturan kualitasnya diperiksa otomatis di setiap eksekusi. Ketiga, analis bisa menjawab pertanyaan per segmen (genre, asal negara, waktu) dan pertanyaan relasi (genre mana yang berdekatan, penulis mana yang berkolaborasi) tanpa menulis kode baru.

## 1.2 Market dan Business Drivers

Pendorong pasar (*market drivers*) adalah kondisi di luar organisasi yang membuat masalah ini semakin penting, sedangkan pendorong bisnis (*business drivers*) adalah tujuan internal yang ingin dicapai lewat solusi. Keduanya dirangkum pada Tabel 1.1.

**Market drivers.** Pendorong pertama adalah ekonomi adaptasi kekayaan intelektual. Web novel semakin sering menjadi bahan baku industri hiburan lain. *Solo Leveling*, web novel Korea, diadaptasi menjadi anime yang tayang pada 2024. *Omniscient Reader's Viewpoint* dirilis sebagai film layar lebar pada 2025. Dari Tiongkok, *Mo Dao Zu Shi* menjadi drama *The Untamed* (2019) dan *Lord of the Mysteries* menjadi serial animasi pada 2025. Setiap kisah sukses seperti ini menaikkan nilai hak atas judul yang belum dilisensi, sehingga kemampuan mengenali judul potensial lebih awal menjadi keunggulan bersaing.

Pendorong kedua adalah persaingan platform terjemahan. Webnovel (Qidian International), Wuxiaworld, dan platform lain bersaing merebut pembaca yang sama. Pendorong ketiga adalah penurunan biaya terjemahan: penerjemahan berbantuan mesin memungkinkan lebih banyak judul masuk ke pasar bahasa Inggris, yang pada gilirannya mempercepat pertumbuhan katalog pada Gambar 1.1. Pendorong keempat adalah komposisi pasokan yang terkonsentrasi. Dari judul yang negara asalnya diketahui, 60,6% berasal dari Tiongkok, 21,9% dari Korea, dan 17,3% dari Jepang. Ketiganya juga berbeda dalam penerimaan pembaca: pada judul dengan rating tepercaya, rata-rata weighted rating judul Tiongkok 3,733, Korea 3,641, dan Jepang 3,599 (Subbab 4.1.4). Platform yang ingin tumbuh perlu memahami perbedaan selera dan kualitas antar-asal ini.

**Business drivers.** Dari sisi organisasi, solusi ini ditujukan untuk empat hal: menurunkan risiko keputusan lisensi dengan dasar data yang terukur; menaikkan retensi pembaca lewat rekomendasi yang relevan; mendeteksi tren genre dan asal negara lebih awal; serta mengurangi kerja manual kurator dengan pipeline yang berjalan otomatis dan bisa diulang.

[[TABLE:Tabel 1.1 Market dan business drivers beserta implikasinya pada rancangan solusi]]
| Jenis | Pendorong | Implikasi pada solusi |
| Market | Adaptasi IP ke anime, drama, dan film | Perlu metrik popularitas dan kualitas yang bisa dibandingkan lintas judul (weighted rating, reading list) |
| Market | Persaingan platform terjemahan | Perlu rekomendasi dan penemuan judul yang lebih baik dari pesaing |
| Market | Biaya terjemahan turun, katalog membengkak | Arsitektur harus tahan pertumbuhan volume dan kedatangan data bertahap |
| Market | Pasokan terkonsentrasi di CN, KR, JP | Analisis harus bisa dipotong per asal negara, genre, dan waktu |
| Bisnis | Menurunkan risiko lisensi | Star schema untuk analisis OLAP per segmen |
| Bisnis | Menaikkan retensi pembaca | Model rekomendasi berbasis konten dan analisis relasi genre |
| Bisnis | Deteksi tren lebih awal | Ingest inkremental dan dimensi waktu |
| Bisnis | Efisiensi kurasi | Pipeline otomatis dengan audit kualitas di setiap eksekusi |

## 1.3 Pemangku Kepentingan yang Terlibat

Pemangku kepentingan dipetakan berdasarkan kepentingan mereka terhadap data dan bagaimana solusi melayani kepentingan itu (Tabel 1.2). Pengguna utama solusi adalah tim lisensi dan tim editorial di platform penerjemahan. Pembaca adalah penerima manfaat tidak langsung, sedangkan penulis asli, penerbit asal, dan NovelUpdates adalah pemilik sumber data yang hak-haknya harus dihormati (lihat Bab 3).

Tingkat pengaruh dan kepentingan mereka tidak sama. Tim lisensi dan tim data berpengaruh tinggi sekaligus berkepentingan tinggi, sehingga kebutuhan merekalah yang menentukan bentuk star schema dan dashboard. NovelUpdates berpengaruh tinggi walaupun tidak memakai solusi, karena akses ke sumber data bergantung pada kebijakan mereka. Itulah alasan etika scraping dimasukkan sebagai syarat rancangan, bukan tambahan.

[[TABLE:Tabel 1.2 Pemangku kepentingan, kepentingan, dan peran solusi]]
| Pemangku kepentingan | Kepentingan | Yang disediakan solusi |
| Tim lisensi dan akuisisi platform | Memilih judul yang layak dibeli haknya | Analisis segmen genre, asal, dan waktu di dashboard warehouse |
| Editor dan kurator konten | Menyusun daftar rekomendasi dan promosi | Peringkat berbasis weighted rating, peta relasi genre |
| Penerjemah | Mengetahui panjang dan status karya sebelum berkomitmen | Atribut jumlah bab, status tamat, kecepatan rilis |
| Pembaca | Menemukan judul yang sesuai selera | Rekomendasi berbasis konten di halaman AI Studio |
| Penulis asli dan penerbit asal (Qidian, JJWXC, Munpia, dll.) | Karya dan nama mereka dipakai secara wajar | Hanya metadata dan sinopsis untuk analitik nonkomersial; nama pena diperlakukan sebagai kredit karya |
| NovelUpdates (pemilik sumber) | Server tidak terbebani, data tidak disalahgunakan | Scraper dengan jeda 6 sampai 45 detik, tanpa permintaan paralel |
| Tim data (engineer, analis, data scientist) | Pipeline yang andal dan bisa diaudit | Orkestrasi dengan retry dan resume, audit tata kelola otomatis |
| Tim legal dan kepatuhan | Kepatuhan hak cipta dan pelindungan data pribadi | Provenance berbasis checksum, zona raw yang tidak pernah ditimpa, pemindaian PII |

# BAB 2 RANCANGAN SOLUSI BIG DATA

Rancangan solusi disusun dalam lima lapisan: sumber data, ingestion, penyimpanan, pemrosesan, dan penyajian (Gambar 2.1). Seluruh komponen berjalan sebagai container Docker dalam satu jaringan internal, sehingga lingkungan yang sama bisa dijalankan ulang di mesin lain dengan satu perintah. Setiap lapisan dipilih karena karakter data dan pertanyaan yang harus dijawab, bukan karena kelengkapan teknologi. Alasan tiap pilihan dijelaskan di subbab berikut, disertai pseudocode dari implementasi yang sebenarnya berjalan.

[[FIG:g2_1_arsitektur.png|Gambar 2.1 Arsitektur solusi Big Data katalog web novel|16]]

Data mengalir dari kiri ke kanan. Berkas sumber tidak pernah diubah setelah masuk zona raw; setiap transformasi membaca dari satu zona dan menulis ke zona berikutnya. Dengan pola ini, kesalahan di satu tahap bisa diperbaiki dengan menjalankan ulang tahap itu saja, tanpa mengambil ulang data dari web.

## 2.1 Metode Pengumpulan Data

Data dikumpulkan lewat empat jalur yang saling melengkapi (Tabel 2.1).

**1. Dataset awal (batch).** Titik awal proyek adalah dua berkas CSV hasil kompilasi katalog publik NovelUpdates: `data_novel.csv` berisi 4.461 judul dengan 27 kolom metadata, dan `novel_dataset_scrapped.csv` berisi 5.000 baris sinopsis. Keduanya dihubungkan lewat kolom `url`. Sebelum join, 538 baris `url` ganda pada berkas sinopsis harus dibuang; tanpa langkah ini tabel hasil join menggelembung menjadi 4.999 baris dan semua agregat menjadi salah.

**2. Web scraping bertahap.** Katalog 4.461 judul ternyata hanya sebagian kecil dari NovelUpdates, jadi scraper ditulis untuk mengambil sisanya dalam dua tahap. Tahap *listing* mengambil 36.679 judul unik dari halaman daftar seri; 35.209 di antaranya belum ada di dataset awal, sehingga katalog gabungan berisi 39.670 baris sebelum pembersihan akhir. Tahap *detail* membuka halaman setiap seri dan mengambil 27 kolom yang sama dengan dataset awal; hasilnya 36.674 halaman berhasil diambil (99,99%). Lima sisanya selalu mengembalikan HTTP 404 karena seri tersebut sudah dihapus atau berganti judul.

Karena NovelUpdates berada di balik Cloudflare, scraper memakai `curl_cffi` yang meniru sidik jari TLS peramban, ditambah cookie sesi dari peramban pengguna. Ada empat keputusan desain yang penting di sini:

- **Jeda adaptif.** Jeda dasar dimulai dari 6 detik, dengan jeda acak antara nilai dasar dan 1,67 kali nilai dasar. Setiap HTTP 429 mengalikan jeda dasar dengan 1,6 sampai batas 45 detik, dan setiap respons normal menurunkannya perlahan dengan faktor 0,93. Percobaan awal dengan jeda tetap 2,5 sampai 5 detik lancar selama sekitar 40 halaman, lalu dibalas 429 terus-menerus.
- **Cooldown berkala.** Setiap 40 permintaan, scraper beristirahat 30 sampai 60 detik, meniru jeda alami pembaca manusia.
- **Resumable.** Setiap halaman yang berhasil langsung ditulis ke CSV. Menjalankan ulang scraper otomatis melewati URL yang sudah ada, sehingga proses yang secara total memakan sekitar 100 jam (kira-kira 10 detik per halaman) bisa dihentikan dan dilanjutkan kapan saja.
- **Prioritas nilai.** Judul dari papan peringkat populer diambil lebih dulu karena merekalah yang punya cukup suara untuk dianalisis. Jika proses berhenti di tengah jalan, data yang sudah terkumpul tetap berguna.

Logika pengambilan satu halaman dirangkum pada Pseudocode 2.1. Perhatikan bahwa HTTP 403 atau halaman tantangan Cloudflare menghentikan proses, bukan diulang. Mengulang permintaan yang ditolak hanya menambah beban server tanpa peluang berhasil, karena penyebabnya adalah cookie yang kedaluwarsa.

[[CODE:Pseudocode 2.1 Pengambilan halaman dengan jeda adaptif (scripts/scraper/novelupdates.py)]]
CONST MIN_DELAY = 6, MAX_DELAY = 10, MAX_ADAPTIVE = 45
CONST STEP_UP = 1.6, STEP_DOWN = 0.93, COOLDOWN_EVERY = 40

FUNCTION fetch_page(url, max_attempts = 4):
    FOR attempt = 1 TO max_attempts:
        sleep(random(delay, delay * MAX_DELAY / MIN_DELAY))
        IF request_count MOD COOLDOWN_EVERY = 0:
            sleep(random(30, 60))               // periodic long pause
        response <- GET(url, timeout = (15, 40)) // (connect, read)
        IF response.status = 200:
            IF "Just a moment" IN response.text:
                STOP "cf_clearance cookie expired"
            delay <- MAX(delay * STEP_DOWN, MIN_DELAY)
            RETURN response.text
        IF response.status = 403:
            STOP "cookie or User-Agent mismatch"
        IF response.status = 429:
            delay <- MIN(delay * STEP_UP, MAX_ADAPTIVE)
            sleep(random(45, 90) * attempt)     // rate-limit specific backoff
            CONTINUE
        sleep(5 * attempt)                      // other errors: retry
    RETURN NULL                                 // permanent failure, URL skipped
// run_detail: FOR EACH url (popular first) NOT YET IN detail.csv:
//     append_row(detail.csv, parse_detail(fetch_page(url)))   -> resumable
[[/CODE]]

**3. Ingestion ke data lake.** Fase 1 pipeline (`phase1_collect.py`) menyalin keempat berkas sumber secara bit-per-bit ke zona `raw-data`. Untuk setiap berkas dibuat *manifest* JSON berisi checksum MD5, jumlah baris, dan waktu unggah. Jika checksum berkas lokal sama dengan yang sudah tersimpan, unggahan dilewati. Jika berbeda, pipeline mencatat perubahan checksum lama ke baru di log sebelum menimpa, sehingga versi sebelumnya tetap bisa dilacak lewat versioning bucket. Dengan cara ini ingestion bersifat idempoten: menjalankannya berkali-kali tidak menggandakan data (Pseudocode 2.2).

[[CODE:Pseudocode 2.2 Ingestion idempoten dengan manifest checksum (src/phases/phase1_collect.py)]]
FUNCTION ingest_file(source, object_key, force = FALSE):
    md5 <- compute_md5(source, chunk = 1 MB)
    rows <- count_csv_rows(source)
    old_manifest <- read_manifest(raw-data, object_key + "_manifest.json")
    IF old_manifest EXISTS AND old_manifest.md5 = md5 AND NOT force:
        log("identical object already in lake, upload skipped")
        RETURN old_manifest
    IF old_manifest EXISTS AND old_manifest.md5 != md5:
        log("source changed: old md5 -> new md5")
    upload(source, raw-data, object_key)
    manifest <- {source, md5, rows, object_uri, uploaded_at_utc}
    write_json(raw-data, object_key + "_manifest.json", manifest)
    RETURN manifest
[[/CODE]]

**4. Ingest inkremental (streaming).** Hasil scraping pada kenyataannya datang bertahap, bukan sekaligus. Untuk kebutuhan ini dipakai Spark Structured Streaming dengan *file source* yang memantau folder `streaming/landing/` di MinIO, *trigger* `availableNow`, dan checkpoint yang juga disimpan di MinIO. Setiap micro-batch melewati validasi ringkas: baris tanpa `series_id` dibuang, penanda rating tepercaya dikonversi ke boolean, dan waktu ingest dicatat (Pseudocode 2.3).

Perilakunya diuji langsung, tidak diasumsikan dari dokumentasi: batch pertama berisi 500 baris menghasilkan 500 baris baru; batch kedua berisi 500 baris menambah total menjadi 1.000; menjalankan ulang job tanpa berkas baru menghasilkan 0 baris baru. Checkpoint mencegah berkas yang sama diproses dua kali, yang merupakan definisi ingest *exactly-once*. Trigger `availableNow` dipilih karena job berhenti sendiri setelah semua berkas baru diproses, sehingga bisa dijadwalkan seperti job batch tetapi dengan semantik streaming.

[[CODE:Pseudocode 2.3 Ingest inkremental dengan Structured Streaming (src/phases/streaming_ingest.py)]]
stream <- spark.readStream
            .format("csv").schema(METRICS_SCHEMA)
            .option("maxFilesPerTrigger", 1)         // one file = one micro-batch
            .load("s3a://raw-data/streaming/landing/")

FUNCTION validate(batch):
    batch <- FILTER batch WHERE series_id IS NOT NULL
    batch.is_rated_reliable <- (lower(is_rated_reliable) = "true")
    batch.ingested_at_utc <- now()
    RETURN batch

query <- validate(stream).writeStream
            .format("parquet")
            .option("checkpointLocation", "s3a://curated-data/streaming/_checkpoint")
            .trigger(availableNow = TRUE)             // process all new files, then stop
            .start("s3a://curated-data/streaming/novel_metrics_incremental")
query.awaitTermination()
// files whose offsets are already in the checkpoint are never read again
[[/CODE]]

[[TABLE:Tabel 2.1 Ringkasan metode pengumpulan data]]
| Jalur | Sumber | Mekanisme | Volume | Frekuensi |
| Dataset awal | CSV katalog dan sinopsis | Unggah batch + manifest MD5 | 4.461 + 5.000 baris | Sekali, idempoten |
| Scraping listing | Halaman daftar NovelUpdates | `curl_cffi`, jeda adaptif | 36.679 judul unik | Berkala |
| Scraping detail | Halaman seri NovelUpdates | `curl_cffi`, resumable | 36.674 halaman | Berkala, bertahap |
| Streaming | Berkas CSV di folder landing | Structured Streaming, checkpoint | Per batch (uji: 2 x 500 baris) | Setiap batch baru tiba |

## 2.2 Teknologi Penyimpanan Data

Penyimpanan dirancang dua tingkat dengan peran berbeda: data lake berbasis object storage sebagai sumber kebenaran, dan HDFS sebagai penyimpanan terdistribusi yang berdampingan dengan komputasi MapReduce. Model data warehouse (star schema) dan graf tidak dijalankan di DBMS terpisah, melainkan dimaterialisasi sebagai tabel Parquet di zona kurasi.

**Data lake MinIO.** MinIO adalah object storage yang kompatibel dengan API Amazon S3, sehingga Spark membacanya lewat konektor `s3a://` yang sama seperti di layanan cloud. Jika sistem ini dipindah ke AWS S3 atau Google Cloud Storage, kode pipeline tidak perlu diubah; cukup endpoint dan kredensial. Data lake dibagi menjadi tiga zona:

- `raw-data` (74,7 MB, 10 objek): berkas sumber apa adanya beserta manifest-nya. Zona ini tidak pernah ditimpa oleh transformasi apa pun, sehingga setiap hasil bisa ditelusuri kembali ke berkas aslinya.
- `clean-data` (116,4 MB): data hasil pembersihan dalam format Parquet terkompresi Snappy, dipartisi fisik berdasarkan `metadata_tier` (direktori `full` dan `listing`).
- `curated-data` (129,1 MB, 120 objek): hasil analitik Fase 3, star schema dan tabel graf di `model/`, hasil audit di `governance/`, serta artefak model di `models/`.

Versioning diaktifkan di ketiga bucket agar setiap objek yang tertimpa masih bisa dipulihkan dan diaudit.

Parquet dipilih untuk zona clean dan curated karena format kolumnar cocok dengan pola baca analitik: kueri yang hanya butuh `weighted_rating` dan `type_origin` tidak perlu membaca kolom sinopsis yang besar. Partisi fisik menambah keuntungan kedua. Kueri yang memfilter `metadata_tier = 'full'` hanya membuka direktori `metadata_tier=full`, dan hal ini terbukti dari munculnya `PartitionFilters` pada physical plan Spark, bukan sekadar `DataFilters` yang diterapkan setelah seluruh berkas dibaca. Untuk itu data ditulis dengan `repartition("metadata_tier")` lalu `partitionBy("metadata_tier")`, sehingga setiap nilai partisi menjadi satu direktori bergaya Hive.

**HDFS.** Klaster HDFS terdiri dari satu NameNode dan dua worker. Setiap worker menjalankan DataNode dan NodeManager YARN dalam container yang sama. Ke HDFS dimuat tiga berkas hasil pemisahan format (Subbab 4.1.3): CSV, JSONL, dan empat shard TXT. Ukuran blok diatur 4 MB supaya berkas berukuran puluhan megabita tetap terpecah menjadi beberapa blok (enam berkas menjadi 21 blok), dengan faktor replikasi 2 sehingga satu DataNode boleh mati tanpa kehilangan data. Hasil `hdfs fsck` menunjukkan 0 blok kekurangan replika. Skrip pemuatannya menunggu NameNode keluar dari *safe mode* dan jumlah DataNode hidup memenuhi faktor replikasi sebelum menulis (Pseudocode 2.4).

Menempatkan DataNode dan NodeManager pada hostname yang sama bukan detail kecil. Pada konfigurasi awal keduanya terpisah, dan counter Hadoop mencatat 0% task *data-local*. Setelah digabung, 30 dari 30 task map pada empat job membaca blok dari disk lokalnya sendiri.

[[CODE:Pseudocode 2.4 Pemuatan dataset ke HDFS (scripts/hdfs/load_hdfs.sh)]]
REPLICATION <- 2 ; BLOCK_SIZE <- 4 MB
REPEAT up to 60 times, 5 seconds apart:
    live <- number of live DataNodes (hdfs dfsadmin -report)
    IF live >= REPLICATION AND safemode = OFF: BREAK
IF live < REPLICATION: FAIL "not enough DataNodes"
make_dirs(/novel/structured, /novel/semistructured, /novel/unstructured/synopsis)
FOR EACH file IN {novel_metrics.csv, novel_labels.jsonl, synopsis-000[0-3].txt}:
    hdfs dfs -D dfs.blocksize=BLOCK_SIZE -D dfs.replication=REPLICATION -put -f file
report <- hdfs fsck /novel -files -blocks -locations
ASSERT report.under_replicated = 0
[[/CODE]]

**Mengapa bukan NoSQL atau graph database terpisah.** Pertanyaan graf pada proyek ini (ko-okurensi genre, jangkauan penulis) berkedalaman paling banyak dua *hop*, dan klaim ini diverifikasi dengan traversal BFS sungguhan (Subbab 4.2). Pada kedalaman tersebut, self-join tabel sisi dan NetworkX di memori sudah cukup tanpa menambah service baru. Tabel simpul dan sisi tetap disimpan dalam format standar agar bisa dimuat ke Neo4j atau GraphFrames jika kebutuhan berkembang. Alasan serupa berlaku untuk basis data dokumen: data semi-terstruktur disimpan sebagai array di Parquet dan JSONL, yang sudah bisa di-*explode* oleh Spark.

[[TABLE:Tabel 2.2 Perbandingan opsi penyimpanan dan keputusannya]]
| Opsi | Kelebihan untuk kasus ini | Kekurangan | Keputusan |
| Data lake (MinIO, S3) | Murah, skema fleksibel, zona raw tetap utuh, versioning | Tanpa indeks, perlu engine kueri | Dipakai sebagai sumber kebenaran |
| HDFS | Blok berreplika, data locality untuk MapReduce | Mengelola NameNode, satu titik kegagalan tanpa HA | Dipakai untuk jalur MapReduce |
| Data warehouse (star schema) | Kueri OLAP cepat dan mudah dipahami analis | Butuh pemodelan di depan | Dipakai, dimaterialisasi sebagai Parquet |
| NoSQL dokumen | Cocok untuk data bersarang | Data sudah bisa disimpan sebagai array di Parquet | Tidak dipakai |
| Graph database | Traversal dalam yang efisien | Menambah service; kedalaman kueri hanya 2 hop | Tidak dipakai, tabel simpul/sisi siap dimigrasikan |

## 2.3 Processing Framework

Tiga kerangka pemrosesan dipakai sesuai jenis bebannya.

**Apache Spark 3.5 (standalone).** Spark adalah mesin utama. Klaster terdiri dari satu master dan dua worker dengan total 4 core, dan jumlah worker bisa ditambah dengan `docker compose up --scale spark-worker=N`. Spark menjalankan Fase 2 (pembersihan dengan DataFrame API), Fase 3 (analitik dengan Spark SQL), pembangunan star schema dan proyeksi graf, Structured Streaming, serta persiapan data untuk machine learning. Spark dipilih karena satu engine melayani batch, SQL, streaming, dan ML dengan API yang sama. Pemrosesan di memori juga cocok untuk pipeline yang menjalankan puluhan transformasi berurutan. Driver dan executor memakai image Docker yang sama, sehingga versi Python dan PySpark di kedua sisi dijamin identik.

**Hadoop MapReduce di YARN.** Sebagai jalur kedua yang independen, empat job Hadoop Streaming dengan mapper dan reducer Python murni dijalankan di atas data HDFS. Setiap job mewakili satu kelas format:

- `wordcount` menghitung frekuensi kata pada korpus sinopsis (TXT, tidak terstruktur).
- `genre_stats` menghitung frekuensi dan ko-okurensi genre (JSONL, semi-terstruktur).
- `origin_stats` menghitung statistik per negara asal (CSV, terstruktur).
- `genre_rating` menghitung rata-rata rating per genre dengan *reduce-side join* antara CSV dan JSONL.

Pseudocode 2.5 memperlihatkan job `genre_stats`. Mapper mengeluarkan dua jenis kunci berprefiks dalam satu job, sehingga berkas JSONL cukup dibaca sekali. Reducer yang sama dipakai sebagai combiner, karena penjumlahan bersifat asosiatif dan komutatif: menjumlah hasil parsial di sisi map lalu menjumlahkannya lagi di reducer memberi hasil yang sama persis. Rata-rata tidak bersifat demikian, sehingga job yang butuh rata-rata mengeluarkan pasangan (jumlah, cacah) dan baru membagi di akhir.

[[CODE:Pseudocode 2.5 Job MapReduce frekuensi dan ko-okurensi genre (src/mapreduce/genre_mapper.py, sum_reducer.py)]]
MAPPER(jsonl_line):
    genres <- sort(unique(json(jsonl_line).genres))
    FOR EACH g IN genres:
        EMIT("G:" + g, 1)
    FOR EACH (a, b) IN combinations(genres, 2):       // a < b, each pair once
        EMIT("P:" + a + "|" + b, 1)

COMBINER = REDUCER(key, values):                     // input already sorted by key
    EMIT(key, SUM(values))
[[/CODE]]

Job `genre_rating` memerlukan data dari dua berkas: rating ada di CSV, genre ada di JSONL. Mapper mengenali asal setiap baris dari path berkas input, lalu menandai rekamannya dengan `R` atau `G`. Fase shuffle mengirim semua rekaman ber-`series_id` sama ke reducer yang sama, dan di sanalah keduanya digabung (Pseudocode 2.6). Pilihan *reduce-side join* dipakai karena kedua sisi berukuran sebanding; *map-side join* hanya tepat bila salah satu sisi cukup kecil untuk dimuat ke memori setiap mapper.

[[CODE:Pseudocode 2.6 Reduce-side join CSV x JSONL (src/mapreduce/join_mapper.py, join_reducer.py)]]
MAPPER(line):
    IF input_path CONTAINS "/structured/":            // structured CSV
        IF is_rated_reliable = 1 AND weighted_rating EXISTS:
            EMIT(series_id, ("R", weighted_rating))
    IF input_path CONTAINS "/semistructured/":        // semi-structured JSONL
        EMIT(series_id, ("G", join(sort(genres), "|")))

REDUCER_STAGE_1(series_id, records):
    rating <- value of the record tagged "R"
    genres <- value of the record tagged "G"
    IF rating EXISTS AND genres EXISTS:
        FOR EACH g IN split(genres, "|"): EMIT(g, (rating, 1))

STAGE_2: summing REDUCER per genre -> (rating_sum, count)
         average <- rating_sum / count              // divided at the end, not in the reducer
[[/CODE]]

Combiner memangkas lalu lintas shuffle secara drastis, misalnya pada `wordcount` dari 3.415.461 rekaman menjadi 236.337 (turun 93,1%) dan pada `genre_stats` dari 717.320 menjadi 4.207 (turun 99,4%). Hasil keempat job dicocokkan baris demi baris dengan jawaban Spark SQL atas pertanyaan yang sama, dan keempatnya identik tanpa selisih.

**NetworkX untuk algoritma graf.** Proyeksi graf dari 858.327 sisi mentah dikerjakan Spark secara terdistribusi. Hasil proyeksinya (35 simpul genre dan 2.578 simpul penulis) cukup kecil untuk dianalisis di memori satu proses dengan NetworkX: PageRank, sentralitas, komponen terhubung, deteksi komunitas, dan jalur terpendek. Algoritma rinci dibahas di Subbab 4.2.

**Mengapa bukan Flink.** Flink unggul untuk streaming latensi rendah dengan state besar. Data katalog ini datang dalam hitungan batch per hari, bukan ribuan event per detik, sehingga micro-batch Structured Streaming sudah memadai dan tidak perlu engine kedua.

**Orkestrasi.** Seluruh langkah dijalankan oleh `run_pipeline.py` secara berurutan: Fase 1 sampai 4, pemisahan format, lalu lapisan tata kelola (katalog elemen, model data, analisis graf, audit). Pipeline berhenti pada kegagalan pertama, karena melanjutkan di atas data yang tidak lengkap akan menghasilkan analitik yang tampak sukses tetapi salah. Opsi `--retry N` mengulang langkah yang gagal karena gangguan sementara, misalnya NameNode yang masih dalam *safe mode*. Opsi `--resume` melanjutkan dari langkah yang gagal terakhir berdasarkan berkas status yang hanya berlaku untuk argumen eksekusi yang sama (Pseudocode 2.7). Eksekusi ulang Fase 2, Fase 3, dan seluruh lapisan tata kelola pada 29 September 2026 selesai dalam 201,3 detik.

[[CODE:Pseudocode 2.7 Orkestrasi dengan retry dan resume (scripts/run_pipeline.py)]]
FUNCTION run_step(name, func, retries):
    FOR attempt = 1 TO retries + 1:
        TRY: func(); RETURN TRUE
        CATCH error:
            IF attempt <= retries: log(error); sleep(5); CONTINUE
            log_error(error); RETURN FALSE

FUNCTION run_pipeline(start, end, retries, resume):
    signature <- hash(start, end, governance_options)
    done <- IF resume: load_state(signature) ELSE {}
    FOR EACH (name, func) IN [phases start..end, variety, catalog, model, graph, audit]:
        IF name IN done: SKIP
        IF NOT run_step(name, func, retries):
            STOP pipeline              // never continue on incomplete data
        done <- done + {name}; save_state(signature, done)
    clear_state()                      // all steps succeeded, nothing to resume
[[/CODE]]

**Penyajian.** Hasil akhir dibaca oleh dashboard Streamlit dengan akun MinIO khusus baca. Selain halaman produk (ikhtisar katalog, analitik, AI Studio, metodologi), dashboard memiliki tiga halaman *data platform*: penjelajah OLAP di atas star schema, peta relasi graf, dan ringkasan tata kelola (Subbab 3.4 dan 4.2). Dashboard tidak menjalankan job Spark sendiri; ia hanya membaca tabel Parquet yang sudah dimaterialisasi, sehingga interaksi terasa instan dan angka di layar selalu sama dengan angka di log pipeline.

# BAB 3 TATA KELOLA IT

Tata kelola tidak cukup ditulis sebagai kebijakan. Pada proyek ini setiap kontrol diperiksa oleh `governance_audit.py`, yang berjalan otomatis di akhir pipeline dan menulis hasilnya ke `curated-data/governance/`. Eksekusi terakhir menjalankan 23 pemeriksaan dengan hasil 15 lulus, 2 peringatan, 0 gagal, dan 6 informasi (Gambar 3.2). Setiap pemeriksaan memiliki kode (misalnya SEC-01 atau QLT-03), bukti yang diukur dari sistem yang sedang hidup, dan langkah perbaikan bila tidak lulus. Subbab berikut menjelaskan kebijakan di balik setiap kelompok pemeriksaan.

## 3.1 Security

Kebijakan keamanan mengikuti prinsip hak akses minimum dan pertahanan berlapis.

- **Bucket tertutup untuk akses anonim.** Ketiga bucket MinIO tidak memiliki policy anonim (SEC-03, lulus untuk ketiganya).
- **Pemisahan hak baca dan tulis.** Dashboard memakai akun `novel-dashboard` yang hanya boleh membaca. Kontrol ini sudah diuji: percobaan menulis dari container dashboard ditolak dengan `AccessDenied`. Bug di lapisan presentasi tidak bisa merusak data kurasi (SEC-04).
- **Isolasi jaringan.** Semua service berada dalam bridge network Docker `bigdata-net`. Hanya port antarmuka pengguna yang dipetakan ke host (SEC-05).

Dua temuan berstatus peringatan dan dicatat terbuka. SEC-01: kredensial root MinIO masih bawaan (`minioadmin`); langkah perbaikannya adalah mengganti `MINIO_ROOT_USER` dan `MINIO_ROOT_PASSWORD` sebelum sistem diekspos ke jaringan mana pun. SEC-02: lalu lintas ke MinIO belum dienkripsi TLS. Risiko ini dapat diterima selama lalu lintas hanya melintasi jaringan internal Docker di satu mesin, tetapi TLS wajib dipasang sebelum sistem keluar dari localhost. Pseudocode 3.1 memperlihatkan bagaimana pemeriksaan ini dijalankan terhadap konfigurasi yang sebenarnya, bukan dari daftar klaim.

[[CODE:Pseudocode 3.1 Pemeriksaan keamanan object storage (src/phases/governance_audit.py)]]
FUNCTION audit_security():
    add("SEC-01", WARN IF access_key IN default_credentials ELSE PASS)
    add("SEC-02", PASS IF endpoint STARTS WITH "https://" ELSE WARN)
    FOR EACH bucket IN {raw-data, clean-data, curated-data}:
        policy <- get_bucket_policy(bucket)          // no policy = closed
        public <- ANY statement WITH Effect = "Allow" AND Principal = "*"
        add("SEC-03/" + bucket, FAIL IF public ELSE PASS)
    add("SEC-04", PASS IF MINIO_READONLY_USER account is defined ELSE WARN)
    add("SEC-05", INFO, "only UI ports are mapped to the host")
[[/CODE]]

## 3.2 Compliance

Kepatuhan pada proyek ini menyangkut tiga hal: hak cipta, etika pengambilan data, dan jejak audit.

**Hak cipta.** Metadata katalog seperti judul, genre, jumlah bab, dan rating bersifat faktual. Sinopsis tetap merupakan ciptaan pemiliknya yang dilindungi Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta. Sinopsis hanya dipakai untuk analitik dan pelatihan model nonkomersial dalam konteks akademik, dan dataset tidak dipublikasikan ulang. Status lisensi redistribusi harus ditegaskan terlebih dahulu jika dataset ini akan dibagikan (CMP-04).

**Etika scraping.** Scraper tidak pernah mengirim permintaan paralel. Jedanya minimal 6 detik, naik otomatis ketika server memberi sinyal beban (HTTP 429), dan diselingi cooldown berkala (Pseudocode 2.1). Scraper juga berhenti sendiri ketika sesi kedaluwarsa, bukan mencoba menembusnya berulang kali. Dengan rata-rata sekitar 10 detik per halaman, beban yang diberikan setara dengan satu pembaca yang membuka halaman satu per satu.

**Jejak audit dan provenance.** Setiap berkas di zona raw memiliki manifest berisi checksum MD5, jumlah baris, dan waktu unggah (CMP-01). Zona raw tidak pernah ditimpa transformasi; semua hasil ditulis ke bucket lain (CMP-02). Versioning aktif di ketiga bucket sehingga setiap versi objek dapat dipulihkan (CMP-03). Pemeriksaan provenance dijalankan dengan membaca kembali manifest yang tersimpan di lake (Pseudocode 3.2).

[[CODE:Pseudocode 3.2 Pemeriksaan provenance dan versioning (src/phases/governance_audit.py)]]
FUNCTION audit_compliance():
    manifests <- ALL objects in raw-data ending with "_manifest.json"
    complete <- EVERY m IN manifests HAS md5, rows, uploaded_at
    add("CMP-01", PASS IF manifests NOT EMPTY AND complete ELSE FAIL,
        evidence = list(name, md5[0..8], rows))
    add("CMP-02", PASS, "transformations only write to clean-data / curated-data")
    status <- {b: get_bucket_versioning(b) FOR b IN buckets}
    add("CMP-03", PASS IF ALL status = "Enabled" ELSE WARN)
    add("CMP-04", INFO, "public source; synopses remain their owners' copyright")
[[/CODE]]

## 3.3 Data Privacy

Dataset ini tidak memuat data tingkat individu pembaca. Sinyal pembaca hanya tersedia sebagai agregat (`rating_votes`, `reading_list_count`), sehingga tidak ada subjek data yang bisa diidentifikasi ulang (PRV-03). Pemindaian pola email, nomor telepon Indonesia, nomor kartu kredit, NIK 16 digit, dan alamat IP pada seluruh kolom teks tidak menemukan satu pun kecocokan langsung (PRV-01). Semua kolom dan pola dipindai dalam satu agregasi Spark, bukan satu job per kolom, sehingga pemindaian 39.668 baris selesai dalam hitungan detik (Pseudocode 3.3).

[[CODE:Pseudocode 3.3 Pemindaian PII dalam satu agregasi (src/phases/governance_audit.py)]]
PATTERNS <- {
    email        : "[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
    phone        : "(?<!\d)(?:\+62|08)\d{8,12}(?!\d)",
    credit_card  : "(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
    nik_16_digit : "(?<!\d)\d{16}(?!\d)",
    ip_address   : "(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)" }

FUNCTION scan_pii(df):
    text_cols <- columns of df with type string
    aggregations <- [ SUM(1 IF df[c] MATCHES PATTERNS[p] ELSE 0) AS c||p
                      FOR c IN text_cols, p IN PATTERNS ]
    result <- df.agg(aggregations)                   // a single Spark job
    RETURN [(c, p, n) FOR (c||p, n) IN result IF n > 0]
[[/CODE]]

Satu hal tetap perlu diperhatikan: kolom `authors` memuat nama pena. Nama pena dipublikasikan platform sebagai kredit karya, tetapi tetap dapat menjadi data pribadi jika bisa dikaitkan ke orang nyata, sebagaimana dimaksud Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi. Karena itu nama pena hanya dipakai sebagai atribut karya (siapa menulis apa), tidak dipadankan dengan sumber lain untuk mengungkap identitas asli, dan tidak dipakai untuk profil perorangan. Prinsip minimisasi data juga diterapkan: kolom `artists` (84,67% kosong) dan `english_publisher` (93,72% kosong) dibuang di Fase 2 karena tidak memberi nilai analitik.

## 3.4 Data Quality

Kualitas data dijaga pada lima dimensi: kelengkapan, validitas, keunikan, konsistensi, dan keterpercayaan. Setiap langkah pembersihan mencatat jumlah baris sesudahnya ke tabel `quality_report` (Gambar 3.1), sehingga perubahan jumlah baris yang tidak terduga langsung terlihat.

[[FIG:g3_1_pembersihan.png|Gambar 3.1 Jumlah baris setelah setiap langkah pembersihan (tabel quality_report)|13.5]]

**Sentinel menjadi NULL.** Sumber data memakai angka 0 dan teks "Unknown" sebagai pengganti nilai kosong. Pada dataset awal, 20,22% `chapter_count` dan 22,66% `year` bernilai 0. Tanpa normalisasi, rata-rata jumlah bab ikut menghitung 902 judul bernilai 0 dan meleset sekitar 20%. Rating 0 tanpa suara berarti "belum dinilai", bukan "nilainya jelek", sehingga juga diubah menjadi NULL agar tidak menarik turun rata-rata genre. Satu kasus menarik ditangani di sini: jumlah bab yang terpotong pemisah ribuan ("1,593 Chapters" tercatat sebagai 593) dipulihkan dari kolom `status_coo`, tetapi hanya bila tiga digit terakhirnya sama. Syarat ini mencegah perbedaan wajar antara status lama dan kolom terbaru ikut tertimpa (Pseudocode 3.4).

[[CODE:Pseudocode 3.4 Normalisasi sentinel (src/phases/phase2_clean.py, normalize_sentinels)]]
FOR EACH col IN {type, type_origin, tags, language, authors, original_publisher,
                 status_coo, release_frequency} THAT EXIST IN df:
    df[col] <- NULL IF lower(trim(df[col])) IN {"unknown", "n/a"} ELSE trim(df[col])

status_chapters <- extract_number(remove_commas(status_coo), "(\d+)\s*chapters?")
df.chapter_count <- status_chapters IF status_chapters > chapter_count > 0
                                       AND status_chapters MOD 1000 = chapter_count
                    ELSE chapter_count               // restore counts truncated at the comma
df.chapter_count <- NULL IF chapter_count <= 0
df.year <- NULL IF year < 1000 OR year > 2030           // 1740 (a classic) stays valid
df.reading_list_count <- NULL IF reading_list_count < 0
df.rating_score <- NULL IF rating_score <= 0 OR rating_votes IS NULL OR rating_votes <= 0
[[/CODE]]

**Temuan baru dari star schema.** Saat membangun dashboard warehouse, tiga cacat kualitas yang sebelumnya lolos audit ditemukan dan diperbaiki pada 29 September 2026:

- Teks "N/A" dari halaman detail belum dikenali sebagai sentinel. Akibatnya 478 judul punya asal negara "N/A", 1.140 judul punya penulis "N/A" (sehingga "N/A" menjadi penulis paling produktif di `dim_author`), dan 4.916 judul punya penerbit "N/A".
- Tujuh judul punya tahun rilis terpotong seperti 15, 21, 202, dan 206. Batas bawah tahun dinaikkan menjadi 1000, bukan 1900, karena *Painted Skin* (1740, Pu Songling) adalah karya klasik yang sah.
- Empat judul punya `reading_list_count` negatif (terendah -162), yang membuat `log1p` pada skor popularitas menghasilkan NaN.

Pemeriksaan QLT-03 juga diperketat. Versi lama hanya mengecek angka 0, itulah sebabnya "N/A" lolos dengan status lulus. Versi baru ikut memeriksa sentinel teks, tahun di bawah 1000, dan nilai negatif. Pelajaran dari temuan ini adalah bahwa aturan kualitas perlu diuji dari sisi konsumen data juga: cacat "N/A" tidak terlihat di log pembersihan, tetapi langsung terlihat begitu tabel dimensi penulis diurutkan.

**Keunikan dan konsistensi.** Deduplikasi memakai `series_id`, bukan judul, karena ada judul kembar. Bila satu seri muncul lebih dari sekali, baris dengan suara terbanyak dipertahankan karena itu snapshot paling matang. Hasil audit: 39.668 kunci unik dari 39.668 baris (QLT-02).

Nama penulis yang sama dalam dua aksara, seperti "Mizume, ミズメ", disatukan lewat resolusi entitas. Dua nama dianggap alias bila salah satunya beraksara CJK, Hangul, atau Kana dan yang lain tidak, serta nama yang lebih jarang tidak pernah muncul tanpa nama yang lebih sering. Hasilnya, 36.391 nama menjadi 27.330 entitas. Sebanyak 365 komponen yang memuat dua atau lebih nama beraksara asli sengaja tidak digabung, karena melebur dua orang berbeda lebih merugikan daripada melewatkan satu penggabungan (Pseudocode 3.5).

[[CODE:Pseudocode 3.5 Resolusi alias penulis dengan union-find (src/phases/data_model.py)]]
count[name] <- number of titles per author name
FOR EACH pair (a, b) appearing on the same title, with shared count n:
    different_script <- (a is CJK/Hangul/Kana) XOR (b is CJK/Hangul/Kana)
    contained        <- n = MIN(count[a], count[b])  // the rarer name never appears alone
    IF different_script AND contained: union(a, b)

FOR EACH component K produced by union-find:
    native <- names in K written in a native script
    IF |K| = 1:          resolution <- "single"
    IF |native| = 1:     resolution <- "merged"; canonical <- most frequent romanisation in K
    ELSE:                resolution <- "ambiguous"   // >= 2 native names: deliberately not merged
[[/CODE]]

**Keterpercayaan rating.** Rating ditampilkan sebagai *weighted rating* Bayesian dengan rumus WR = (v / (v + m)) x R + (m / (v + m)) x C, dengan R rating mentah, v jumlah suara, m ambang suara (10), dan C rata-rata global pada populasi tepercaya (3,6522). Judul bernilai 5,0 dari 1 suara ditarik mendekati rata-rata, sedangkan judul bernilai 4,4 dari 280 suara nyaris tidak bergeser. Hanya 23.238 judul (58,6%) yang punya minimal 10 suara, dan populasi inilah yang dipakai untuk peringkat (Pseudocode 3.6). Cakupan sinopsis pada tier metadata lengkap mencapai 98,38% (QLT-04).

[[CODE:Pseudocode 3.6 Bayesian weighted rating (src/phases/phase2_clean.py, add_weighted_rating)]]
m <- 10
reliable <- FILTER df WHERE rating_score IS NOT NULL AND rating_votes >= m
C <- MEAN(reliable.rating_score)                      // 3.6522 over 23,238 titles
IF C IS NULL: FAIL "no reliable ratings"
v <- df.rating_votes
df.weighted_rating <- ROUND((v / (v + m)) * rating_score + (m / (v + m)) * C, 4)
                      IF rating_score IS NOT NULL ELSE NULL
df.is_rated_reliable <- rating_score IS NOT NULL AND v >= m
[[/CODE]]

[[FIG:g3_2_dashboard_governance.png|Gambar 3.2 Halaman Quality & Governance di dashboard: temuan yang perlu ditindaklanjuti dan hasil audit per pilar|10]]

# BAB 4 DATA

## 4.1 Deskripsi Data

### 4.1.1 Sumber dan Volume

Tabel 4.1 merangkum empat berkas sumber di zona raw. Setelah digabung, dideduplikasi, dan dibersihkan, katalog berisi 39.668 judul: 39.652 bermetadata lengkap (tier `full`) dan 16 hanya bermodal teks dan genre (tier `listing`). Total data di tiga zona lake adalah 305,3 MB dalam 133 objek.

Volume ini perlu dinyatakan dengan jujur. Sebanyak 305 MB masih muat di satu mesin, sehingga dari sisi ukuran data proyek ini belum termasuk "big". Yang dirancang untuk skala besar adalah arsitekturnya: penyimpanan terdistribusi, pemrosesan terdistribusi, ingest inkremental, dan partisi fisik. Jika katalog tumbuh sepuluh kali lipat, yang perlu ditambah adalah jumlah worker, bukan kode pipeline.

[[TABLE:Tabel 4.1 Berkas sumber data di zona raw]]
| Berkas | Isi | Ukuran | Baris |
| `data_novel.csv` | Katalog awal, 27 kolom metadata | 2,44 MB | 4.461 |
| `novel_dataset_scrapped.csv` | Sinopsis, dihubungkan lewat `url` | 4,89 MB | 5.000 |
| `scraped_listing.csv` | Hasil scraping halaman daftar seri | 42,57 MB | 36.679 |
| `scraped_detail.csv` | Hasil scraping halaman detail seri | 24,72 MB | 36.674 |

### 4.1.2 Elemen Data dan Formatnya

Modul `data_catalog.py` menginventarisasi 28 elemen data dan mengklasifikasikan formatnya. Klasifikasi ditentukan dari struktur isi, bukan dari tipe fisik. Contohnya, `genres` dan `type` sama-sama bertipe string, tetapi `genres` berisi daftar multinilai sehingga termasuk semi-terstruktur. Karena itu kelas format ditulis eksplisit di sebuah registri beserta alasannya, lalu statistiknya (nilai unik, persentase sentinel, contoh nilai) diukur dari data (Pseudocode 4.1). Kolom sumber yang belum terdaftar di registri memicu peringatan, sehingga kolom baru dari scraper tidak bisa lolos tanpa diklasifikasikan. Hasilnya: 20 elemen terstruktur, 6 semi-terstruktur, dan 2 tidak terstruktur (Tabel 4.2).

[[CODE:Pseudocode 4.1 Profil elemen data berbasis registri (src/phases/data_catalog.py)]]
REGISTRY <- [(name, source, format_class, domain, sentinel_rule, rationale), ...]  // 28 elements

FUNCTION build_catalog(raw):
    result <- []
    FOR EACH e IN REGISTRY:
        column <- raw[e.name]
        n_sentinel <- COUNT rows WHERE sentinel_condition(e.rule)(column)
        result.append({e.name, e.format_class, e.domain,
                       n_distinct = COUNT_DISTINCT(column),
                       pct_sentinel = 100 * n_sentinel / row_count,
                       sample = first_non_sentinel_value(column), e.rationale})
    unregistered <- columns(raw) - names(REGISTRY)
    IF unregistered NOT EMPTY: warn("not yet classified", unregistered)
    RETURN result
[[/CODE]]

[[TABLE:Tabel 4.2 Klasifikasi elemen data berdasarkan format]]
| Kelas format | Jumlah | Elemen | Ciri |
| Terstruktur | 20 | `series_id`, `type`, `type_origin`, `language`, `year`, `chapter_count`, `rating_score`, `rating_votes`, `reading_list_count`, lima kolom peringkat, `is_completed`, `licensed`, `completely_translated`, `original_publisher`, `artists`, `english_publisher` | Nilai atomik, satu nilai per sel |
| Semi-terstruktur | 6 | `genres`, `tags`, `authors`, `url`, `status_coo`, `release_frequency` | Ada struktur di dalam string: daftar berkoma, path URL, atau pola angka dan satuan |
| Tidak terstruktur | 2 | `summary` (sinopsis), `title` | Prosa bebas; median sinopsis 133 kata |

Beberapa elemen semi-terstruktur menyimpan informasi yang baru bisa dipakai setelah diurai. `status_coo` ("1,593 Chapters (Completed)") dipakai untuk memulihkan satu `chapter_count` yang terpotong pemisah ribuan. `release_frequency` ("Every 0.9 Day(s)") diurai menjadi `days_per_chapter`, yang lalu dikelompokkan menjadi lima kategori kecepatan rilis. Kolom `authors` juga lebih rumit dari kelihatannya: 14.679 judul mencantumkan dua nama penulis atau lebih, tetapi setelah resolusi alias hanya 1.804 judul yang benar-benar ditulis bersama. Sisanya adalah satu orang yang ditulis dalam dua aksara.

### 4.1.3 Pembersihan dan Organisasi Data

Urutan langkah pembersihan terlihat pada Gambar 3.1: memuat data dari zona raw, memastikan kunci dan judul ada, mengubah sentinel menjadi NULL, deduplikasi berdasarkan `series_id`, join sinopsis, memperluas katalog dengan hasil scraping listing, menggabungkan hasil halaman detail, lalu membentuk fitur turunan. Saat menggabungkan data detail, peringkat aktivitas mingguan dan bulanan dibatasi pada plafon #2000, karena NovelUpdates kini memotong peringkat di angka itu sementara dataset lama tidak. Mencampur keduanya tanpa aturan tidak memicu error apa pun, tetapi membuat peringkat tidak bisa dibandingkan.

Kolom multinilai dipecah menjadi array yang bersih: dipisah per koma, dipangkas spasinya, dibuang elemen kosongnya, dan diunikkan. Genre pertama dijadikan `primary_genre` untuk visualisasi, dan kombinasi genre diurutkan agar {Action, Romance} dan {Romance, Action} dihitung sebagai kombinasi yang sama. Teks sinopsis ditokenisasi dengan pipeline Spark ML (RegexTokenizer lalu StopWordsRemover dengan 191 stopword bahasa Inggris dan istilah domain). Pseudocode 4.2 merangkum deduplikasi dan fitur turunan.

[[CODE:Pseudocode 4.2 Deduplikasi, array multinilai, dan fitur turunan (src/phases/phase2_clean.py)]]
window <- PARTITION BY series_id ORDER BY rating_votes DESC, reading_list_count DESC
df <- FILTER df WHERE row_number() OVER window = 1          // most mature snapshot

FUNCTION to_array(text): RETURN unique(drop_empty(trim(split(text, ","))))
df.genre_list <- to_array(genres); df.tag_list <- to_array(tags)
df.primary_genre <- genre_list[0]
df.genre_combo <- join(sort(genre_list), " + ")

df.days_per_chapter <- extract_number(release_frequency, "([\d.]+)\s*day")
df.release_pace <- BUCKET(days_per_chapter; <=2 Daily, <=7 Weekly, <=30 Monthly,
                          <=180 Slow, otherwise Stalled)
df.length_bucket <- BUCKET(chapter_count; <50, <200, <500, <1000, 1000+)
df.decade <- FLOOR(year / 10) * 10
df.popularity_score <- log1p(rating_votes) + log1p(reading_list_count)
                       IF metadata_tier = "full" ELSE NULL
[[/CODE]]

Zona clean menyimpan ketiga kelas format dalam satu tabel Parquet. Agar variasi format juga nyata di tingkat berkas, `data_variety.py` memecahnya menjadi tiga berkas dengan format yang sesuai sifatnya (Tabel 4.3). Kolom atomik ditulis ke CSV; daftar genre, tag, penulis, serta status dan kecepatan rilis ditulis sebagai objek JSON bersarang; sinopsis ditulis sebagai teks biasa yang dipecah ke empat shard untuk menghindari masalah berkas kecil di HDFS. Setiap berkas dicatat dalam manifest dengan MD5, lalu dibaca ulang untuk memastikan jumlah rekamannya sama dengan sumber (Pseudocode 4.3). Ketiga berkas inilah yang dimuat ke HDFS dan diolah MapReduce.

[[CODE:Pseudocode 4.3 Pemisahan fisik per kelas format (src/phases/data_variety.py)]]
clean <- spark.read.parquet("s3a://clean-data/novels_clean")
write_csv("structured/novel_metrics.csv", clean[ATOMIC_COLUMNS])          // 23 columns
FOR EACH row IN clean:
    write_jsonl("semistructured/novel_labels.jsonl",
        {series_id, title, url, genres: [..], tags: [..], authors: [..],
         status: {raw, chapter_count, is_completed},
         release: {raw, days_per_chapter}})
with_synopsis <- FILTER clean WHERE synopsis IS NOT EMPTY
FOR EACH shard i IN 0..3:
    write_txt("unstructured/synopsis/synopsis-000i.txt",
              lines "series_id<TAB>title<TAB>synopsis" for shard i)
manifest <- {file, class, format, size, md5, record_count}
ASSERT reread(csv) = 39,668 AND reread(jsonl) = 39,668 AND reread(txt) = 39,342
[[/CODE]]

[[TABLE:Tabel 4.3 Berkas hasil pemisahan format]]
| Kelas | Berkas | Format | Baris |
| Terstruktur | `structured/novel_metrics.csv` | CSV, 23 kolom | 39.668 |
| Semi-terstruktur | `semistructured/novel_labels.jsonl` | JSONL, genre/tag/penulis sebagai array | 39.668 |
| Tidak terstruktur | `unstructured/synopsis/synopsis-*.txt` | TXT, 4 shard | 39.342 (yang bersinopsis) |

Ringkasan isi katalog setelah pembersihan: 55,4% judul sudah tamat, 14,4% berlisensi resmi, dan 20,3% selesai diterjemahkan. Terdapat 35 genre dan 772 tag. Genre terbanyak adalah Romance (25.083 judul) dan Fantasy (23.119 judul).

### 4.1.4 Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami bentuk data sebelum memilih model data dan metrik. Semua angka di subbab ini dihitung dari ekspor zona clean (`Dataset/variety/`, hasil eksekusi 29 September 2026). Kecuali disebutkan lain, analisis rating memakai populasi tepercaya (23.238 judul dengan minimal 10 suara), sedangkan analisis kelengkapan dan distribusi memakai tier `full` (39.652 judul). Langkah perhitungannya dirangkum pada Pseudocode 4.4.

[[CODE:Pseudocode 4.4 Alur perhitungan EDA (dari Dataset/variety)]]
m <- read_csv("structured/novel_metrics.csv"); full <- FILTER m WHERE metadata_tier = "full"
labels <- read_jsonl("semistructured/novel_labels.jsonl")
reliable <- FILTER full WHERE is_rated_reliable = 1

missing[k] <- 100 * MEAN(full[k] IS NULL) FOR k IN key_columns
stats[k] <- {count, mean, std, min, Q1, median, Q3, max, skewness} FOR k IN numeric_columns
chapter_upper <- Q3 + 1.5 * (Q3 - Q1); outliers <- COUNT(chapter_count > chapter_upper)

genres <- explode(labels.genres) JOIN reliable ON series_id
rating_by_genre <- GROUP genres BY genre: MEAN(weighted_rating), COUNT   // at least 300 titles
rating_by_length <- GROUP reliable BY length_bucket(chapter_count): MEAN(weighted_rating)
rating_by_pace <- GROUP reliable BY pace_bucket(days_per_chapter): MEAN(weighted_rating)
correlation <- SPEARMAN(reliable[weighted_rating, log1p(votes), log1p(reading_list),
                                 log1p(chapters), log1p(days_per_chapter), year])
words <- tokens [a-z]{3,} across all synopsis shards, stopwords removed; vocabulary <- COUNT_DISTINCT(words)
[[/CODE]]

**Kelengkapan data.** Gambar 4.1 menunjukkan persentase nilai kosong per kolom setelah sentinel dinormalisasi. Kolom jumlah suara, reading list, dan kecepatan rilis hampir selalu terisi. Tiga kolom perlu perhatian: `year` kosong pada 16,48% judul, `chapter_count` pada 14,45%, dan `original_publisher` pada 14,43%. Implikasinya langsung pada analisis: setiap tren per tahun hanya mewakili sekitar 83,5% katalog, dan setiap rata-rata jumlah bab hanya mewakili 85,6%. Menganggap nilai kosong itu sebagai 0, seperti yang dilakukan sumber data, akan menggeser hasil secara sistematis. Rating kosong pada 5,36% judul adalah judul yang belum pernah dinilai.

[[FIG:eda_1_kelengkapan.png|Gambar 4.1 Persentase nilai kosong per kolom pada tier full setelah normalisasi sentinel|13]]

**Distribusi variabel numerik.** Statistik deskriptif pada Tabel 4.4 memperlihatkan bahwa hampir semua variabel numerik sangat menceng ke kanan. Jumlah bab punya median 135 tetapi rata-rata 291,5 dan maksimum 14.276, dengan skewness 7,53. Dengan aturan IQR (Q3 + 1,5 x IQR = 595,5 bab), sebanyak 3.822 judul (11,27% dari yang jumlah babnya diketahui) tergolong outlier panjang. Judul-judul ini tidak dibuang karena mereka nyata, tetapi karena itulah median dan skala logaritma dipakai di analisis berikutnya, bukan rata-rata dan skala linear. Jumlah suara dan reading list menceng lebih ekstrem lagi (skewness 9,46 dan 4,88), yang menjelaskan mengapa korelasi dihitung dengan Spearman atas nilai log.

[[TABLE:Tabel 4.4 Statistik deskriptif variabel numerik (tier full)]]
| Variabel | n | Rata-rata | Std | Min | Median | Q3 | Maks | Skewness |
| `chapter_count` | 33.924 | 291,48 | 563,33 | 1 | 135 | 279 | 14.276 | 7,53 |
| `rating_score` | 37.526 | 3,49 | 0,82 | 1,0 | 3,6 | 4,0 | 5,0 | -0,95 |
| `weighted_rating` | 37.526 | 3,64 | 0,35 | 1,81 | 3,62 | 3,85 | 4,88 | -0,01 |
| `rating_votes` | 39.652 | 56,90 | 141,40 | 0 | 15 | 56 | 4.225 | 9,46 |
| `reading_list_count` | 39.648 | 974,96 | 2.045,73 | 1 | 200 | 946 | 44.569 | 4,88 |
| `days_per_chapter` | 39.651 | 218,25 | 767,99 | 0,1 | 25,6 | 101,3 | 20.610 | 8,26 |
| `year` | 33.118 | 2021,09 | 5,11 | 1740 | 2022 | 2024 | 2026 | -8,28 |

Gambar 4.2 memperlihatkan empat distribusi kunci. Panel kanan atas paling penting untuk rancangan metrik. Rating mentah punya lonjakan tajam di 1,0 dan 5,0 yang berasal dari judul bersuara sedikit: 1.587 judul bernilai 5,0 dari kurang dari 10 suara, sementara hanya 2 judul yang bernilai 5,0 dengan minimal 10 suara. Setelah weighted rating diterapkan, lonjakan itu hilang, simpangan baku turun dari 0,821 menjadi 0,351, dan distribusinya hampir simetris (skewness -0,01). Karena itu rating mentah tidak dipakai untuk peringkat. Panel kiri bawah menunjukkan bahwa median suara hanya 15, dengan garis ambang 10 suara membelah katalog menjadi dua bagian yang hampir sama besar.

[[FIG:eda_2_distribusi.png|Gambar 4.2 Distribusi jumlah bab, rating mentah dibanding weighted rating, jumlah suara, dan panjang sinopsis|15]]

**Data tidak terstruktur: sinopsis.** Sebanyak 39.342 judul punya sinopsis, dengan median 133 kata, rata-rata 161,3 kata, dan persentil ke-90 sebesar 314 kata. Sekitar 11,9% sinopsis kurang dari 50 kata, terlalu pendek untuk dijadikan satu-satunya dasar klasifikasi. Setelah tokenisasi (huruf kecil, minimal tiga huruf) dan pembuangan 191 stopword, kosakata korpus berisi 62.416 kata unik. Angka ini sama persis dengan jumlah baris keluaran job MapReduce `wordcount`, yang menjadi pemeriksaan silang independen antara jalur Python lokal dan jalur Hadoop. Kata paling sering adalah "one", "world", "life", "love", dan "family". Kata-kata ini generik dan muncul di hampir semua genre, sehingga pada tahap pemodelan bobot TF-IDF dipakai untuk menurunkan pengaruh kata yang terlalu umum.

**Data semi-terstruktur: genre dan tag.** Setiap judul menyandang rata-rata 5,19 genre dan 15,41 tag (median 11) dari 772 tag berbeda. Tag paling sering bersifat deskriptif tokoh dan latar: Male Protagonist (14.088), Female Protagonist (13.770), Transmigration (10.667), dan Handsome Male Lead (10.140). Gambar 4.3 menampilkan 15 genre terbesar beserta rata-rata weighted rating-nya. Genre terbanyak ternyata bukan genre dengan rating tertinggi. Romance menyandang 25.083 judul (63,2% katalog) dengan rata-rata 3,71. Di antara genre dengan minimal 300 judul tepercaya, rata-rata tertinggi dimiliki Shounen Ai (3,904), Yaoi (3,879), Historical (3,814), Mystery (3,808), dan Horror (3,784), sedangkan terendah adalah Ecchi (3,438), Harem (3,503), Adult (3,554), Xuanhuan (3,564), dan Shounen (3,583). Karena satu judul dihitung di beberapa genre, jumlah semua batang melebihi total katalog; fakta inilah yang mendorong penggunaan bridge table di model data (Subbab 4.2).

[[FIG:eda_3_genre.png|Gambar 4.3 Lima belas genre terbesar beserta rata-rata weighted rating (WR) judul tepercaya|14]]

**Variabel kategorikal lain.** Pada populasi tepercaya, judul yang sudah tamat rata-rata dinilai lebih tinggi (3,738) daripada yang masih berjalan (3,594), dan judul berlisensi resmi sedikit lebih tinggi (3,733) daripada yang tidak (3,671). Menurut asal negara, judul Tiongkok unggul (3,733, n = 12.445) atas Korea (3,641, n = 5.541) dan Jepang (3,599, n = 5.130). Format didominasi Web Novel (36.829 judul), disusul Light Novel (1.574) dan Published Novel (686).

**Panjang dan kecepatan rilis terhadap rating.** Gambar 4.4 menguji dua dugaan yang sering muncul di komunitas pembaca. Dugaan pertama, judul yang lebih panjang lebih disukai, tidak didukung data. Rata-rata weighted rating stabil sekitar 3,73 untuk judul di bawah 200 bab, lalu turun konsisten ke 3,659 (200 sampai 499 bab), 3,584 (500 sampai 999 bab), dan 3,559 (1.000 bab atau lebih). Dugaan kedua, judul yang rilis lebih cepat lebih disukai, juga tidak didukung: kelompok rilis lambat (1 sampai 6 bulan per bab) justru memiliki rata-rata tertinggi (3,714), dan selisih antarkelompok tidak membentuk pola yang berurutan. Bagi tim lisensi, ini berarti panjang karya dan kecepatan rilis tidak bisa dipakai sebagai pengganti ukuran kualitas.

[[FIG:eda_4_panjang_rilis.png|Gambar 4.4 Rata-rata weighted rating menurut kelompok panjang dan kecepatan rilis (populasi tepercaya)|14]]

**Korelasi antarvariabel.** Matriks korelasi Spearman pada Gambar 4.5 dihitung atas nilai log untuk variabel yang menceng. Weighted rating berkorelasi sedang dengan jumlah suara (0,43) dan reading list (0,44): judul yang disukai cenderung juga banyak dibaca. Korelasi rating dengan panjang karya lemah dan negatif (-0,13), sejalan dengan Gambar 4.4. Dua temuan lain penting untuk tahap pemodelan. Pertama, jumlah suara dan reading list berkorelasi sangat kuat (0,91), sehingga keduanya hampir redundan dan tidak boleh diperlakukan sebagai dua sinyal independen. Kedua, tahun rilis berkorelasi negatif dengan reading list (-0,52) dan jumlah suara (-0,41). Judul baru belum sempat mengumpulkan pembaca, sehingga membandingkan popularitas judul lintas tahun tanpa koreksi waktu paparan akan bias terhadap judul lama.

[[FIG:eda_5_korelasi.png|Gambar 4.5 Matriks korelasi Spearman antarvariabel numerik pada populasi tepercaya|11]]

**Tren waktu.** Gambar 4.6 memperlihatkan dua tren sejak 2010. Rata-rata weighted rating per tahun rilis turun dari 3,785 (2011) menjadi 3,570 (2025). Penurunan ini perlu dibaca hati-hati. Sebagian mencerminkan efek paparan yang sama: judul baru bersuara sedikit, sehingga rating mereka lebih banyak ditarik ke arah rata-rata global dan lebih sensitif terhadap beberapa suara awal. Panel kanan menunjukkan median jumlah bab per tahun berkisar 130 sampai 175 bab sejak 2016, sedangkan proporsi judul tamat memuncak pada 70,5% untuk judul tahun 2020 dan turun ke 41,8% untuk judul tahun 2025, karena banyak judul baru memang masih berjalan. Dimensi waktu di star schema dirancang untuk menjawab pertanyaan seperti ini per genre dan per asal negara.

[[FIG:eda_6_tren.png|Gambar 4.6 Tren rata-rata weighted rating, median jumlah bab, dan persentase judul tamat per tahun rilis|14]]

**Kesimpulan EDA untuk rancangan.** Temuan di atas langsung memengaruhi keputusan model data. Distribusi yang menceng menuntut median dan skala log di tampilan analitik. Rating mentah yang bias menuntut weighted rating dan filter populasi tepercaya sebagai bawaan. Genre yang multinilai menuntut bridge table agar ukuran tidak tergandakan. Nilai kosong pada tahun dan jumlah bab menuntut dimensi waktu yang hanya berisi tahun sah. Korelasi kuat antara suara dan reading list, serta bias waktu paparan, dicatat sebagai batasan untuk tahap pemodelan di Assignment II.

## 4.2 Model Data

Satu model data tidak cukup untuk katalog ini, karena setiap kelas format menuntut operasi yang berbeda. Tabel 4.5 memetakan karakteristik data ke model dan operasi yang dipilih.

[[TABLE:Tabel 4.5 Pemetaan karakteristik data, model data, dan operasi]]
| Karakteristik data | Model data | Operasi utama |
| Metrik dan dimensi atomik (terstruktur) | Star schema: `fact_novel` + 6 tabel dimensi | Filter, star join, GROUP BY, slice, dice, roll-up, drill-down, pivot |
| Genre, tag, penulis multinilai (semi-terstruktur) | Bridge table hasil `explode`; dokumen JSONL | GROUP BY per nilai tanpa `LIKE`, map-reduce per elemen array |
| Relasi antarentitas (novel, genre, tag, penulis) | Property graph: `graph_nodes` + `graph_edges` | Traversal, PageRank, komunitas, komponen terhubung, jalur terpendek |
| Sinopsis prosa (tidak terstruktur) | Matriks fitur TF-IDF dan embedding kalimat | Kemiripan kosinus, klasifikasi multilabel, klasterisasi |

**Star schema.** Tabel fakta `fact_novel` memiliki *grain* satu baris per judul (Gambar 4.7). Grain ini dipilih dengan sengaja. Jika grain dibuat per pasangan judul dan genre, ukuran seperti `rating_votes` akan tergandakan untuk judul bergenre banyak, dan total suara katalog membengkak dua sampai tiga kali lipat. Dimensi bernilai tunggal (`dim_novel`, `dim_time`, `dim_publisher`) terhubung langsung ke fakta. Dimensi bernilai banyak (`dim_genre`, `dim_tag`, `dim_author`) terhubung lewat bridge table. `dim_time` hanya berisi tahun yang sah, sesuai temuan EDA bahwa 16,48% tahun kosong.

Kunci pengganti (*surrogate key*) diberikan secara deterministik, diurutkan menurut jumlah judul lalu nama, bukan dengan `monotonically_increasing_id()`. Dengan cara ini kunci yang sama keluar di setiap eksekusi, sehingga artefak dari dua eksekusi berbeda tetap bisa dibandingkan. Efek sampingnya berguna: `publisher_key <= 15` berarti tepat 15 penerbit terbesar (Pseudocode 4.5).

[[FIG:g4_2_star_schema.png|Gambar 4.7 Star schema katalog web novel beserta jumlah baris setiap tabel|16]]

[[CODE:Pseudocode 4.5 Pembangunan dimensi, bridge, dan fakta (src/phases/data_model.py)]]
FUNCTION surrogate_key(df, natural_key, key_name):
    RETURN df + row_number() OVER (ORDER BY n_novels DESC, natural_key ASC) AS key_name

FUNCTION explode_dimension(novel, array_col, value_name, key_name):
    pairs <- DISTINCT(series_id, explode(array_col)) WHERE value IS NOT EMPTY
    dim <- surrogate_key(GROUP pairs BY value: n_novels = COUNT, value_name, key_name)
    bridge <- pairs JOIN dim ON value -> (series_id, key_name)
    RETURN dim, bridge

dim_genre, bridge_genre <- explode_dimension(novel, genre_list, "genre", "genre_key")
dim_tag, bridge_tag     <- explode_dimension(novel, tag_list, "tag", "tag_key")
dim_author, bridge_author <- resolve_aliases_then_explode(novel.authors)   // Pseudocode 3.5
dim_publisher <- surrogate_key(GROUP novel BY original_publisher)
dim_time <- DISTINCT(year, decade) WHERE year IS NOT NULL                  // valid years only
fact_novel <- novel JOIN dim_publisher -> (series_id, publisher_key, year, decade,
              additive measures, non-additive measures, text measures, flags)  // 1 row per title
[[/CODE]]

Ukuran di tabel fakta dibedakan menurut sifat agregasinya. `chapter_count`, `rating_votes`, dan `reading_list_count` bersifat aditif, sehingga boleh dijumlahkan. `weighted_rating`, `rating_score`, dan `days_per_chapter` bersifat non-aditif: menjumlahkan rating lintas judul tidak bermakna, jadi yang sah hanya rata-rata atau median.

**Operasi OLAP di dashboard.** Halaman *Data Warehouse* di dashboard menjawab pertanyaan analis langsung di atas star schema (Gambar 4.8). Pengguna memilih dimensi baris, dimensi pembanding (opsional), dan ukuran. Setiap tampilan dijalankan sebagai operasi OLAP berikut, dan dashboard menyebutkan operasi yang sedang aktif:

- **Slice**: satu dimensi dikunci pada satu nilai, misalnya hanya judul asal Tiongkok.
- **Dice**: beberapa dimensi dikunci sekaligus, misalnya asal CN, status tamat, dan tahun 2015 sampai 2026.
- **Roll-up dan drill-down**: hierarki waktu tahun ke dekade. Memilih dekade 2010 menampilkan rincian per tahun di dalamnya.
- **Pivot**: dua dimensi disilangkan menjadi matriks, misalnya genre dikali asal negara.

Logika agregasinya dirangkum pada Pseudocode 4.6. Dua aturan dijaga di sini. Ukuran non-aditif hanya ditampilkan untuk grup berisi minimal 10 judul, supaya tidak ada rata-rata yang bertumpu pada segelintir judul. Ukuran aditif yang dipecah menurut dimensi bernilai banyak diberi peringatan, karena jumlah semua batangnya melebihi total katalog.

[[CODE:Pseudocode 4.6 Agregasi OLAP di halaman Data Warehouse (app/streamlit_app.py)]]
FUNCTION olap_view(row_dim, col_dim, measure, filters):
    facts <- fact_novel JOIN dim_novel JOIN dim_publisher        // one row per title
    facts <- apply filters on origin, format, status, year, >= 10 votes   // slice or dice
    frame <- (series_id, row_dim value)                          // exploded via bridge if multi-valued
    IF col_dim EXISTS: frame <- frame JOIN (series_id, col_dim value)       // pivot
    result <- GROUP frame JOIN facts BY dimensions:
                  aggregate(measure), n_titles = COUNT_DISTINCT(series_id)
    IF measure is non-additive: result <- FILTER result WHERE COUNT(measure values) >= 10
    IF row_dim = "Decade" AND a decade is selected:              // drill-down
        show the same aggregation per Year within that decade
    IF measure is additive AND a multi-valued dimension is used:
        warn("bar total > catalogue total; read as reach per group")
    show(chart(result), equivalent_sql(row_dim, col_dim, measure, filters))
[[/CODE]]

Dashboard juga menampilkan SQL setara untuk setiap tampilan. Contoh berikut adalah SQL untuk pivot genre dikali asal negara dengan ukuran rata-rata weighted rating pada judul tamat yang bersuara minimal 10:

[[CODE]]
SELECT   g.genre, n.type_origin,
         ROUND(AVG(f.weighted_rating), 3) AS measure,
         COUNT(DISTINCT f.series_id)      AS n_titles
FROM     fact_novel f
JOIN     dim_novel n           ON n.series_id = f.series_id
JOIN     bridge_novel_genre bg ON bg.series_id = f.series_id
JOIN     dim_genre g           ON g.genre_key = bg.genre_key
WHERE    n.is_completed = 1 AND f.is_rated_reliable
  AND    n.type_origin IS NOT NULL
GROUP BY g.genre, n.type_origin
HAVING   COUNT(f.weighted_rating) >= 10
ORDER BY measure DESC
[[/CODE]]

Kebenaran SQL ini diuji dengan menjalankannya di Spark dan membandingkan hasilnya dengan perhitungan dashboard. Pada lima kombinasi dimensi, ukuran, dan filter yang berbeda, seluruh grup cocok dengan jumlah judul identik; selisih nilai hanya berasal dari pembulatan `ROUND`. Selain itu, 588 kombinasi dimensi dan ukuran dijalankan tanpa error, dan jumlah judul per genre sama persis dengan kolom `n_novels` di `dim_genre`. Ketika ukuran aditif dipecah menurut genre, jumlah semua batang (misalnya 13.403.084 suara) melebihi total suara katalog, karena satu judul terhitung sekali di setiap genrenya.

[[FIG:g4_5_dashboard_olap.png|Gambar 4.8 Halaman Data Warehouse: rata-rata weighted rating per genre, dengan genre tertinggi ditandai|8]]

Contoh hasil operasi pivot genre dikali asal negara (dimensi Genre dibandingkan dengan Country of origin di dashboard): di antara judul dengan rating tepercaya, judul asal Tiongkok memiliki rata-rata tertinggi pada sebagian besar dari 12 genre terbesar, misalnya Yaoi (3,92) dan Slice of Life (3,82). Judul Korea justru unggul pada Action (3,69 dibanding 3,66) dan Harem (3,59 dibanding 3,48). Temuan seperti ini langsung relevan bagi tim lisensi yang memilih judul per segmen.

**Property graph.** Relasi antarentitas dimodelkan sebagai property graph dengan empat label simpul dan tiga tipe sisi (Gambar 4.9): 67.805 simpul dan 858.327 sisi. Setiap simpul diberi awalan tipe (`N:`, `G:`, `T:`, `A:`) supaya kunci genre 7 dan kunci tag 7 tidak bertabrakan menjadi satu simpul.

[[FIG:g4_3_property_graph.png|Gambar 4.9 Skema property graph dan dua proyeksi yang dianalisis|11.5]]

Dari graf ini dibuat dua proyeksi yang masing-masing menjawab pertanyaan berbeda. Proyeksi genre dibangun dengan self-join tabel `bridge_novel_genre` di Spark: dua genre dihubungkan bila minimal 5 judul menyandang keduanya, dan bobot sisinya adalah jumlah judul bersama. Proyeksi penulis hanya memakai judul dengan dua penulis atau lebih, sehingga sisinya adalah kolaborasi sungguhan, bukan sekadar kemiripan genre. Kedua graf lalu dianalisis dengan NetworkX (Pseudocode 4.7). Betweenness dihitung dengan sampel 200 simpul sumber karena perhitungan eksaknya berkompleksitas O(V x E).

[[CODE:Pseudocode 4.7 Proyeksi dan analisis graf (src/phases/graph_analysis.py)]]
// projection in Spark (distributed)
pairs <- bridge_novel_genre a JOIN bridge_novel_genre b
         ON a.series_id = b.series_id AND a.genre_key < b.genre_key
genre_edges <- GROUP pairs BY (a.genre_key, b.genre_key): weight = COUNT
               WHERE weight >= 5
multi <- titles in bridge_novel_author with >= 2 authors
author_edges <- DISTINCT author pairs on the same title IN multi

// in-memory analysis (NetworkX)
FOR EACH G IN {genre_graph(genre_edges), author_graph(author_edges)}:
    pr <- pagerank(G, weight)                          // influence from structure
    dc <- degree_centrality(G)
    bc <- betweenness_centrality(G, k = MIN(200, |V|), seed = 42)
    components <- connected_components(G)
    communities <- greedy_modularity_communities(G, weight)
    Q <- modularity(G, communities)
    save(pr, dc, bc, community_id) TO analytics/graph_*_pagerank
// traversal: find two non-adjacent genres linked through one intermediary
path <- shortest_path(genre_graph, g1, g2)   // proves depth <= 2 hops
[[/CODE]]

Hasil analisis kedua proyeksi dirangkum pada Tabel 4.6.

[[TABLE:Tabel 4.6 Hasil analisis dua proyeksi graf]]
| Ukuran | Ko-okurensi genre | Kolaborasi penulis |
| Simpul / sisi | 35 / 579 | 2.578 / 2.699 |
| Komponen terhubung | 1 | 931 |
| Komponen terbesar | 35 simpul | 22 penulis |
| Komunitas (greedy modularity) | 3 | 931 |
| Modularity | 0,0857 | 0,9947 |
| Tafsiran | Genre saling berbaur, label genre saja lemah untuk membedakan judul | Tim penulis hampir tidak pernah berkolaborasi lintas tim |

Pada graf genre, PageRank tertinggi dimiliki Romance (0,0966) dan Fantasy (0,0927). Keduanya sering muncul dan terhubung dengan genre lain yang juga berpengaruh; PageRank memperhitungkan kedua hal itu. Kedekatan antar-genre diukur dengan koefisien Jaccard (jumlah judul bersama dibagi jumlah judul yang menyandang salah satunya), bukan jumlah mentah, supaya genre besar tidak otomatis tampak dekat dengan semua genre lain. Pasangan terdekat adalah Action dan Adventure (Jaccard 0,518), lalu Romance dan Drama (0,426). Modularity graf genre yang rendah (0,0857) sejalan dengan temuan EDA bahwa satu judul rata-rata menyandang lima genre: genre tidak membentuk kelompok yang tegas.

Halaman *Relationship Graph* di dashboard menampilkan peta genre berwarna komunitas dengan ambang Jaccard yang bisa diatur, serta pencari rute antar-genre (Gambar 4.10). Pada ambang 0,15, rute dari Yaoi ke Martial Arts membutuhkan tiga hop (Yaoi, Romance, Action, Martial Arts). Tanpa ambang, rute terpanjang dalam graf (diameter) hanya dua hop: dari 595 pasangan genre, hanya 16 yang tidak bertetangga langsung, dan semuanya terhubung lewat satu genre perantara, misalnya Yaoi ke Yuri. Pengukuran ini membuktikan klaim kedalaman dua hop yang menjadi alasan tidak memakai graph database terpisah (Subbab 2.2).

[[FIG:g4_6_dashboard_graf.png|Gambar 4.10 Halaman Relationship Graph: peta genre (ukuran titik = PageRank, warna = komunitas) dan peringkat PageRank|11]]

**Model untuk data semi-terstruktur dan tidak terstruktur.** Untuk jalur MapReduce, data semi-terstruktur disimpan sebagai dokumen JSONL, satu objek per judul dengan genre, tag, dan penulis sebagai array (Pseudocode 4.3). Mapper cukup mengiterasi array tersebut, tanpa perlu mengurai string berkoma. Sinopsis diubah menjadi matriks fitur: TF-IDF untuk menangkap frasa khas seperti "cultivation", dan embedding kalimat pralatih untuk menangkap parafrasa. Pilihan TF-IDF didukung temuan EDA bahwa kata tersering di korpus bersifat generik ("one", "world", "life"), sehingga perlu dibobot turun. Matriks inilah yang dipakai untuk kemiripan kosinus pada rekomendasi dan klasifikasi genre multilabel. Evaluasi model tersebut dibahas pada Assignment II.

# REFERENSI

- Clauset, A., Newman, M. E. J., & Moore, C. (2004). Finding community structure in very large networks. *Physical Review E*, 70(6), 066111.
- Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on large clusters. *Communications of the ACM*, 51(1), 107-113.
- Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling* (3rd ed.). Wiley.
- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab.
- Republik Indonesia. (2014). Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta.
- Republik Indonesia. (2022). Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Zaharia, M., Xin, R. S., Wendell, P., et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56-65.
- NovelUpdates. Direktori web novel terjemahan. https://www.novelupdates.com
