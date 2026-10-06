# Kontrak Data — dua sumber

Dokumen ini mengunci **skema sumber** dan **keputusan pemetaan kolom** yang
dipakai seluruh fase pipeline (1–5). Setiap fase wajib mengacu ke sini agar
tidak ada asumsi kolom yang berbeda antar tahap.

Status: **disepakati** · Profil dijalankan 2026-08-29 / diperbarui 2026-08-30

| Berkas | Bentuk | Peran |
|---|---|---|
| `Dataset/data_novel.csv` | 4.461 × 27 | Tabel fakta: metadata, rating, jumlah bab, tag |
| `Dataset/novel_dataset_scrapped.csv` | 5.000 × 4 | Tabel dimensi: **sinopsis**, di-join lewat `url` |

---

## 1. Penyatuan dua sumber

Dataset utama **tidak memiliki kolom `synopsis`**; kolom itu datang dari berkas
kedua dengan nama `summary`. Keduanya digabung di Fase 2 lewat `url`.

| Spesifikasi awal | Sumber aktual | Keputusan |
|---|---|---|
| `synopsis` | `novel_dataset_scrapped.csv` → `summary` | **Dipakai** sebagai korpus utama, digabung lewat `url` |
| `rating` | `data_novel.csv` → `rating_score` | Dipakai dengan `rating_votes` sebagai bobot kepercayaan |
| `title`, `genres`, `chapter_count` | `data_novel.csv` | Sesuai |

**Kualitas join (terukur, bukan asumsi):**

- Cakupan **100%** — seluruh 4.461 `url` dataset utama punya pasangan.
- Setelah menyaring sinopsis <10 kata, cakupan efektif **98,45%** (4.392 judul).
- `summary` rata-rata **136 kata** (median 118, maksimum 914).

**Tiga jebakan yang ditangani sebelum join:**

1. **538 baris `url` duplikat** di berkas sinopsis (499 URL unik, isinya salinan
   persis). Tanpa dedup, tabel fakta menggelembung 4.461 → 4.999 baris dan
   **setiap agregat di Fase 3 ikut salah**. Aturan: sinopsis terpanjang menang.
2. **37 `summary` kosong + 79 di bawah 10 kata** → ditandai `has_synopsis = false`,
   bukan dibuang, agar statistik katalog tetap mewakili katalog.
3. **721 baris dengan `genres` berbeda antar berkas.** `data_novel.csv`
   dijadikan sumber kebenaran karena ia yang memuat rating dan metadata lain;
   kolom `title`/`genres` dari berkas sinopsis diabaikan.

**Peran `tags` setelah sinopsis tersedia:** tetap dipakai, tidak digantikan.
Sinopsis memberi konteks naratif (prosa), tag memberi label semantik
ternormalisasi (`Time Travel`, `Weak to Strong`) dengan presisi tinggi.
Keduanya masuk korpus pemodelan; novel tanpa sinopsis masih punya sinyal tag.

**Dua korpus dibangun terpisah di Fase 2:**

| Kolom | Isi | Dipakai oleh |
|---|---|---|
| `corpus_clean` | `title` + `synopsis` + `tags` | Fase 4 (seluruh model) |
| `synopsis_clean` | `synopsis` saja | Fase 3 (word cloud, korelasi panjang teks) |

Pemisahan ini perlu: word cloud dari korpus gabungan hanya menampilkan frasa
katalog seperti *"male protagonist"* dan menutupi kosakata cerita sebenarnya.

---

## 2. Skema sumber (27 kolom)

| Kolom | Tipe | Peran di pipeline |
|---|---|---|
| `title` | string | Identitas + pelengkap korpus teks |
| `genres` | string (dipisah `, `) | Label multi-kelas, analitik kategorikal (Fase 3) |
| `tags` | string (dipisah `, `) | **Korpus teks utama** (Fase 2 & 4) |
| `series_id` | int | **Kunci unik** untuk deduplikasi |
| `url` | string | Metadata, tidak dipakai analitik |
| `type`, `type_origin`, `language` | string | Dimensi agregasi |
| `authors`, `artists`, `original_publisher`, `english_publisher` | string | Dimensi opsional (banyak `Unknown`) |
| `year`, `status_coo` | int / string | Dimensi temporal & status |
| `chapter_count` | int | Metrik numerik inti |
| `is_completed`, `licensed`, `completely_translated` | 0/1 | Flag biner |
| `release_frequency` | string | Perlu parsing (`Every 0.9 Day(s)`) bila dipakai |
| `rating_score`, `rating_votes` | float / int | **Target & bobot rating** |
| `activity_week_rank`, `activity_month_rank`, `activity_all_rank` | int | Sinyal popularitas |
| `reading_list_count`, `reading_list_month_rank`, `reading_list_all_rank` | int | Sinyal popularitas |

---

## 3. Masalah kualitas data (temuan profil)

**Tidak ada NaN sama sekali** — nilai kosong disamarkan sebagai *sentinel*.
Ini jebakan utama: `dropna()` tidak akan menangkap apa pun.

| Sentinel | Jumlah | Porsi | Penanganan di Fase 2 |
|---|---|---|---|
| `chapter_count == 0` | 902 | 20,22% | Jadikan null → jangan ikut hitung rata-rata panjang |
| `year == 0` | 1.011 | 22,66% | Jadikan null → kecualikan dari analisis temporal |
| `rating_score == 0` (dan `rating_votes == 0`) | 248 | 5,56% | Novel belum dinilai → null, bukan rating rendah |
| `artists == "Unknown"` | 3.777 | 84,67% | Kolom di-drop dari analitik |
| `english_publisher == "Unknown"` | 4.181 | 93,72% | Kolom di-drop dari analitik |
| `original_publisher == "Unknown"` | 803 | 18,00% | Pertahankan, perlakukan sebagai kategori tersendiri |
| `tags == "Unknown"` | 105 | 2,35% | Baris dikecualikan dari pemodelan berbasis teks |
| `status_coo == "Unknown"` | 393 | 8,81% | Kategori tersendiri |

**Duplikasi:** 0 baris identik, 0 `series_id` ganda, 1 `title` ganda (dua seri
berbeda berjudul sama). → Deduplikasi memakai `series_id`, **bukan** `title`.

**Bias rating:** hanya 2.777 baris (62%) punya `rating_score > 0` dengan minimal
5 suara; 2.058 baris (46%) dengan minimal 10 suara. Rata-rata rating mentah
menyesatkan karena novel bersuara 1–2 mendominasi ekor distribusi. Fase 3 & 4
memakai **Bayesian weighted rating**:

```
WR = (v / (v + m)) * R + (m / (v + m)) * C
     v = rating_votes, R = rating_score
     m = ambang suara minimum (default 10)
     C = rata-rata rating global pada populasi tervalidasi
```

---

## 4. Aturan turunan untuk fase berikutnya

1. **Fase 2 (Clean):** ubah sentinel → null *sebelum* imputasi/agregasi; dedup by
   `series_id`; pecah `genres` dan `tags` menjadi array; bangun kolom
   `text_corpus = title + tags` yang sudah ditokenisasi & dibersihkan stopwords.
2. **Fase 3 (Explore):** seluruh agregasi rating memakai populasi
   `rating_votes >= 10` atau *weighted rating*; laporkan jumlah baris yang
   dikecualikan di setiap kueri agar hasil tetap jujur.
3. **Fase 4 (Model):** fitur teks dari `text_corpus`; label/target ditentukan saat
   fase tersebut dieksekusi; baris `tags == "Unknown"` dikeluarkan dari training.
4. **Fase 5 (Deploy):** visualisasi menampilkan populasi yang dipakai
   (n tervalidasi vs n total) supaya pembaca dashboard tidak salah tafsir.

---

## 5. Audit lanjutan (2026-08-30) — apa lagi yang bisa diperbaiki

Bagian 1–4 mengunci keputusan yang sudah dipakai sejak awal. Bagian ini
melaporkan audit berikutnya: setiap dugaan perbaikan diuji dengan angka, lalu
**diterima atau ditolak secara eksplisit**. Yang ditolak tetap ditulis, karena
tanpa itu orang berikutnya akan mencoba hal yang sama lagi.

### 5.1 Diterima — `release_frequency` diurai jadi angka

Kolom ini sebelumnya hanya lewat sebagai teks (`"Every 0.9 Day(s)"`) dan tidak
dipakai fase mana pun.

| Ukuran | Nilai |
|---|---|
| Baris yang berhasil diurai | **4.460 dari 4.461 (100%)** — sisanya `Unknown` |
| Sebaran hari/bab | median 37,3 · p10 3,0 · p90 630,8 · maks 20.610 |
| Korelasi `log(hari/bab)` ~ `rating_score` (n=2.058, ≥10 suara) | r = −0,089 |
| Korelasi `log(hari/bab)` ~ `weighted_rating` (kueri Fase 3) | **r = −0,099** |

Korelasinya **lebih kuat** daripada korelasi jumlah bab dengan rating (−0,0875)
yang selama ini jadi temuan utama Fase 3. Artinya: novel yang bab barunya lama
datang dinilai sedikit lebih rendah. Dua kolom turunan ditambahkan di Fase 2:

- `days_per_chapter` (double) — dipakai Head D sebagai fitur.
- `release_pace` (string, 5 kelompok) — dipakai analitik Fase 3.

Fitur ini **bukan** fitur popularitas: kecepatan terbit adalah keputusan
penulis/penerjemah, bukan turunan jumlah suara, jadi tidak membocorkan target
seperti `rating_votes`.

### 5.2 Diterima — satu `chapter_count` terpotong pemisah ribuan

`status_coo` menuliskan jumlah bab lengkap dengan koma. Dibandingkan pada
3.021 baris yang kedua sumbernya terisi: **3.020 cocok persis, 1 tidak.**

| Judul | `chapter_count` | `status_coo` | Seharusnya |
|---|---|---|---|
| The Quick Transmigrating Cannon Fodder Chooses… | 593 | `1,593 Chapters (Completed)` | **1.593** |

Diperbaiki di Fase 2, dengan syarat ketat: hanya bila digit ekor angka utuh
sama persis dengan nilai kolom (ciri pemotongan pada koma). Tanpa syarat itu,
perbedaan wajar — status basi vs kolom yang lebih baru — ikut tertimpa.

### 5.3 Ditolak — sentinel `chapter_count = 0` tidak dapat dipulihkan

Dugaan: 902 baris ber-`chapter_count = 0` masih menyimpan angkanya di
`status_coo`. **Salah.** Dari 902 baris itu, yang `status_coo`-nya memuat
jumlah bab: **0**; yang memuat jumlah volume: **0**. Sentinel ini memang
berarti benar-benar tidak diketahui.

Catatan terkait: 538 baris ber-`chapter_count > 0` tidak punya angka bab di
`status_coo` karena satuannya volume (`"2 Volumes (Complete)"`) — light novel
terbit per jilid, bukan per bab. Bukan kesalahan data.

### 5.4 Ditolak — `year = 0` juga tidak dapat dipulihkan

Dari 1.011 baris sentinel, hanya **1** yang `status_coo`-nya memuat pola tahun.
Tidak ada sumber pemulihan di dalam baris.

### 5.5 Ditolak — menggabungkan genre dari kedua berkas

Berkas sinopsis juga punya kolom `genres`, dan 649 baris (14,5%) berbeda dari
tabel fakta. Terlihat seperti label yang bisa dipulihkan — ternyata bukan:

| Genre "tambahan" | Jumlah |
|---|---|
| **Completed** | **556** |
| Supernatural | 6 |
| Adventure | 5 |
| sisanya (11 genre) | ≤3 masing-masing |

**`Completed` adalah status penerjemahan, bukan genre.** Setelah dikeluarkan,
seluruh sisa tambahan hanya ~35 label pada 4.461 novel. Kardinalitas label
praktis tidak berubah (4,24 → 4,26 per novel). Keputusan bagian 1 —
`data_novel.csv` sebagai sumber kebenaran — tetap berlaku.

### 5.6 Ditolak — melengkapi genre yang hilang dari `tags`

Hipotesis: sebagian genre yang benar tidak dituliskan kurator, dan `tags`
(759 tag unik, jauh lebih rinci daripada 33 genre) bisa memulihkannya.

Diuji dengan aturan implikasi empiris — tag dianggap mengimplikasikan genre
bila `P(genre | tag) ≥ 0,80` **dan** lift ≥ 1,8. Terkumpul 62 aturan; sebuah
genre dianggap hilang bila **≥2 tag berbeda** menyepakatinya.

| Ukuran | Nilai |
|---|---|
| Positif tercatat | 18.787 |
| Positif hilang (dugaan) | **108 (+0,6%)** |
| Novel terdampak | 106 dari 4.460 (2,4%) |

Terlalu kecil untuk mengubah apa pun, dan 97 dari 108 hanya pada satu genre
(`Action`). **Label pada dataset ini tidak sistematis tidak lengkap.**

Pemeriksaan silang memperkuatnya: di antara 7.287 positif palsu model,
hanya **0,9%** didukung ≥2 tag implikatif. Angka itu memang ~25× lebih tinggi
daripada sel negatif biasa (0,04%) — jadi arahnya benar, sebagian kecil
"kesalahan" model memang label yang hilang — tetapi besarannya jauh terlalu
kecil untuk menjelaskan jarak menuju batas atas.

**Konsekuensi:** batas atas ~0,653 makro-F1 di [model_card.md](model_card.md)
**bukan** akibat label yang tidak lengkap, melainkan label yang memang
subjektif. Menaikkannya butuh pelabelan ulang, bukan pembersihan otomatis.

### 5.7 Catatan — ruang label mencampur tiga taksonomi

Dari 33 genre, yang dipakai model 24 (≥50 novel). Yang dibuang sebagian besar
adalah **label demografi**, bukan genre: `Josei` (29), `Shounen` (18),
`Seinen` (15), `Shoujo` (4). Di antara 24 yang dipakai pun masih tercampur:

| Jenis label | Contoh | F1 model |
|---|---|---|
| Genre sungguhan | Romance, Fantasy, Action | 0,78–0,84 |
| Rating konten | Mature, Adult, Ecchi | 0,32–0,42 |
| Nada cerita | Psychological, Tragedy | 0,33–0,42 |

Tiga jenis label ini ditentukan oleh hal yang berbeda, dan yang bukan genre
sungguhan konsisten mendapat skor terendah. Memisahkannya menjadi beberapa
tugas prediksi adalah arah perbaikan berikutnya yang paling menjanjikan —
belum dikerjakan karena mengubah definisi tugas, bukan sekadar modelnya.

---

## 6. Audit ketiga (2026-09-03) — elemen, model data, dan tata kelola

Bagian ini melengkapi kontrak dengan tiga hal yang sebelumnya hanya tersirat di
kode: klasifikasi format tiap elemen, model data yang dipilih, dan kontrol tata
kelola yang benar-benar terpasang. Ketiganya kini **dihasilkan oleh modul yang
dijalankan**, bukan ditulis tangan, sehingga ikut berubah bila datanya berubah:

| Modul | Menghasilkan |
|---|---|
| `src/phases/data_catalog.py` | `curated-data/governance/data_elements` + `Doc/generated/data_elements.md` |
| `src/phases/data_model.py` | 15 tabel di `curated-data/model/` (star schema, bridge, graf, operasi contoh) |
| `src/phases/governance_audit.py` | `curated-data/governance/audit_report` + `Doc/generated/governance_audit.md` |

### 6.1 Klasifikasi format — 20 / 6 / 2

Dari 28 elemen: **20 terstruktur, 6 semi-terstruktur, 2 tidak terstruktur.**
Yang semi-terstruktur: `genres`, `tags`, `authors` (daftar berpemisah koma),
`release_frequency` (`"Every 0.9 Day(s)"` — angka tertanam di kalimat),
`status_coo` (teks status + pola `"N Chapters"`), dan `url` (path hierarkis).
Yang tidak terstruktur hanya `summary` dan `title`.

Klasifikasi ini **tidak** disimpulkan dari tipe fisik. `genres` dan
`release_frequency` sama-sama `string`, tetapi yang satu daftar dan yang lain
kalimat berpola — tidak ada tipe fisik yang membedakannya.

**Angka sentinel-nya divalidasi silang.** Modul inventaris menghitung ulang
porsi sentinel secara independen dan hasilnya identik dengan §3: `year` 22,66%,
`chapter_count` 20,22%, `rating_score` 5,56%, `artists` 84,67%,
`english_publisher` 93,72%, `status_coo` 8,81%, `original_publisher` 18,00%.

Satu koreksi metodologis yang perlu dicatat: aturan sentinel **wajib
dideklarasikan per kolom**, tidak boleh disimpulkan dari tipe. Percobaan
pertama memperlakukan semua `0` numerik sebagai sentinel, dan akibatnya
`is_completed` terlaporkan 52,63% "hilang" padahal `0` di situ berarti *belum
tamat* — nilai yang sah. Sama untuk `licensed` (94,06%) dan
`completely_translated` (87,25%). Kolom-kolom itu kini beraturan `none`.

### 6.2 `authors` — satu orang, dua aksara

**Temuan baru yang tidak tercatat di audit sebelumnya.** Kolom `authors` bukan
selalu daftar beberapa penulis. Pada **1.494 novel (34,6%)** isinya satu orang
yang ditulis dua kali — nama asli dan romanisasinya:

```
"Mizume, ミズメ"            → satu penulis
"Chae Yeon-sil, 채연실"      → satu penulis
"Tang Jia San Shao, 唐家三少" → satu penulis
```

Diperlakukan apa adanya, dimensi penulis berisi **5.439 entitas untuk 4.461
novel**, dan setiap analitik tingkat penulis menghitung ganda.

**Aturan yang dipakai.** Dua nama dianggap alias bila (a) salah satu beraksara
CJK/Hangul/Kana dan yang lain tidak, DAN (b) yang jarang tidak pernah muncul
tanpa yang sering (`co_occurrence == min(count_a, count_b)`). Syarat (b) memakai
*containment*, bukan "selalu bersama" — nama asli kerap muncul sendirian di
novel lain yang romanisasinya tidak dicatat, sehingga aturan "selalu bersama"
melewatkan 1.510 → hanya menangkap sebagian.

**Batas yang sengaja dipasang.** Menutup alias secara transitif salah: trio
penulis yang selalu berkarya bersama tertarik jadi satu orang —

```
"Aono Hakuto, Imai Santarou, Maraiya Mū, マライヤ・ムー, 今井三太郎, 蒼乃白兎"
   → 3 penulis x 2 aksara, BUKAN 1 orang
```

Karena itu komponen hanya digabung bila mengandung **tepat satu** nama
beraksara asli. Hasilnya:

| Kategori | Komponen | Nama |
|---|---:|---:|
| Tunggal (tak berubah) | 2.674 | 2.674 |
| Digabung (1 nama asli + n romanisasi) | 1.266 | 2.624 |
| **Ambigu — sengaja tidak digabung** | 36 | 141 |
| **Total entitas** | **4.081** | 5.439 |

Yang ambigu tetap terpecah dan ditandai `resolution = "ambigu"` di
`dim_author`, termasuk kasus yang sebenarnya sama seperti `荆柯守`/`荊柯守`
(Mandarin sederhana vs tradisional) dan `Gu Long`/`古龍`/`古龙`. **Melewatkan
penggabungan yang benar lebih murah daripada melebur dua orang berbeda** —
kesalahan jenis pertama menyisakan duplikat yang kelihatan, jenis kedua
menghasilkan entitas palsu yang tidak kelihatan.

### 6.3 Model data — dua model, bukan satu

| Karakteristik data | Model | Operasi yang jadi murah |
|---|---|---|
| Metrik & dimensi atomik | Star schema (`fact_novel` + 6 `dim_*`) | filter, join berbintang, roll-up OLAP |
| Multi-nilai (`genres`/`tags`/`authors`) | 3 bridge table hasil `explode` | `GROUP BY` per nilai, tanpa `LIKE '%..%'` |
| Relasi antar-entitas | Property graph (9.334 simpul, 77.493 sisi) | traversal 2-hop, ko-okurensi, jangkauan |
| Sinopsis prosa | Matriks fitur TF-IDF/embedding (Fase 4) | kemiripan kosinus, klasifikasi |

**Grain tabel fakta = satu baris per novel**, bukan per novel-genre. Grain
per novel-genre akan menggandakan `rating_votes` pada novel bergenre banyak dan
membuat total suara katalog membengkak 2–3x. Analisis per genre tetap bisa
lewat `bridge_novel_genre`, dengan konsekuensi yang disadari: hasilnya aditif
lintas novel, **bukan** lintas genre.

Grafnya tidak dijalankan di database graf terpisah karena semua pertanyaan graf
proyek ini berkedalaman ≤ 2 hop pada 4.461 simpul novel — self-join
menyelesaikannya dalam hitungan detik tanpa menambah satu service pun. Bentuk
`nodes`/`edges` standar tetap dimaterialisasi supaya bisa dimuat ke
Neo4j/GraphFrames tanpa memodelkan ulang bila kebutuhannya berkembang.

### 6.4 Kontrol tata kelola yang terpasang

Audit menjalankan 21 pemeriksaan: **14 lulus, 2 peringatan, 0 gagal.**

Yang benar-benar dipasang (bukan sekadar didokumentasikan):

- **Versioning aktif** di ketiga bucket — pemulihan & jejak audit.
- **Akses anonim ditutup** (`mc anonymous set none`) di ketiga bucket.
- **Akun read-only `novel-dashboard`** dengan policy `novel-readonly`, dipakai
  service streamlit. Diuji nyata: baca berhasil, `PutObject` ditolak
  `AccessDenied`. Ini yang membuat SEC-04 berstatus lulus, bukan klaim.
- Kredensial berjenjang di `MinioConfig` (`MINIO_ACCESS_KEY` → fallback root)
  sehingga least privilege per service tidak butuh perubahan kode fase.

Dua peringatan yang **sengaja diterima** untuk lingkungan lokal, dan tercatat
sebagai tindak lanjut alih-alih disembunyikan: kredensial masih bawaan
(`minioadmin`) dan tidak ada TLS (`http://minio:9000`, hanya melintasi bridge
network Docker). Keduanya wajib ditutup sebelum stack ini keluar dari localhost.

**Privasi:** 0 kecocokan PII langsung (email/telepon) di seluruh kolom teks
kedua berkas, dan tidak ada satu pun kolom ber-*grain* pengguna — sinyal
pembaca hanya tersedia sebagai agregat (`rating_votes`, `reading_list_count`),
sehingga tidak ada subjek data yang dapat diidentifikasi ulang. Yang tetap
perlu dinyatakan dasar pemrosesannya: `authors` memuat nama pena yang
dipublikasikan platform.

### 6.5 Plafon peringkat aktivitas — dua kolom yang TIDAK sebanding lintas waktu

**Ditemukan 2026-09-04, sebelum tahap `detail` dijalankan.**

NovelUpdates kini **memotong** `activity_week_rank` dan `activity_month_rank`
di `#2000`. Nilai itu berarti "di luar 2000 besar", bukan peringkat sebenarnya.
Diverifikasi pada tiga novel dengan popularitas sangat berbeda:

| Novel | week | month | all time | reading list |
|---|---:|---:|---:|---:|
| Emperor's Domination (18.228 pembaca) | **2000** | **2000** | 1 | 18.228 |
| Oya ga Saikon (1.161 pembaca) | **2000** | **2000** | 11.841 | 1.161 |
| 2 Years Until Humanity's Extinct (56 pembaca) | **2000** | **2000** | 47.037 | 56 |

`activity_all_rank` dan kedua peringkat reading list **tidak** terpotong.

**Kenapa ini penting.** `data_novel.csv` diambil sebelum situs berubah dan
tidak punya plafon ini — maksimumnya 19.302 (week) dan 25.077 (month). Jadi:

- Baris lama bernilai 2000 = peringkat 2000 yang sesungguhnya.
- Baris hasil scrape baru bernilai 2000 = "tidak masuk 2000 besar", bisa berarti
  peringkat 40.000.

Membandingkan keduanya tanpa penanganan membuat novel yang sebenarnya sepi
tampak jauh lebih populer daripada novel lama yang benar-benar di peringkat
19.302. Ini kesalahan yang tidak akan melempar error apa pun.

**Aturan yang WAJIB diterapkan saat `detail.csv` digabungkan:**

1. Pada baris yang berasal dari scrape baru, `activity_week_rank == 2000` dan
   `activity_month_rank == 2000` → **NULL**, diperlakukan seperti sentinel
   `chapter_count = 0` dan `year = 0`.
2. Baris `metadata_tier = 'full'` (dari `data_novel.csv`) **tidak** disentuh.
3. Analitik apa pun yang memakai kedua kolom itu harus melaporkan populasinya,
   atau memakai `activity_all_rank` yang bebas plafon.

Sengaja **belum** dipasang di `normalize_sentinels`: kolom peringkat baru masuk
lewat tahap `detail` yang belum diintegrasikan, dan guard di fungsi itu akan
jadi kode mati karena `metadata_tier` belum ada saat ia berjalan.

### 6.6 Perluasan katalog — 4.461 → 39.670

Scrape ulang 2026-09-03/04 memakai `series-ranking` (bukan `series-finder`,
yang meng-clamp di 100 halaman) tanpa parameter `ge=`:

| | Sebelum | Sesudah |
|---|---:|---:|
| Novel | 4.461 | **39.670** |
| Genre unik | 33 | **35** |
| Genre di bawah ambang 50 | 9 | **0** |
| Total label genre | 18.905 | **205.743** |
| Rating tepercaya (≥10 suara) | 2.058 (46,1%) | 2.058 (**5,2%**) |

Sembilan genre yang dulu tersingkir dari model kini jauh di atas ambang —
`Yaoi` 4→5.468, `Shounen` 18→4.519, `Josei` 29→4.001, `Shoujo` 4→3.357,
`Gender Bender` 1→3.024, `Smut` 2→2.520, `Seinen` 15→2.287,
`Shounen Ai` 1→1.902, `Mecha` 43→695 — ditambah dua genre yang sebelumnya
**sama sekali tidak ada**: `Yuri` 0→5.072 dan `Shoujo Ai` 0→700.

**Harga yang harus disadari:** proporsi rating tepercaya anjlok dari 46,1% ke
5,2%. Bukan karena ada yang hilang — jumlah absolutnya tetap 2.058 — melainkan
karena 35.209 baris baru berasal dari halaman listing yang tidak memuat rating
sama sekali. Kolom `metadata_tier` menandainya, dan setiap metrik dari halaman
detail wajib memfilter `metadata_tier = 'full'` sampai tahap `detail` selesai.
