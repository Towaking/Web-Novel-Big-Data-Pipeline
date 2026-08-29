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
