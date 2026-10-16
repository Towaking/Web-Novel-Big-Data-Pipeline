# Inventaris Elemen Data (dihasilkan otomatis)

> Dihasilkan `src/phases/data_catalog.py` pada 2026-10-06 20:33 UTC. **Jangan diedit manual** — jalankan ulang modulnya. Angka di sini terukur dari data; klasifikasi formatnya adalah keputusan desain yang alasannya ada di `ELEMENT_REGISTRY`.

## Rekap per kelas format

| Kelas format | Jumlah elemen |
|---|---:|
| Terstruktur | 20 |
| Semi-terstruktur | 6 |
| Tidak terstruktur | 2 |

## Inventaris lengkap

| Elemen | Sumber | Tipe fisik | Kelas format | Domain | Unik | Sentinel | Contoh |
|---|---|---|---|---|---:|---:|---|
| `series_id` | novel_catalogue.csv | integer | **Terstruktur** | Identitas | 39,652 | 0.04% | 155 |
| `url` | novel_catalogue.csv | string | **Semi-terstruktur** | Identitas | 39,670 | 0.0% | https://www.novelupdates.com/series/%c2%bd-prince/ |
| `title` | novel_catalogue.csv | string | **Tidak terstruktur** | Identitas | 39,636 | 0.0% | Half Prince |
| `genres` | novel_catalogue.csv | string | **Semi-terstruktur** | Label | 18,256 | 0.0% | Action, Adventure, Comedy, Gender Bender, Romance, Sci-fi |
| `tags` | novel_catalogue.csv | string | **Semi-terstruktur** | Label | 36,829 | 1.92% | Adapted to Manhua, Artificial Intelligence, Clingy Lover, College/Univ... |
| `synopsis` | novel_catalogue.csv | string | **Tidak terstruktur** | Deskripsi | 39,284 | 0.82% | During one of our arguments, my younger brother taunted me, claiming t... |
| `type` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 5 | 1.45% | Published Novel |
| `type_origin` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 11 | 1.48% | CN |
| `language` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 10 | 0.12% | Chinese |
| `status_coo` | novel_catalogue.csv | string | **Semi-terstruktur** | Dimensi | 19,450 | 7.08% | 9 Volumes / 76 Chapters (Complete) |
| `authors` | novel_catalogue.csv | string | **Semi-terstruktur** | Dimensi | 28,426 | 3.29% | Yu Wo, 御我 |
| `original_publisher` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 3,124 | 14.44% | Shueisha |
| `artists` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 3,306 | 82.36% | Funatsu Kazuki |
| `english_publisher` | novel_catalogue.csv | string | **Terstruktur** | Dimensi | 113 | 93.28% | J-Novel Club |
| `year` | novel_catalogue.csv | integer | **Terstruktur** | Temporal | 159 | 1.93% | 2004 |
| `release_frequency` | novel_catalogue.csv | string | **Semi-terstruktur** | Temporal | 5,305 | 0.04% | Every 337.6 Day(s) |
| `chapter_count` | novel_catalogue.csv | integer | **Terstruktur** | Metrik | 2,180 | 14.46% | 76 |
| `rating_score` | novel_catalogue.csv | double | **Terstruktur** | Metrik | 41 | 5.37% | 4.0 |
| `rating_votes` | novel_catalogue.csv | integer | **Terstruktur** | Metrik | 894 | 5.37% | 278 |
| `reading_list_count` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 5,532 | 0.04% | 3874 |
| `activity_week_rank` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 628 | 0.04% | 8522 |
| `activity_month_rank` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 8,753 | 0.04% | 15131 |
| `activity_all_rank` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 26,968 | 0.04% | 5791 |
| `reading_list_month_rank` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 14,336 | 0.04% | 10983 |
| `reading_list_all_rank` | novel_catalogue.csv | integer | **Terstruktur** | Popularitas | 22,602 | 0.04% | 2530 |
| `is_completed` | novel_catalogue.csv | integer (0/1) | **Terstruktur** | Flag | 2 | 0.0% | 0 |
| `licensed` | novel_catalogue.csv | integer (0/1) | **Terstruktur** | Flag | 2 | 58.55% | 0 |
| `completely_translated` | novel_catalogue.csv | integer (0/1) | **Terstruktur** | Flag | 2 | 0.04% | 1 |

## Alasan klasifikasi

- **`series_id`** → Terstruktur. Kunci primer alami: integer unik, atomik, tanpa struktur internal. Dipakai sebagai kunci dedup karena `title` tidak unik (ada judul kembar).
- **`url`** → Semi-terstruktur. Path hierarkis (`/series/<slug>/`) — nilainya membawa struktur yang bisa dipecah jadi segmen. Kunci penyatuan sumber scrape, bukan atribut analitik.
- **`title`** → Tidak terstruktur. Teks bahasa alami bebas, tanpa skema internal. Pendek, tetapi tetap perlu normalisasi/tokenisasi sebelum dipakai model — ikut masuk korpus Fase 4.
- **`genres`** → Semi-terstruktur. Daftar multi-nilai berpemisah `, ` dalam satu sel. Skemanya tetap, isinya tidak atomik — harus di-split jadi array sebelum bisa di-`explode` atau dihitung frekuensinya. Menjadi label target Fase 4 Head A.
- **`tags`** → Semi-terstruktur. Sama seperti `genres` — daftar berpemisah koma, tetapi jauh lebih padat (median 12 tag). Label semantik ternormalisasi hasil kurasi komunitas.
- **`synopsis`** → Tidak terstruktur. Sinopsis prosa bebas. Inti data tidak terstruktur di proyek ini: baru berguna setelah tokenisasi + TF-IDF/embedding di Fase 2 & 4.
- **`type`** → Terstruktur. Kategori tertutup berkardinalitas rendah, satu nilai per baris.
- **`type_origin`** → Terstruktur. Negara/asal terbitan — kategori atomik, dipakai sebagai sumbu agregasi Fase 3.
- **`language`** → Terstruktur. Kategori atomik berkardinalitas sangat rendah.
- **`status_coo`** → Semi-terstruktur. Campuran teks status dan pola angka+satuan (`"182 Chapters (Completed)"`). Fase 2 mengekstrak angka babnya dengan regex untuk memulihkan satu `chapter_count` yang terpotong pemisah ribuan — bukti nyata bahwa kolom ini menyimpan struktur di dalam string.
- **`authors`** → Semi-terstruktur. Daftar multi-nilai berpemisah koma — satu novel bisa punya beberapa penulis. Berisi nama pena yang dipublikasikan platform, bukan identitas privat; tetap dicatat di sini karena relevan untuk Bab 3.3.
- **`original_publisher`** → Terstruktur. Nama badan usaha, satu nilai per baris. Penerbit yang tidak diketahui dijadikan null di Fase 2.
- **`artists`** → Terstruktur. DIBUANG di Fase 2: hampir seluruhnya Unknown, tidak cukup sinyal untuk analitik apa pun.
- **`english_publisher`** → Terstruktur. DIBUANG di Fase 2: hampir seluruhnya Unknown.
- **`year`** → Terstruktur. Tahun terbit — skalar ordinal. Sentinel `0` TIDAK dapat dipulihkan dari kolom lain, jadi dijadikan null dan dikecualikan dari analisis temporal (data_contract §5.4).
- **`release_frequency`** → Semi-terstruktur. Kalimat berpola `"Every 0.9 Day(s)"` — angka dan satuan tertanam di dalam teks. 100% baris dapat diurai regex menjadi `days_per_chapter`, dan hasilnya berkorelasi dengan rating sekuat jumlah bab. Contoh paling jelas kenapa 'bertipe string' tidak sama dengan 'tidak terstruktur'.
- **`chapter_count`** → Terstruktur. Ukuran numerik aditif — langsung bisa di-`AVG`/`SUM`. Sentinel `0` dijadikan null lebih dulu, kalau tidak rata-rata panjang katalog turun palsu.
- **`rating_score`** → Terstruktur. Skalar kontinu 0–5. Sentinel `0` berarti *belum pernah dinilai*, bukan *dinilai buruk* — dijadikan null, kalau tidak seluruh agregat rating bias ke bawah.
- **`rating_votes`** → Terstruktur. Jumlah suara — dipakai sebagai **bobot kepercayaan**, bukan metrik berdiri sendiri. Menjadi `v` pada rumus Bayesian weighted rating.
- **`reading_list_count`** → Terstruktur. Cacah agregat berapa pembaca memasukkan novel ke daftar baca. Ini angka gabungan — tidak ada baris per pengguna, sehingga tidak ada PII sama sekali (lihat catatan privasi di bawah).
- **`activity_week_rank`** → Terstruktur. Peringkat aktivitas mingguan — ordinal, makin kecil makin populer.
- **`activity_month_rank`** → Terstruktur. Peringkat aktivitas bulanan.
- **`activity_all_rank`** → Terstruktur. Peringkat aktivitas sepanjang masa.
- **`reading_list_month_rank`** → Terstruktur. Peringkat daftar baca bulanan.
- **`reading_list_all_rank`** → Terstruktur. Peringkat daftar baca sepanjang masa.
- **`is_completed`** → Terstruktur. Boolean tersandi angka — atomik.
- **`licensed`** → Terstruktur. Boolean tersandi angka. Relevan untuk Bab 3.2 (compliance): menandai judul yang punya lisensi resmi berbahasa Inggris.
- **`completely_translated`** → Terstruktur. Boolean tersandi angka.

## Sentinel value per elemen

| Elemen | Penyandian kekosongan |
|---|---|
| `series_id` | tidak ada — selalu terisi |
| `url` | tidak ada |
| `title` | tidak ada |
| `genres` | string kosong bila tidak berlabel |
| `tags` | 'Unknown' pada baris tanpa tag |
| `synopsis` | string kosong; <10 kata dianggap bukan deskripsi |
| `type` | 'Unknown' / 'N/A' |
| `type_origin` | 'Unknown' / 'N/A' |
| `language` | 'Unknown' / 'N/A' |
| `status_coo` | 'Unknown' / 'N/A' |
| `authors` | 'Unknown' / 'N/A' |
| `original_publisher` | 'Unknown' / 'N/A' |
| `artists` | 'Unknown' / 'N/A' |
| `english_publisher` | 'Unknown' / 'N/A' |
| `year` | 0 atau kosong |
| `release_frequency` | tidak ada — selalu terisi, tetapi perlu parsing |
| `chapter_count` | 0 atau kosong |
| `rating_score` | 0 atau kosong |
| `rating_votes` | 0 sejalan dengan rating_score |
| `reading_list_count` | 0 berarti nol pembaca, bukan hilang |
| `activity_week_rank` | 0 = tidak masuk peringkat |
| `activity_month_rank` | 0 = tidak masuk peringkat |
| `activity_all_rank` | 0 = tidak masuk peringkat |
| `reading_list_month_rank` | 0 = tidak masuk peringkat |
| `reading_list_all_rank` | 0 = tidak masuk peringkat |
| `is_completed` | tidak ada; 0 berarti belum tamat |
| `licensed` | kosong bila halaman detail menulis N/A |
| `completely_translated` | tidak ada |
