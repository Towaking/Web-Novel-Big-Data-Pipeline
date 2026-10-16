# Audit Tata Kelola IT (dihasilkan otomatis)

> Dihasilkan `src/phases/governance_audit.py` pada 2026-10-15 13:48 UTC. **Jangan diedit manual.** Berkas ini memuat hasil pemeriksaan; argumentasi kebijakan untuk Bab 3 laporan ditulis terpisah oleh penulis.

**Ringkasan:** 17 lulus · 0 peringatan · 0 gagal · 6 informasi

## 3.1 Security

| ID | Pemeriksaan | Status | Bukti |
|---|---|---|---|
| `SEC-01` | Kredensial bukan bawaan | **LULUS** | kredensial khusus, rahasia >= 16 karakter, berbeda dari .env.example |
| `SEC-02` | Enkripsi in-transit (TLS) | **LULUS** | TLSv1.3, sertifikat server terverifikasi terhadap CA stack |
| `SEC-03/raw-data` | Bucket 'raw-data' tertutup dari anonim | **LULUS** | tidak ada policy anonim |
| `SEC-03/clean-data` | Bucket 'clean-data' tertutup dari anonim | **LULUS** | tidak ada policy anonim |
| `SEC-03/curated-data` | Bucket 'curated-data' tertutup dari anonim | **LULUS** | tidak ada policy anonim |
| `SEC-04` | Pemisahan hak baca/tulis | **LULUS** | akun read-only 'novel-dashboard' tersedia untuk dashboard |
| `SEC-05` | Isolasi jaringan antar-service | **INFO** | seluruh service berada pada bridge network 'bigdata-net'; port UI hanya dipetakan ke 127.0.0.1 host (HOST_BIND) |

## 3.2 Compliance

| ID | Pemeriksaan | Status | Bukti |
|---|---|---|---|
| `CMP-01` | Provenance terekam (checksum + waktu) | **LULUS** | novel/novel_catalogue.csv: md5=377569db..., 39670 baris |
| `CMP-02` | Raw zone tidak pernah ditimpa transformasi | **LULUS** | 2 objek di raw-data/novel/ = 1 sumber + manifest, ukuran tiap sumber sama dengan manifest ingestion |
| `CMP-03` | Versioning bucket (pemulihan & audit) | **LULUS** | raw-data=Enabled, clean-data=Enabled, curated-data=Enabled |
| `CMP-04` | Sumber data terdokumentasi | **INFO** | Dataset berasal dari katalog publik NovelUpdates (metadata karya + sinopsis hasil scraping). Metadata katalog bersifat faktual; teks sinopsis tetap karya berhak cipta pemiliknya dan hanya dipakai untuk analitik non-komersial |

**Tindak lanjut:**

- `CMP-04` — Status lisensi redistribusi harus ditegaskan sebelum dataset ini dipublikasikan ulang — keputusan ini ada di penulis laporan, bukan di kode.

## 3.3 Data Privacy

| ID | Pemeriksaan | Status | Bukti |
|---|---|---|---|
| `PRV-01` | Pemindaian PII langsung (email/telepon) | **LULUS** | tidak ada kecocokan pada seluruh kolom teks (novel_catalogue.csv) |
| `PRV-02` | Pola angka mirip identitas (perlu verifikasi) | **INFO** | tidak ada |
| `PRV-03` | Tidak ada data tingkat individu | **LULUS** | tidak ada kolom ber-grain pengguna; sinyal pembaca hanya tersedia sebagai agregat (`rating_votes`, `reading_list_count`) sehingga tidak ada subjek data yang dapat diidentifikasi ulang |
| `PRV-04` | Nama orang yang ikut terproses | **INFO** | kolom `authors` memuat nama pena yang dipublikasikan platform sebagai kredit karya (bukan identitas privat); `artists` dibuang di Fase 2 karena 82,36% kosong atau Unknown |

**Tindak lanjut:**

- `PRV-02` — Kecocokan pada kolom prosa umumnya deretan angka biasa, bukan identitas.
- `PRV-04` — Nama pena tetap data pribadi bila dapat dikaitkan ke orang nyata — dasar pemrosesannya perlu dinyatakan di Bab 3.3.

## 3.4 Data Quality

| ID | Pemeriksaan | Status | Bukti |
|---|---|---|---|
| `QLT-01` | Katalog bersih memuat seluruh baris sumber (>= 99,9%) | **LULUS** | 39,668 dari 39,670 baris sumber (99.99%) \| full=39,652 \| listing=16 \| total=39,668 |
| `QLT-01b` | Baris tier 'listing' ditandai, bukan disamarkan | **LULUS** | 16 novel hanya bermodal teks & genre; rating/bab/tahun/popularitas NULL di semua baris itu (diperiksa, bukan diasumsikan). Setiap metrik dari halaman detail WAJIB memfilter metadata_tier = 'full' |
| `QLT-02` | Kunci `series_id` unik | **LULUS** | 39,668 unik dari 39,668 |
| `QLT-03` | Sentinel sudah dinormalisasi jadi NULL | **LULUS** | tidak ada sentinel tersisa di clean zone |
| `QLT-04` | Cakupan sinopsis inti katalog >= 98% | **LULUS** | 98.38% (39,011/39,652) pada tier 'full' |
| `QLT-04b` | Cakupan sinopsis baris perluasan | **INFO** | 100.00% (16/16) |
| `QLT-05` | Populasi rating tepercaya dilaporkan | **INFO** | 23,256 novel (>= 10 suara) dari 39,668 — agregasi rating di Fase 3 memakai populasi ini atau weighted rating, bukan rata-rata mentah |
| `QLT-06` | Rating berada di rentang sah (0-5) | **LULUS** | semua nilai dalam rentang |
