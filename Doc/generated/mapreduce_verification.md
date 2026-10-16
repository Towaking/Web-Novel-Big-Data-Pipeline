# Verifikasi MapReduce vs Spark (dihasilkan otomatis)

> Dihasilkan `scripts/mapreduce/verify_mapreduce.py` pada 2026-10-15 13:54 UTC. **Jangan diedit manual.**

**Hasil:** 4 dari 4 job identik dengan jawaban Spark.

Tiap job MapReduce (Hadoop Streaming, input dari HDFS) dijawab ulang oleh Spark SQL dari zona clean. Kolom *selisih* = baris yang hanya ada di satu sisi + baris dengan nilai berbeda (toleransi relatif 1e-06 untuk bilangan pecahan).

| Job | Format input | Pertanyaan | Baris MR | Baris Spark | Hanya MR | Hanya Spark | Nilai beda | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `wordcount` | TXT (tidak terstruktur) | frekuensi kata pada korpus sinopsis | 62,412 | 62,412 | 0 | 0 | 0 | **LULUS** |
| `genre_stats` | JSONL (semi-terstruktur) | frekuensi genre + ko-okurensi genre berpasangan | 618 | 618 | 0 | 0 | 0 | **LULUS** |
| `origin_stats` | CSV (terstruktur) | cacah, jumlah rating & jumlah bab per negara asal | 10 | 10 | 0 | 0 | 0 | **LULUS** |
| `genre_rating` | CSV x JSONL (join lintas format) | rata-rata weighted rating per genre (reduce-side join) | 35 | 35 | 0 | 0 | 0 | **LULUS** |

## Counter Hadoop

| Job | Map task | Reduce task | Lokalitas | Map output | Combine in → out | Reduce in | Reduce groups | Shuffle bytes | Durasi (s) |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| `wordcount` | 11 | 2 | 11 data-local, 0 rack-local | 3,415,443 | 3,415,443 → 236,329 (93.1% terpangkas) | 236,329 | 62,412 | 2,958,609 | 25 |
| `genre_stats` | 7 | 2 | 7 data-local, 0 rack-local | 717,552 | 717,552 → 4,207 (99.4% terpangkas) | 4,207 | 618 | 99,074 | 19 |
| `origin_stats` | 2 | 1 | 2 data-local, 0 rack-local | 39,668 | 39,668 → 17 (100.0% terpangkas) | 17 | 10 | 621 | 14 |
| `genre_rating` | 2 | 1 | 2 data-local, 0 rack-local | 125,762 | 125,762 → 70 (99.9% terpangkas) | 70 | 35 | 2,307 | 15 |

*Combine in → out* adalah efek combiner: rekaman yang dijumlahkan di sisi map sebelum dikirim lewat jaringan. Job `genre_rating` tahap 1 (join) tidak memakai combiner karena join tidak bisa dijumlahkan sebagian.

*Lokalitas* 22/22 task map data-local: DataNode dan NodeManager berjalan di container yang sama (`worker1`/`worker2` di `docker-compose.yml`), jadi YARN menjadwalkan task di node pemegang bloknya.

## Jawaban job `genre_rating` (genre dengan >= 100 novel bernilai)

| Genre | Rata-rata weighted rating | Novel |
|---|---:|---:|
| Shounen Ai | 3.9035 | 1,333 |
| Yaoi | 3.8789 | 3,875 |
| Historical | 3.8147 | 1,503 |
| Mystery | 3.8081 | 2,220 |
| Horror | 3.7835 | 1,257 |
