"""
FASE 2 — CLEAN (Processing)
===========================
Membaca data mentah dari `s3a://raw-data/`, membersihkannya dengan PySpark,
lalu menulis hasilnya sebagai Parquet ke `s3a://clean-data/`.

Temuan profil data yang menentukan desain modul ini
(detail: `Doc/data_contract.md`):

  * **Tidak ada satu pun NaN.** Nilai kosong disamarkan sebagai *sentinel*:
    angka `0` (chapter_count, year, rating_score) dan string `"Unknown"`.
    `dropna()` polos tidak akan menangkap apa pun — sentinel harus
    dinormalisasi lebih dulu menjadi NULL sebelum agregasi apa pun.
  * **Deduplikasi memakai `series_id`, bukan `title`.** Ada judul kembar milik
    dua seri berbeda; dedup by title membuang baris yang valid.
  * **Rating mentah bias.** 5,56% novel belum pernah dinilai dan mayoritas
    sisanya bersuara sangat sedikit, jadi rata-rata mentah didominasi novel
    bersuara 1-2. Karena itu dihitung *Bayesian weighted rating*.

Seluruh transformasi dikerjakan terdistribusi di DataFrame Spark; tidak ada
`collect()` ke driver selain untuk statistik ringkas yang dicetak ke log.

Jalankan:
    docker compose exec pipeline python -m src.phases.phase2_clean
"""

from __future__ import annotations

import sys

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("phase2.clean")

# --- Parameter pembersihan --------------------------------------------------

#: Ambang suara minimum agar sebuah rating dianggap dapat dipercaya.
MIN_VOTES_RELIABLE = 10

#: Kolom string yang memakai sentinel "Unknown" sebagai pengganti NULL.
UNKNOWN_SENTINEL_COLS = (
    "type",
    "type_origin",
    "tags",
    "language",
    "authors",
    "artists",
    "original_publisher",
    "english_publisher",
    "status_coo",
    "release_frequency",
)

#: Kolom yang dibuang: >80% nilainya "Unknown" sehingga tidak punya daya analitik.
DROP_COLS = ("artists", "english_publisher")

#: Stopword domain-spesifik di luar daftar bahasa Inggris bawaan Spark.
#: Kata-kata ini muncul di hampir semua judul/tag sehingga tidak membedakan apa pun.
DOMAIN_STOPWORDS = (
    "novel",
    "novels",
    "story",
    "stories",
    "chapter",
    "chapters",
    "volume",
    "series",
    "web",
    "light",
    "the",
    "and",
)

#: Skema eksplisit — inferSchema dihindari karena (a) butuh satu pass ekstra
#: membaca seluruh file dan (b) tipe hasil inferensi bisa berubah bila sampel
#: datanya berubah, sehingga pipeline jadi tidak deterministik.
#: Skema dataset pendamping berisi sinopsis (`novel_dataset_scrapped.csv`).
#: Kolom `title` dan `genres` di file ini sengaja TIDAK dipakai: 721 barisnya
#: berbeda dari dataset utama, dan dataset utama-lah yang memuat rating serta
#: metadata lain, sehingga ia yang dijadikan sumber kebenaran.
SYNOPSIS_SCHEMA = StructType(
    [
        StructField("title", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("url", StringType(), True),
    ]
)

RAW_SCHEMA = StructType(
    [
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("url", StringType(), True),
        StructField("series_id", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("type_origin", StringType(), True),
        StructField("tags", StringType(), True),
        StructField("language", StringType(), True),
        StructField("authors", StringType(), True),
        StructField("artists", StringType(), True),
        StructField("original_publisher", StringType(), True),
        StructField("english_publisher", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("status_coo", StringType(), True),
        StructField("chapter_count", IntegerType(), True),
        StructField("is_completed", IntegerType(), True),
        StructField("licensed", IntegerType(), True),
        StructField("completely_translated", IntegerType(), True),
        StructField("release_frequency", StringType(), True),
        StructField("rating_score", DoubleType(), True),
        StructField("rating_votes", IntegerType(), True),
        StructField("activity_week_rank", IntegerType(), True),
        StructField("activity_month_rank", IntegerType(), True),
        StructField("activity_all_rank", IntegerType(), True),
        StructField("reading_list_count", IntegerType(), True),
        StructField("reading_list_month_rank", IntegerType(), True),
        StructField("reading_list_all_rank", IntegerType(), True),
    ]
)


# --- Langkah-langkah transformasi ------------------------------------------


def read_raw(spark: SparkSession) -> DataFrame:
    """Baca CSV mentah dari landing zone MinIO dengan skema eksplisit."""
    settings = get_settings()
    uri = settings.minio.s3a_uri(settings.minio.bucket_raw, settings.raw_object_key)
    logger.info("Membaca data mentah dari %s", uri)

    df = (
        spark.read.option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        # multiLine wajib: kolom `tags` dan `title` berisi teks berkutip yang
        # dapat memuat newline; tanpa ini satu baris logis terpecah jadi banyak.
        .option("multiLine", "true")
        .option("mode", "PERMISSIVE")
        .schema(RAW_SCHEMA)
        .csv(uri)
    )
    return df


def read_synopsis(spark: SparkSession) -> DataFrame:
    """
    Baca dataset sinopsis dan siapkan sebagai tabel dimensi yang siap di-join.

    Dua masalah pada sumber ini yang harus diselesaikan SEBELUM join:

      1. **538 baris URL duplikat.** Menggabungkannya langsung akan menggandakan
         baris fakta (4.461 -> 4.999) dan diam-diam menggelembungkan setiap
         hitungan agregat di Fase 3.
      2. **37 sinopsis kosong.** Dinormalkan menjadi NULL agar konsisten dengan
         penanganan sentinel di dataset utama.

    Aturan dedup: pertahankan sinopsis terpanjang per URL. Pada snapshot ini
    seluruh duplikat isinya identik, tetapi aturan "terpanjang menang" tetap
    deterministik bila suatu saat salah satu salinan terpotong.
    """
    settings = get_settings()
    uri = settings.minio.s3a_uri(settings.minio.bucket_raw, settings.raw_synopsis_key)
    logger.info("Membaca dataset sinopsis dari %s", uri)

    raw = (
        spark.read.option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", "true")  # sinopsis prosa memuat newline di dalam kutip
        .option("mode", "PERMISSIVE")
        .schema(SYNOPSIS_SCHEMA)
        .csv(uri)
    )

    cleaned = raw.select(
        F.trim(F.col("url")).alias("url"),
        F.when(F.length(F.trim(F.col("summary"))) == 0, F.lit(None))
        .otherwise(F.trim(F.col("summary")))
        .alias("synopsis"),
    ).filter(F.col("url").isNotNull())

    window = Window.partitionBy("url").orderBy(F.length(F.col("synopsis")).desc_nulls_last())
    deduped = (
        cleaned.withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )

    n_raw, n_dedup = raw.count(), deduped.count()
    logger.info("Sinopsis: %s baris mentah -> %s URL unik (-%s duplikat).",
                f"{n_raw:,}", f"{n_dedup:,}", f"{n_raw - n_dedup:,}")
    return deduped


def attach_synopsis(df: DataFrame, synopsis: DataFrame) -> DataFrame:
    """
    Gabungkan sinopsis ke tabel fakta lewat `url` (LEFT JOIN).

    LEFT, bukan INNER: novel tanpa pasangan sinopsis tetap harus ikut dalam
    analitik rating/panjang bab — membuangnya akan membuat statistik katalog
    tidak lagi mewakili katalog.
    """
    joined = df.join(F.broadcast(synopsis), on="url", how="left")
    return (
        joined.withColumn(
            "synopsis_word_count",
            F.when(F.col("synopsis").isNull(), F.lit(None)).otherwise(
                F.size(F.split(F.trim(F.col("synopsis")), r"\s+"))
            ),
        )
        # Sinopsis <10 kata (79 baris) terlalu pendek untuk dianggap deskripsi
        # bermakna - ditandai agar analitik teks bisa mengecualikannya.
        .withColumn(
            "has_synopsis",
            (F.col("synopsis").isNotNull() & (F.col("synopsis_word_count") >= 10)).cast("boolean"),
        )
    )


def normalize_sentinels(df: DataFrame) -> DataFrame:
    """
    Ubah sentinel menjadi NULL sejati.

    Ini langkah paling menentukan di seluruh fase: tanpa normalisasi ini,
    `avg(chapter_count)` ikut menghitung 902 novel bernilai 0 dan hasilnya
    turun ~20% dari nilai sebenarnya.
    """
    out = df

    # (a) Sentinel string "Unknown" (case-insensitive) -> NULL
    for col in UNKNOWN_SENTINEL_COLS:
        out = out.withColumn(
            col,
            F.when(F.lower(F.trim(F.col(col))) == "unknown", F.lit(None)).otherwise(
                F.trim(F.col(col))
            ),
        )

    # (b0) Perbaiki jumlah bab yang terpotong pemisah ribuan.
    #
    #      `status_coo` menuliskan jumlah bab lengkap dengan koma
    #      ("1,593 Chapters (Completed)"). Pada satu baris, kolom
    #      `chapter_count` berisi 593 sementara status_coo menyebut 1.593 —
    #      pola khas angka yang terpotong pada koma saat pengambilan data.
    #      Diverifikasi pada 3.021 baris yang kedua sumbernya sama-sama terisi:
    #      3.020 cocok persis, 1 tidak. Yang satu itu diperbaiki di sini karena
    #      sumber kebenarannya masih ada di dalam baris yang sama.
    chapters_in_status = F.regexp_extract(
        F.regexp_replace(F.col("status_coo"), ",", ""), r"(\d+)\s*[Cc]hapters?", 1
    ).cast("int")
    out = out.withColumn(
        "chapter_count",
        F.when(
            chapters_in_status.isNotNull()
            & (F.col("chapter_count") > 0)
            & (chapters_in_status > F.col("chapter_count"))
            # Hanya bila digit ekornya sama — ciri pemotongan pada koma.
            # Tanpa syarat ini, perbedaan wajar (status basi vs kolom terbaru)
            # ikut tertimpa.
            & (chapters_in_status % F.lit(1000) == F.col("chapter_count")),
            chapters_in_status,
        ).otherwise(F.col("chapter_count")),
    )

    # (b) Sentinel numerik 0 -> NULL (0 bab / tahun 0 mustahil secara domain)
    out = out.withColumn(
        "chapter_count",
        F.when(F.col("chapter_count") <= 0, F.lit(None)).otherwise(F.col("chapter_count")),
    )
    out = out.withColumn(
        "year",
        F.when((F.col("year") <= 0) | (F.col("year") > 2030), F.lit(None)).otherwise(F.col("year")),
    )

    # (c) rating_score 0 dengan 0 suara = "belum dinilai", bukan "nilainya jelek".
    #     Membiarkannya sebagai 0 akan menarik turun seluruh rata-rata genre.
    out = out.withColumn(
        "rating_score",
        F.when(
            (F.col("rating_score") <= 0) | (F.col("rating_votes").isNull()) | (F.col("rating_votes") <= 0),
            F.lit(None),
        ).otherwise(F.col("rating_score")),
    )

    # (d) String kosong -> NULL agar konsisten dengan sentinel lain
    for field in out.schema.fields:
        if isinstance(field.dataType, StringType):
            out = out.withColumn(
                field.name,
                F.when(F.length(F.trim(F.col(field.name))) == 0, F.lit(None)).otherwise(
                    F.col(field.name)
                ),
            )
    return out


def deduplicate(df: DataFrame) -> DataFrame:
    """
    Deduplikasi berbasis `series_id` (kunci alami dataset).

    Bila ada ganda, baris dengan jumlah suara terbanyak dipertahankan karena
    itu snapshot paling matang dari seri yang sama.
    """
    window = Window.partitionBy("series_id").orderBy(
        F.col("rating_votes").desc_nulls_last(),
        F.col("reading_list_count").desc_nulls_last(),
    )
    return (
        df.withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def parse_multivalue(df: DataFrame) -> DataFrame:
    """Pecah kolom multi-nilai (`genres`, `tags`) menjadi array yang bersih."""

    def to_array(col_name: str):
        # Pisah per koma, buang spasi berlebih, buang elemen kosong.
        return F.array_distinct(
            F.filter(
                F.transform(F.split(F.col(col_name), ","), lambda x: F.trim(x)),
                lambda x: F.length(x) > 0,
            )
        )

    return (
        df.withColumn("genre_list", F.coalesce(to_array("genres"), F.array()))
        .withColumn("tag_list", F.coalesce(to_array("tags"), F.array()))
        .withColumn("n_genres", F.size("genre_list"))
        .withColumn("n_tags", F.size("tag_list"))
        # Genre pertama dipakai sebagai label utama pada visualisasi bivariat.
        .withColumn(
            "primary_genre",
            F.when(F.size("genre_list") > 0, F.col("genre_list").getItem(0)).otherwise(F.lit(None)),
        )
        # Kombinasi genre diurutkan agar {Action, Romance} dan {Romance, Action}
        # dihitung sebagai kombinasi yang sama pada analisis Fase 3.
        .withColumn("genre_combo", F.concat_ws(" + ", F.array_sort("genre_list")))
    )


def build_text_pipeline(input_col: str, output_col: str) -> Pipeline:
    """
    Susun pipeline text preprocessing Spark ML untuk satu kolom teks.

    Tahapan: tokenisasi regex -> stopwords removal.
    Memakai `Pipeline` (bukan transformasi manual) supaya tahapan yang sama
    bisa dipakai ulang secara identik pada korpus yang berbeda, dan dicerminkan
    persis oleh jalur inferensi di `src/common/text_prep.py`.
    """
    tokenizer = RegexTokenizer(
        inputCol=input_col,
        outputCol=f"{output_col}_raw",
        pattern=r"\s+",
        minTokenLength=3,
        toLowercase=True,
    )
    remover = StopWordsRemover(
        inputCol=f"{output_col}_raw",
        outputCol=output_col,
        stopWords=list(StopWordsRemover.loadDefaultStopWords("english")) + list(DOMAIN_STOPWORDS),
    )
    return Pipeline(stages=[tokenizer, remover])


def _normalize_text_column(col_name: str):
    """Lowercase, buang karakter non-huruf (angka, tanda baca, CJK), rapatkan spasi."""
    return F.regexp_replace(
        F.regexp_replace(F.lower(F.col(col_name)), r"[^a-z\s]", " "), r"\s+", " "
    )


def apply_text_processing(df: DataFrame) -> DataFrame:
    """
    Bangun DUA korpus teks dan proses keduanya menjadi token bersih.

    1. `tokens` / `corpus_clean` — gabungan `title` + `synopsis` + `tags`.
       Ini korpus untuk pemodelan (Fase 4): sinopsis memberi konteks naratif,
       tag memberi label semantik yang sudah ternormalisasi. Keduanya saling
       melengkapi, dan novel tanpa sinopsis tetap punya sinyal dari tag.

    2. `synopsis_tokens` / `synopsis_clean` — sinopsis saja.
       Dipisah karena analitik teks Fase 3 (word cloud, korelasi panjang
       deskripsi) harus mengukur prosa asli. Bila dicampur tag, hasilnya
       didominasi frasa katalog seperti "male protagonist" dan bukan
       kosakata cerita.
    """
    with_corpus = (
        df.withColumn(
            "corpus_raw",
            F.concat_ws(
                " ",
                F.coalesce(F.col("title"), F.lit("")),
                F.coalesce(F.col("synopsis"), F.lit("")),
                F.coalesce(F.col("tags"), F.lit("")),
            ),
        )
        .withColumn("corpus_normalized", _normalize_text_column("corpus_raw"))
        .withColumn(
            "synopsis_normalized",
            F.when(F.col("synopsis").isNull(), F.lit("")).otherwise(_normalize_text_column("synopsis")),
        )
    )

    processed = build_text_pipeline("corpus_normalized", "tokens").fit(with_corpus).transform(with_corpus)
    processed = (
        build_text_pipeline("synopsis_normalized", "synopsis_tokens").fit(processed).transform(processed)
    )

    return (
        processed.withColumn("token_count", F.size("tokens"))
        .withColumn("corpus_clean", F.concat_ws(" ", F.col("tokens")))
        .withColumn("synopsis_token_count", F.size("synopsis_tokens"))
        .withColumn("synopsis_clean", F.concat_ws(" ", F.col("synopsis_tokens")))
        .drop("tokens_raw", "synopsis_tokens_raw", "corpus_normalized", "synopsis_normalized")
    )


def add_weighted_rating(df: DataFrame, min_votes: int = MIN_VOTES_RELIABLE) -> tuple[DataFrame, float]:
    """
    Hitung Bayesian weighted rating.

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    R = rating_score, v = rating_votes, m = ambang suara, C = rata-rata global
    pada populasi tepercaya. Efeknya: novel bernilai 5,0 dari 1 suara ditarik
    mendekati rata-rata, sedangkan novel 4,4 dari 280 suara nyaris tak bergeser.
    """
    reliable = df.filter((F.col("rating_score").isNotNull()) & (F.col("rating_votes") >= min_votes))
    global_mean = reliable.select(F.avg("rating_score")).first()[0]

    if global_mean is None:  # tidak mungkin pada dataset ini, tapi jangan diam-diam gagal
        raise ValueError("Tidak ada baris dengan rating tepercaya - weighted rating tak terdefinisi.")

    logger.info(
        "Rata-rata global (C) pada populasi >= %d suara: %.4f (n=%s)",
        min_votes,
        global_mean,
        f"{reliable.count():,}",
    )

    v = F.col("rating_votes").cast("double")
    m = F.lit(float(min_votes))
    weighted = (v / (v + m)) * F.col("rating_score") + (m / (v + m)) * F.lit(global_mean)

    out = df.withColumn(
        "weighted_rating",
        F.when(F.col("rating_score").isNotNull(), F.round(weighted, 4)).otherwise(F.lit(None)),
    ).withColumn(
        "is_rated_reliable",
        (F.col("rating_score").isNotNull() & (F.col("rating_votes") >= min_votes)).cast("boolean"),
    )
    return out, float(global_mean)


def add_derived_features(df: DataFrame) -> DataFrame:
    """Fitur turunan untuk analitik Fase 3 dan pemodelan Fase 4."""
    return (
        # Kecepatan rilis, diurai dari `release_frequency` ("Every 0.9 Day(s)").
        #
        # Kolom itu sebelumnya hanya lewat begitu saja sebagai teks, padahal
        # 100% baris dapat diurai dan hasilnya berkorelasi dengan rating sekuat
        # jumlah bab (r = -0,089 vs -0,088): novel yang bab barunya lama datang
        # dinilai sedikit lebih rendah. Disimpan sebagai hari per bab, bukan
        # bab per hari, supaya nilainya tidak meledak pada rilis harian.
        df.withColumn(
            "days_per_chapter",
            F.regexp_extract(F.col("release_frequency"), r"([\d.]+)\s*[Dd]ay", 1).cast("double"),
        )
        .withColumn(
            "release_pace",
            F.when(F.col("days_per_chapter").isNull(), F.lit(None))
            .when(F.col("days_per_chapter") <= 2, F.lit("1. Harian (<=2 hari)"))
            .when(F.col("days_per_chapter") <= 7, F.lit("2. Mingguan (3-7 hari)"))
            .when(F.col("days_per_chapter") <= 30, F.lit("3. Bulanan (8-30 hari)"))
            .when(F.col("days_per_chapter") <= 180, F.lit("4. Lambat (1-6 bulan)"))
            .otherwise(F.lit("5. Terhenti (>6 bulan)")),
        )
        .withColumn("decade", (F.floor(F.col("year") / 10) * 10).cast("int"))
        .withColumn(
            "length_bucket",
            F.when(F.col("chapter_count").isNull(), F.lit(None))
            .when(F.col("chapter_count") < 50, F.lit("1. Pendek (<50)"))
            .when(F.col("chapter_count") < 200, F.lit("2. Sedang (50-199)"))
            .when(F.col("chapter_count") < 500, F.lit("3. Panjang (200-499)"))
            .when(F.col("chapter_count") < 1000, F.lit("4. Sangat Panjang (500-999)"))
            .otherwise(F.lit("5. Epik (1000+)")),
        )
        .withColumn(
            "popularity_score",
            # Proksi popularitas: gabungan jumlah suara dan reading list,
            # di-log agar ekor ekstrem tidak mendominasi.
            F.round(
                F.log1p(F.coalesce(F.col("rating_votes"), F.lit(0)))
                + F.log1p(F.coalesce(F.col("reading_list_count"), F.lit(0))),
                4,
            ),
        )
    )


# --- Orkestrasi fase --------------------------------------------------------


def run() -> dict:
    """Jalankan seluruh Fase 2 dan kembalikan ringkasan kualitas data."""
    log_section(logger, "fase 2 - clean (pyspark processing)")
    settings = get_settings()
    report: list[dict] = []

    with spark_session("phase2-clean") as spark:
        raw = read_raw(spark).cache()
        n_raw = raw.count()
        logger.info("Baris mentah terbaca: %s", f"{n_raw:,}")
        report.append({"step": "1_raw_loaded", "rows": n_raw, "note": "CSV dari raw-data"})

        # --- Buang baris tanpa identitas (tidak bisa dipakai maupun ditelusuri)
        identified = raw.filter(F.col("series_id").isNotNull() & F.col("title").isNotNull())
        n_identified = identified.count()
        if n_identified < n_raw:
            logger.warning("%s baris dibuang: series_id/title kosong.", f"{n_raw - n_identified:,}")
        report.append({"step": "2_identified", "rows": n_identified, "note": "series_id & title ada"})

        # --- Normalisasi sentinel
        normalized = normalize_sentinels(identified)
        null_stats = normalized.select(
            F.sum(F.col("chapter_count").isNull().cast("int")).alias("chapter_count"),
            F.sum(F.col("year").isNull().cast("int")).alias("year"),
            F.sum(F.col("rating_score").isNull().cast("int")).alias("rating_score"),
            F.sum(F.col("tags").isNull().cast("int")).alias("tags"),
        ).first()
        logger.info(
            "Sentinel dinormalkan jadi NULL -> chapter_count=%s, year=%s, rating_score=%s, tags=%s",
            f"{null_stats['chapter_count']:,}",
            f"{null_stats['year']:,}",
            f"{null_stats['rating_score']:,}",
            f"{null_stats['tags']:,}",
        )
        report.append(
            {
                "step": "3_sentinels_to_null",
                "rows": n_identified,
                "note": (
                    f"null: chapter_count={null_stats['chapter_count']}, "
                    f"year={null_stats['year']}, rating_score={null_stats['rating_score']}, "
                    f"tags={null_stats['tags']}"
                ),
            }
        )

        # --- Deduplikasi
        deduped = deduplicate(normalized)
        n_dedup = deduped.count()
        logger.info("Setelah dedup by series_id: %s baris (-%s).", f"{n_dedup:,}", n_identified - n_dedup)
        report.append({"step": "4_deduplicated", "rows": n_dedup, "note": "kunci: series_id"})

        # --- Buang kolom miskin informasi
        pruned = deduped.drop(*DROP_COLS)
        logger.info("Kolom dibuang (>80%% kosong): %s", ", ".join(DROP_COLS))

        # --- Gabungkan sinopsis dari dataset pendamping
        with_synopsis = attach_synopsis(pruned, read_synopsis(spark))
        n_synopsis = with_synopsis.filter(F.col("has_synopsis")).count()
        coverage = 100 * n_synopsis / n_dedup
        logger.info(
            "Sinopsis tergabung: %s dari %s novel (%.2f%% cakupan, minimal 10 kata).",
            f"{n_synopsis:,}", f"{n_dedup:,}", coverage,
        )
        report.append(
            {
                "step": "5_synopsis_joined",
                "rows": n_synopsis,
                "note": f"cakupan {coverage:.2f}% (join by url, LEFT)",
            }
        )

        # --- Transformasi utama
        enriched = parse_multivalue(with_synopsis)
        enriched = apply_text_processing(enriched)
        enriched, global_mean = add_weighted_rating(enriched)
        enriched = add_derived_features(enriched)

        # --- Baris tanpa sinyal teks tidak berguna untuk Fase 4
        final = enriched.withColumn(
            "has_text_signal", (F.col("token_count") >= 3).cast("boolean")
        ).cache()
        n_final = final.count()
        n_text = final.filter(F.col("has_text_signal")).count()
        n_reliable = final.filter(F.col("is_rated_reliable")).count()
        logger.info(
            "Hasil akhir: %s baris | punya sinyal teks: %s | rating tepercaya: %s",
            f"{n_final:,}",
            f"{n_text:,}",
            f"{n_reliable:,}",
        )
        report.append({"step": "6_enriched", "rows": n_final, "note": f"C_global={global_mean:.4f}"})
        report.append({"step": "7_with_text_signal", "rows": n_text, "note": "token_count >= 3"})
        report.append({"step": "8_reliable_rating", "rows": n_reliable, "note": f">= {MIN_VOTES_RELIABLE} suara"})

        # --- Tulis hasil bersih ke lake
        clean_uri = settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
        (
            final.repartition(4)  # 4 file part: seimbang untuk data ~4.5k baris
            .write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(clean_uri)
        )
        logger.info("Data bersih ditulis ke %s", clean_uri)

        # --- Simpan laporan kualitas untuk ditampilkan di dashboard
        report_uri = settings.minio.s3a_uri(settings.minio.bucket_curated, "analytics/quality_report")
        spark.createDataFrame(report).coalesce(1).write.mode("overwrite").parquet(report_uri)
        logger.info("Laporan kualitas ditulis ke %s", report_uri)

        final.select(
            "title", "primary_genre", "chapter_count", "rating_score",
            "weighted_rating", "synopsis_word_count", "token_count",
        ).show(5, truncate=32)

    logger.info("FASE 2 SELESAI")
    return {
        "rows_out": n_final,
        "rows_with_text": n_text,
        "rows_with_synopsis": n_synopsis,
        "global_mean_rating": global_mean,
    }


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("FASE 2 GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
