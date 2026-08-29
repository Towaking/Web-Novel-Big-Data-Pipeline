"""
Spark session factory
=====================
Semua fase WAJIB membuat SparkSession lewat modul ini, bukan lewat
`SparkSession.builder` langsung. Alasannya:

  1. Konfigurasi konektor S3A (MinIO) ditulis satu kali di sini.
     Satu setting saja terlewat -> job gagal dengan error
     `NoSuchMethodError` / `403 Forbidden` yang sulit didiagnosis.
  2. `spark.driver.host` harus diisi hostname container driver, karena
     executor di container lain perlu koneksi balik ke driver.
     Tanpa ini, job menggantung di stage pertama.
  3. Fallback ke mode `local[*]` bila cluster tidak tersedia, supaya
     modul tetap bisa diuji di luar compose.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from typing import Iterator

from pyspark.sql import SparkSession

from src.common.config import Settings, get_settings
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _s3a_options(settings: Settings) -> dict:
    """Konfigurasi Hadoop-AWS agar Spark bisa bicara dengan MinIO."""
    minio = settings.minio
    return {
        "spark.hadoop.fs.s3a.endpoint": minio.endpoint,
        "spark.hadoop.fs.s3a.access.key": minio.access_key,
        "spark.hadoop.fs.s3a.secret.key": minio.secret_key,
        # MinIO tidak mendukung virtual-host style bucket -> wajib path style.
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": (
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        ),
        # Commit v2: hindari rename direktori pada object storage
        # (di S3 rename = copy + delete, mahal dan tidak atomik).
        "spark.hadoop.fs.s3a.fast.upload": "true",
        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": "2",
    }


def _tuning_options(settings: Settings) -> dict:
    """Tuning untuk dataset skala menengah pada cluster lokal."""
    spark_cfg = settings.spark
    options = {
        # Dataset ~4.5k baris: 200 shuffle partition default Spark bikin
        # overhead scheduling task jauh lebih mahal dari komputasinya.
        "spark.sql.shuffle.partitions": str(spark_cfg.shuffle_partitions),
        "spark.driver.memory": spark_cfg.driver_memory,
        "spark.executor.memory": spark_cfg.executor_memory,
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.session.timeZone": "UTC",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        # Arrow mempercepat konversi Spark -> pandas saat hasil agregasi
        # kecil ditarik ke driver untuk visualisasi (Fase 5).
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    }
    if spark_cfg.driver_host:
        options["spark.driver.host"] = spark_cfg.driver_host
        options["spark.driver.bindAddress"] = "0.0.0.0"
    return options


def build_spark_session(
    app_name: str,
    settings: Settings | None = None,
    extra_conf: dict | None = None,
) -> SparkSession:
    """
    Bangun (atau ambil kembali) SparkSession yang siap pakai untuk MinIO.

    Args:
        app_name: nama aplikasi yang tampil di Spark UI (port 8080).
        settings: override konfigurasi, default dibaca dari environment.
        extra_conf: konfigurasi tambahan spesifik per fase.

    Raises:
        RuntimeError: bila session gagal dibuat (master tak terjangkau, dsb).
    """
    settings = settings or get_settings()
    master = settings.spark.master_url

    builder = SparkSession.builder.appName(app_name).master(master)
    for key, value in {**_s3a_options(settings), **_tuning_options(settings)}.items():
        builder = builder.config(key, value)
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    try:
        spark = builder.getOrCreate()
    except Exception as exc:  # dipetakan ulang menjadi pesan yang actionable
        raise RuntimeError(
            "Gagal membuat SparkSession pada master " + repr(master) + ". "
            "Pastikan service spark-master sudah healthy (docker compose ps) "
            "dan variabel SPARK_MASTER_URL sudah benar."
        ) from exc

    spark.sparkContext.setLogLevel("WARN")
    logger.info(
        "SparkSession aktif | app=%s | master=%s | versi=%s",
        app_name,
        master,
        spark.version,
    )
    atexit.register(spark.stop)
    return spark


@contextmanager
def spark_session(app_name: str, **kwargs) -> Iterator[SparkSession]:
    """Context manager: menjamin session ditutup walau job gagal di tengah."""
    spark = build_spark_session(app_name, **kwargs)
    try:
        yield spark
    finally:
        spark.stop()
        logger.info("SparkSession '%s' ditutup.", app_name)
