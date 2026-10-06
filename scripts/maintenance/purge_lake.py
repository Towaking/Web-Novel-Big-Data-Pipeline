"""
Bersihkan data lake dari versi dan objek yang tidak lagi dipakai
================================================================
Versioning menyimpan setiap tulisan ulang pipeline (termasuk berkas
`_temporary` Spark yang sudah dihapus), sehingga versi lama tumbuh jauh
melampaui data aktif. Aturan lifecycle di minio-init membuangnya secara
berkala; skrip ini melakukannya sekarang juga, dan menghapus:

  * versi lama (noncurrent) dan delete marker di ketiga bucket;
  * objek raw-data/novel/ yang bukan sumber terdaftar atau manifestnya;
  * artefak riset `models/_*` di curated-data (tidak dibaca dashboard).

Default-nya hanya menghitung (dry run). `--apply` untuk benar-benar menghapus.

    docker compose exec pipeline python scripts/maintenance/purge_lake.py [--apply]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import PurePosixPath

from src.common.config import get_settings
from src.common.logging_utils import get_logger
from src.common.storage import get_s3_client

logger = get_logger("maintenance.purge")


def registered_raw_keys(settings) -> set[str]:
    keys = set()
    for _, key in settings.raw_sources:
        path = PurePosixPath(key)
        keys |= {key, f"{path.parent.as_posix()}/{path.stem}_manifest.json"}
    return keys


def plan(client, bucket: str, keep_current) -> tuple[list[dict], int]:
    """(versions to delete, bytes) for one bucket."""
    doomed, size = [], 0
    for page in client.get_paginator("list_object_versions").paginate(Bucket=bucket):
        for v in page.get("Versions", []):
            if v["IsLatest"] and keep_current(v["Key"]):
                continue
            doomed.append({"Key": v["Key"], "VersionId": v["VersionId"]})
            size += v["Size"]
        for m in page.get("DeleteMarkers", []):
            doomed.append({"Key": m["Key"], "VersionId": m["VersionId"]})
    return doomed, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--apply", action="store_true", help="delete instead of only counting")
    args = parser.parse_args()

    settings = get_settings()
    minio = settings.minio
    client = get_s3_client(settings)
    raw_keep = registered_raw_keys(settings)
    rules = {
        minio.bucket_raw: lambda k: not k.startswith("novel/") or k in raw_keep,
        minio.bucket_clean: lambda k: True,
        minio.bucket_curated: lambda k: not k.startswith("models/_"),
    }
    total = 0
    for bucket, keep_current in rules.items():
        doomed, size = plan(client, bucket, keep_current)
        total += size
        logger.info("%-13s %5d versions/markers, %8.1f MB", bucket, len(doomed), size / 1e6)
        if args.apply:
            for i in range(0, len(doomed), 1000):
                result = client.delete_objects(Bucket=bucket, Delete={"Objects": doomed[i:i + 1000], "Quiet": True})
                if result.get("Errors"):
                    logger.error("%s: %d deletions failed, first: %s", bucket, len(result["Errors"]), result["Errors"][0])
                    return 1
    logger.info("%s %.1f MB", "Freed" if args.apply else "Would free", total / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
