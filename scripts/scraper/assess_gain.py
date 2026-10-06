"""
UKUR PEROLEHAN SCRAPE — apa yang benar-benar baru?
===================================================
Membandingkan hasil `scripts/scraper/novelupdates.py listing` dengan dataset
yang sudah ada, lalu menjawab tiga pertanyaan yang menentukan apakah scrape ini
sepadan:

  1. Berapa novel yang benar-benar **baru** (bukan duplikat 4.461 yang sudah ada)?
  2. Apakah **9 genre langka** yang terbuang dari model (`Shoujo` 4, `Yaoi` 4,
     `Smut` 2, `Gender Bender` 1, ...) akhirnya melewati ambang 50 novel?
  3. Apakah bias seleksi berkurang — yaitu apakah novel baru itu memang datang
     dari wilayah katalog yang dulu tidak terjangkau `ge=` 10 genre?

Aman dijalankan saat scrape masih berlangsung: berkas listing ditulis
inkremental, jadi skrip ini melaporkan potret sementara, bukan menunggu selesai.

Jalankan:
    docker compose exec -T pipeline python scripts/scraper/assess_gain.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/opt/project")

from src.common.logging_utils import get_logger, log_section  # noqa: E402

logger = get_logger("scraper.assess")

#: Ambang minimum novel per genre agar layak dilatih (dipakai Fase 4 Head A).
MIN_GENRE_SUPPORT = 50

DATASET_DIR = Path("/opt/project/Dataset")
LISTING = DATASET_DIR / "scraped" / "listing.csv"


def split_genres(series: pd.Series) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for value in series.dropna():
        counter.update(g.strip() for g in str(value).split(",") if g.strip())
    return counter


def normalise_url(series: pd.Series) -> pd.Series:
    """URL dinormalisasi sebelum dibandingkan — trailing slash & skema berbeda."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"^http://", "https://", regex=True)
        .str.rstrip("/")
        .str.lower()
    )


def main() -> int:
    log_section(logger, "perolehan scrape vs dataset lama")

    if not LISTING.is_file():
        raise SystemExit(f"Belum ada hasil listing di {LISTING}")

    new = pd.read_csv(LISTING, encoding="utf-8-sig")
    old = pd.read_csv(DATASET_DIR / "data_novel.csv")

    new["_key"] = normalise_url(new["url"])
    old["_key"] = normalise_url(old["url"])
    new = new.drop_duplicates("_key")

    old_keys = set(old["_key"])
    is_new = ~new["_key"].isin(old_keys)

    logger.info("Listing terkumpul  : %s baris unik", f"{len(new):,}")
    logger.info("Katalog lama       : %s baris", f"{len(old):,}")
    logger.info("Beririsan          : %s", f"{(~is_new).sum():,}")
    logger.info("BENAR-BENAR BARU   : %s", f"{is_new.sum():,}")
    logger.info("")

    if "source" in new.columns:
        logger.info("Sumbangan per sumber listing:")
        per_source = (
            new.assign(baru=is_new).groupby("source")["baru"].agg(["size", "sum"])
        )
        for source, row in per_source.iterrows():
            logger.info("  %-14s %6s baris | %6s baru (%.0f%%)",
                        source, f"{int(row['size']):,}", f"{int(row['sum']):,}",
                        100 * row["sum"] / max(row["size"], 1))
        logger.info("")

    # Genre: apakah yang langka akhirnya cukup?
    old_genres = split_genres(old["genres"])
    combined = collections.Counter(old_genres)
    combined.update(split_genres(new.loc[is_new, "genres"]))

    under_before = {g: c for g, c in old_genres.items() if c < MIN_GENRE_SUPPORT}
    rescued = {g: (old_genres.get(g, 0), combined[g])
               for g in under_before if combined[g] >= MIN_GENRE_SUPPORT}
    still_under = {g: (old_genres.get(g, 0), combined[g])
                   for g in under_before if combined[g] < MIN_GENRE_SUPPORT}

    logger.info("Genre unik: %d -> %d", len(old_genres), len(combined))
    logger.info("Di bawah ambang %d: %d -> %d",
                MIN_GENRE_SUPPORT, len(under_before), len(still_under))
    logger.info("")
    if rescued:
        logger.info("TERTOLONG (kini layak dilatih):")
        for genre, (before, after) in sorted(rescued.items(), key=lambda x: -x[1][1]):
            logger.info("  %-18s %4d -> %5d", genre, before, after)
    if still_under:
        logger.info("Masih di bawah ambang:")
        for genre, (before, after) in sorted(still_under.items(), key=lambda x: -x[1][1]):
            logger.info("  %-18s %4d -> %5d", genre, before, after)
    logger.info("")

    # Genre yang sama sekali tidak ada di katalog lama
    brand_new = {g: c for g, c in combined.items() if g not in old_genres}
    if brand_new:
        logger.info("Genre yang SAMA SEKALI tidak ada di katalog lama:")
        for genre, count in sorted(brand_new.items(), key=lambda x: -x[1]):
            logger.info("  %-18s %5d", genre, count)
        logger.info("")

    logger.info("Total label genre  : %s -> %s",
                f"{sum(old_genres.values()):,}", f"{sum(combined.values()):,}")
    logger.info("Novel model-ready  : %s -> %s (perkiraan)",
                f"{len(old):,}", f"{len(old) + int(is_new.sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
