"""
Orchestrator pipeline
=====================
Menjalankan Fase 1 → 4 secara berurutan dalam satu proses.

Fase 5 (Streamlit) tidak ikut di sini karena ia adalah service yang menyala
terus-menerus, bukan job batch — ia dijalankan oleh docker compose dan
membaca artefak yang dihasilkan fase-fase di bawah ini.

Perilaku saat gagal: pipeline BERHENTI di fase pertama yang gagal. Melanjutkan
ke fase berikutnya di atas data yang tidak lengkap akan menghasilkan analitik
dan model yang "sukses" tetapi salah — kegagalan diam-diam yang jauh lebih
berbahaya daripada berhenti lebih awal.

Jalankan:
    docker compose exec pipeline python scripts/run_pipeline.py
    docker compose exec pipeline python scripts/run_pipeline.py --from 3
    docker compose exec pipeline python scripts/run_pipeline.py --only 2
"""

from __future__ import annotations

import argparse
import sys
import time

from src.common.logging_utils import get_logger, log_section

logger = get_logger("orchestrator")


def _phase1(force: bool):
    from src.phases import phase1_collect

    return phase1_collect.run(force=force)


def _phase2(_force: bool):
    from src.phases import phase2_clean

    return phase2_clean.run()


def _phase3(_force: bool):
    from src.phases import phase3_explore

    return phase3_explore.run()


def _phase4(_force: bool):
    from src.phases import phase4_model

    return phase4_model.run()


PHASES = {
    1: ("Collect (Ingestion ke MinIO)", _phase1),
    2: ("Clean (PySpark preprocessing)", _phase2),
    3: ("Explore (Spark SQL analytics)", _phase3),
    4: ("Model (Arsitektur AI multi-head)", _phase4),
}


def run_pipeline(start: int = 1, end: int = 4, force: bool = False) -> int:
    log_section(logger, "menjalankan pipeline big data novel")
    durations: list[tuple[int, str, float]] = []

    for number in range(start, end + 1):
        label, func = PHASES[number]
        logger.info("")
        logger.info("### FASE %d — %s", number, label)
        started = time.perf_counter()
        try:
            func(force)
        except Exception as exc:  # noqa: BLE001 - berhenti dengan pesan yang jelas
            elapsed = time.perf_counter() - started
            logger.error("FASE %d GAGAL setelah %.1f detik: %s", number, elapsed, exc, exc_info=True)
            logger.error("Pipeline dihentikan. Fase berikutnya TIDAK dijalankan.")
            return 1
        durations.append((number, label, time.perf_counter() - started))

    log_section(logger, "ringkasan eksekusi")
    for number, label, elapsed in durations:
        logger.info("Fase %d | %-38s | %6.1f detik", number, label, elapsed)
    logger.info("Total: %.1f detik", sum(d for _, _, d in durations))
    logger.info("Dashboard: http://localhost:8501")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator pipeline Fase 1-4")
    parser.add_argument("--from", dest="start", type=int, default=1, choices=sorted(PHASES),
                        help="Fase awal (default 1)")
    parser.add_argument("--to", dest="end", type=int, default=4, choices=sorted(PHASES),
                        help="Fase akhir (default 4)")
    parser.add_argument("--only", type=int, choices=sorted(PHASES),
                        help="Jalankan satu fase saja")
    parser.add_argument("--force", action="store_true",
                        help="Paksa ingestion ulang di Fase 1 walau checksum sama")
    args = parser.parse_args()

    start, end = (args.only, args.only) if args.only else (args.start, args.end)
    if start > end:
        parser.error("--from tidak boleh lebih besar dari --to")
    return run_pipeline(start=start, end=end, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
