"""
Smoke test dashboard (Fase 5)
=============================
Menjalankan `app/streamlit_app.py` secara headless memakai `AppTest` bawaan
Streamlit, lalu memeriksa setiap halaman tidak melempar exception.

Kenapa perlu: kesalahan pada dashboard (nama kolom salah, tabel belum ada,
argumen widget tidak valid) hanya muncul saat halaman dirender. Tanpa tes ini,
kegagalan baru ketahuan ketika seseorang membukanya di browser.

Catatan: tes ini memvalidasi LOGIKA halaman, bukan tampilannya.

Jalankan:
    docker compose exec streamlit python scripts/verify_app.py
"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

from src.common.logging_utils import get_logger, log_section

logger = get_logger("verify_app")

APP_PATH = "/opt/project/app/streamlit_app.py"
PAGES = ["Overview", "Catalogue Analytics", "AI Studio", "Methodology"]
TIMEOUT_SECONDS = 180


def check_page(page: str) -> bool:
    """Render satu halaman dan laporkan exception yang muncul."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()

    if app.exception:
        logger.error("[%s] gagal saat inisialisasi: %s", page, app.exception[0].value)
        return False

    # Halaman dipilih lewat radio di sidebar.
    app.sidebar.radio[0].set_value(page).run()

    if app.exception:
        for exc in app.exception:
            logger.error("[%s] exception: %s", page, exc.value)
        return False

    logger.info(
        "[%s] OK | %d judul, %d chart, %d tabel, %d metrik",
        page,
        len(app.title) + len(app.subheader),
        len(app.get("plotly_chart")),
        len(app.dataframe),
        len(app.metric),
    )
    return True


def main() -> int:
    log_section(logger, "smoke test dashboard streamlit")
    results = [(page, check_page(page)) for page in PAGES]

    log_section(logger, "ringkasan")
    for page, ok in results:
        logger.info("%-26s : %s", page, "LULUS" if ok else "GAGAL")

    if all(ok for _, ok in results):
        logger.info("Dashboard siap dibuka di http://localhost:8501")
        return 0
    logger.error("Ada halaman yang gagal dirender.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
