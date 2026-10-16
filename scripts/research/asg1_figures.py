"""
Grafik Bab 1 sampai 4.2 laporan (bagian yang diwarisi dari Assignment I), dari snapshot data saat ini.

Sumber:
  Dataset/variety/                ekspor zona clean (EDA)
  curated-data/analytics, model/  quality_report, star schema, property graph

Dijalankan di container pipeline karena membaca lake:
    docker compose exec pipeline python scripts/research/asg1_figures.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from asg2_figures import AMBER, GREY, INK, MUTED, PURPLE, TEAL, fmt, save  # noqa: E402

VARIETY = ROOT / "Dataset/variety"
SNAPSHOT = ROOT / "Doc/generated/lake_snapshot.json"


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def dotted(axis) -> None:
    """Pemisah ribuan titik di sumbu, sama dengan angka pada label batang."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: thousands(v)))


def load_eda():
    m = pd.read_csv(VARIETY / "structured/novel_metrics.csv")
    labels = pd.read_json(VARIETY / "semistructured/novel_labels.jsonl", lines=True)
    words = []
    for path in sorted(glob.glob(str(VARIETY / "unstructured/synopsis/*.txt"))):
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3:
                words.append(len(parts[2].split()))
    return m, labels, np.array(words)


def lake_snapshot() -> dict:
    """Angka dari lake untuk gambar 3.1, 4.7, dan 4.9, disimpan supaya teks laporan bisa dicek ulang."""
    from src.common.config import get_settings
    from src.common.storage import read_parquet_dir

    s = get_settings()
    curated = lambda p: read_parquet_dir(s.minio.bucket_curated, p)  # noqa: E731
    quality = curated("analytics/quality_report").sort_values("step")
    tables = ["fact_novel", "dim_novel", "dim_time", "dim_publisher", "dim_genre", "dim_tag", "dim_author",
              "bridge_novel_genre", "bridge_novel_tag", "bridge_novel_author", "graph_nodes", "graph_edges"]
    counts = {t: int(len(curated(f"model/{t}"))) for t in tables}
    summary = curated("analytics/graph_summary").set_index("label")
    snap = {
        "quality_report": quality[["step", "rows", "note"]].to_dict("records"),
        "row_counts": counts,
        "projections": {
            label: {k: (None if pd.isna(v) else (float(v) if isinstance(v, float) else v))
                    for k, v in summary.loc[label].items()}
            for label in summary.index
        },
    }
    SNAPSHOT.write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("ditulis", SNAPSHOT.relative_to(ROOT))
    return snap


def titles_per_year(full: pd.DataFrame) -> None:
    years = full["year"].dropna().astype(int)
    counts = years[(years >= 2010) & (years <= 2025)].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = [TEAL if y == 2025 else PURPLE for y in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.7)
    for y in (2010, 2015, 2020, 2025):
        ax.text(y, counts[y] + 60, thousands(counts[y]), ha="center", fontsize=8, color=INK)
    ax.set_xticks(counts.index)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel("Judul")
    dotted(ax.yaxis)
    ax.grid(axis="x", visible=False)
    save(fig, "a1_1_judul_per_tahun.png")


def votes_buckets(full: pd.DataFrame) -> None:
    v = full["rating_votes"].fillna(0)
    edges = [(0, 0, "0"), (1, 9, "1-9"), (10, 49, "10-49"), (50, 199, "50-199"), (200, 999, "200-999"),
             (1000, np.inf, "1.000+")]
    counts = [int(((v >= lo) & (v <= hi)).sum()) for lo, hi, _ in edges]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(edges))
    ax.bar(x, counts, color=[AMBER, AMBER] + [PURPLE] * 4, width=0.66)
    for i, c in enumerate(counts):
        ax.text(i, c + 150, thousands(c), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [e[2] for e in edges])
    ax.set_xlabel("Jumlah suara rating per judul")
    ax.set_ylabel("Judul")
    dotted(ax.yaxis)
    ax.grid(axis="x", visible=False)
    save(fig, "a1_2_sebaran_suara.png")


QUALITY_LABELS = {
    "1_raw_loaded": "Dimuat dari raw", "2_identified": "URL & judul ada", "3_sentinels_to_null": "Sentinel jadi NULL",
    "4_deduplicated": "Deduplikasi", "5_with_synopsis": "Bersinopsis (>= 10 kata)", "6_enriched": "Fitur turunan",
    "7_with_text_signal": "Punya sinyal teks", "8_reliable_rating": "Rating tepercaya (>= 10 suara)",
}
SUBSET_STEPS = {"5_with_synopsis", "7_with_text_signal", "8_reliable_rating"}


def quality_steps(snap: dict) -> None:
    steps = pd.DataFrame(snap["quality_report"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(steps))[::-1]
    colors = [GREY if s in SUBSET_STEPS else PURPLE for s in steps["step"]]
    ax.barh(y, steps["rows"], color=colors, height=0.62)
    for yy, r in zip(y, steps["rows"]):
        ax.text(r + 400, yy, thousands(r), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [QUALITY_LABELS.get(s, s) for s in steps["step"]])
    ax.set_xlim(0, steps["rows"].max() * 1.14)
    ax.set_xlabel("Baris setelah langkah (abu-abu: subset untuk analisis tertentu)")
    dotted(ax.xaxis)
    ax.grid(axis="y", visible=False)
    save(fig, "a1_3_quality_report.png")


def missing_share(full: pd.DataFrame) -> None:
    cols = ["year", "chapter_count", "original_publisher", "weighted_rating", "rating_score", "type_origin", "type",
            "language", "reading_list_count", "days_per_chapter", "rating_votes"]
    share = (full[cols].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    colors = [AMBER if v > 10 else PURPLE for v in share.values]
    ax.barh(share.index, share.values, color=colors, height=0.62)
    for i, v in enumerate(share.values):
        ax.text(v + 0.25, i, fmt(v, 2) + "%", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, share.max() * 1.18)
    ax.set_xlabel(f"Persentase nilai kosong (tier full, {thousands(len(full))} judul)")
    ax.grid(axis="y", visible=False)
    save(fig, "a1_4_nilai_kosong.png")


def distributions(full: pd.DataFrame, words: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    ch = full["chapter_count"].dropna()
    ax = axes[0, 0]
    ax.hist(np.log10(ch), bins=45, color=PURPLE)
    med = ch.median()
    ax.axvline(np.log10(med), color=AMBER, lw=1.5)
    ax.text(np.log10(med) + 0.05, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", color=AMBER, fontsize=8)
    ax.set_xticks(range(0, 5), ["1", "10", "100", "1.000", "10.000"])
    ax.set_title("Jumlah bab (skala log10)", fontsize=10, weight="bold")
    ax = axes[0, 1]
    bins = np.linspace(1, 5, 41)
    ax.hist(full["rating_score"].dropna(), bins=bins, color=GREY, alpha=0.9, label="rating_score mentah")
    ax.hist(full["weighted_rating"].dropna(), bins=bins, color=PURPLE, alpha=0.85, label="weighted_rating")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Rating mentah vs weighted rating", fontsize=10, weight="bold")
    ax = axes[1, 0]
    votes = full["rating_votes"].fillna(0)
    ax.hist(np.log10(votes + 1), bins=45, color=TEAL)
    ax.axvline(np.log10(11), color=AMBER, lw=1.5)
    ax.text(np.log10(11) + 0.05, ax.get_ylim()[1] * 0.92, "ambang 10 suara", color=AMBER, fontsize=8)
    ax.set_xticks(np.log10([1, 11, 101, 1001]), ["0", "10", "100", "1.000"])
    ax.set_title("Jumlah suara (skala log10, +1)", fontsize=10, weight="bold")
    ax = axes[1, 1]
    ax.hist(np.clip(words, 0, 600), bins=50, color=PURPLE)
    wmed = float(np.median(words))
    ax.axvline(wmed, color=AMBER, lw=1.5)
    ax.text(wmed + 8, ax.get_ylim()[1] * 0.92, f"median {wmed:.0f}", color=AMBER, fontsize=8)
    ax.set_title("Panjang sinopsis (kata, dipotong 600)", fontsize=10, weight="bold")
    for a in axes.flat:
        a.set_ylabel("Judul", fontsize=8)
        dotted(a.yaxis)
    fig.tight_layout()
    save(fig, "a1_5_distribusi.png")


def top_genres(full: pd.DataFrame, labels: pd.DataFrame) -> None:
    rel = full[full["is_rated_reliable"] == 1][["series_id", "weighted_rating"]]
    pairs = labels[["series_id", "genres"]].explode("genres").rename(columns={"genres": "genre"}).dropna()
    count = pairs["genre"].value_counts().head(15)
    wr = pairs.merge(rel, on="series_id").groupby("genre")["weighted_rating"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(count))[::-1]
    ax.barh(y, count.values, color=PURPLE, height=0.66)
    for yy, (g, c) in zip(y, count.items()):
        ax.text(c + 200, yy, f"{thousands(c)}  |  WR {fmt(wr[g], 2)}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, count.index)
    ax.set_xlim(0, count.max() * 1.32)
    ax.set_xlabel("Judul yang menyandang genre (satu judul bisa dihitung di beberapa genre)")
    dotted(ax.xaxis)
    ax.grid(axis="y", visible=False)
    save(fig, "a1_6_genre_teratas.png")


def length_and_pace(full: pd.DataFrame) -> None:
    rel = full[full["is_rated_reliable"] == 1]
    length = pd.cut(rel["chapter_count"], [0, 49, 199, 499, 999, np.inf],
                    labels=["<50", "50-199", "200-499", "500-999", "1000+"])
    pace = pd.cut(rel["days_per_chapter"], [0, 2, 7, 30, 180, np.inf],
                  labels=["Harian", "Mingguan", "Bulanan", "Lambat", "Terhenti"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, groups, title in ((axes[0], length, "Kelompok panjang (bab)"), (axes[1], pace, "Kecepatan rilis")):
        stats_ = rel.groupby(groups, observed=True)["weighted_rating"].agg(["mean", "size"])
        x = np.arange(len(stats_))
        ax.bar(x, stats_["mean"], color=PURPLE, width=0.62)
        for i, v in enumerate(stats_["mean"]):
            ax.text(i, v + 0.003, fmt(v, 3), ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [f"{k}\nn={thousands(n)}" for k, n in zip(stats_.index, stats_["size"])], fontsize=7.5)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(3.50, 3.78)
    axes[0].set_ylabel("Rata-rata weighted rating")
    save(fig, "a1_7_panjang_kecepatan.png")


def spearman_matrix(full: pd.DataFrame) -> None:
    rel = full[full["is_rated_reliable"] == 1]
    frame = pd.DataFrame({
        "weighted_rating": rel["weighted_rating"], "log suara": np.log1p(rel["rating_votes"]),
        "log reading list": np.log1p(rel["reading_list_count"]), "log bab": np.log1p(rel["chapter_count"]),
        "log hari/bab": np.log1p(rel["days_per_chapter"]), "tahun": rel["year"],
    })
    corr = frame.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, fmt(v, 2), ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.7 else INK)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    save(fig, "a1_8_korelasi.png")


def trends(full: pd.DataFrame) -> None:
    span = full[(full["year"] >= 2010) & (full["year"] <= 2025)]
    wr = span.groupby("year")["weighted_rating"].mean()
    med = span.groupby("year")["chapter_count"].median()
    done = span.groupby("year")["is_completed"].mean() * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    a1.plot(wr.index, wr.values, color=PURPLE, marker="o", ms=4)
    a1.set_title("Rata-rata weighted rating per tahun", fontsize=10, weight="bold")
    l1, = a2.plot(med.index, med.values, color=TEAL, marker="o", ms=4, label="Median jumlah bab")
    a2.set_title("Median jumlah bab dan % tamat per tahun", fontsize=10, weight="bold")
    a3 = a2.twinx()
    l2, = a3.plot(done.index, done.values, color=AMBER, ls="--", label="% tamat (sumbu kanan)")
    a2.legend(handles=[l1, l2], frameon=False, fontsize=7.5, loc="lower center")
    a3.set_ylabel("% tamat", color=AMBER, fontsize=8)
    a3.tick_params(axis="y", colors=AMBER)
    a3.spines["right"].set_visible(True)
    a3.grid(False)
    for a in (a1, a2):
        a.set_xticks(range(2010, 2026, 2))
        a.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    save(fig, "a1_9_tren.png")


def _right_edge(ax, artist) -> float:
    """Batas kanan teks dalam koordinat data, diukur dari renderer."""
    bbox = artist.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    return ax.transData.inverted().transform((bbox.x1, bbox.y0))[0]


def _box(ax, x, y, w, rows, title, count, accent=False, header=None):
    """Kotak tabel (satuan inci): judul tebal dan jumlah baris di kepala, satu baris teks per elemen."""
    head, line, pad = 0.30, 0.19, 0.10
    h = head + pad + line * len(rows)
    edge = PURPLE if accent else "#C9CCD8"
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec=edge, lw=1.8 if accent else 1.0))
    ax.add_patch(FancyBboxPatch((x, y - head), w, head, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=header or ("#EEEBFD" if accent else "#F2F3F7"), ec="none"))
    label = ax.text(x + 0.08, y - head / 2, title, fontsize=8.8, weight="bold", va="center",
                    color=PURPLE if accent else ("#0E6B52" if header else INK))
    tail = ax.text(_right_edge(ax, label) + 0.06, y - head / 2, f"({count})", fontsize=7.8, va="center", color=MUTED)
    texts = [tail]
    for i, (text, key) in enumerate(rows):
        texts.append(ax.text(x + 0.08, y - head - pad / 2 - line * (i + 0.5), text, fontsize=8, va="center",
                             color=PURPLE if key else INK))
    widest = max(_right_edge(ax, a) for a in texts)
    assert widest <= x + w - 0.03, f"teks melebihi kotak {title}: {widest:.2f} > {x + w:.2f}"
    return h


def star_schema(snap: dict) -> None:
    """Satu baris dimensi tunggal di atas fakta, bridge dan dimensi multinilai di bawahnya."""
    c = snap["row_counts"]
    bottom, top = 0.36, 5.55
    fig, ax = plt.subplots(figsize=(6.8, top - bottom))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 6.8)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    fig.canvas.draw()
    xs, w = (0.03, 2.31, 4.59), 2.18
    centers = [x + w / 2 for x in xs]
    singles = [("dim_novel", [("series_id PK", True), ("title, type, type_origin,", False),
                              ("language, status,", False), ("length_bucket, release_pace", False)]),
               ("dim_time", [("year PK, decade", True)]),
               ("dim_publisher", [("publisher_key PK, publisher", True)])]
    fact_top = 3.95
    for (name, rows), x, cx in zip(singles, xs, centers):
        h = _box(ax, x, top - 0.03, w, rows, name, thousands(c[name]))
        ax.plot([cx, cx], [top - 0.03 - h, fact_top], color=MUTED, lw=1.2)
    fact_rows = [("series_id, publisher_key, year (FK)", True),
                 ("Aditif: chapter_count, rating_votes, reading_list_count", False),
                 ("Non-aditif: rating_score, weighted_rating, days_per_chapter,", False),
                 ("popularity_score", False),
                 ("Teks: synopsis_word_count, token_count", False),
                 ("Penanda: is_rated_reliable, has_synopsis", False)]
    fact_h = _box(ax, 0.6, fact_top, 5.6, fact_rows, "fact_novel",
                  f"{thousands(c['fact_novel'])}; grain: 1 baris per judul", accent=True)
    bridge_top = fact_top - fact_h - 0.32
    multi = [("bridge_novel_genre", "series_id, genre_key", "dim_genre", [("genre_key PK, genre", True)]),
             ("bridge_novel_tag", "series_id, tag_key", "dim_tag", [("tag_key PK, tag", True)]),
             ("bridge_novel_author", "series_id, author_key", "dim_author",
              [("author_key PK, author,", True), ("resolution, aliases", True)])]
    for (bridge, bridge_row, dim, dim_rows), x, cx in zip(multi, xs, centers):
        ax.plot([cx, cx], [fact_top - fact_h, bridge_top], color=MUTED, lw=1.2)
        hb = _box(ax, x, bridge_top, w, [(bridge_row, False)], bridge, thousands(c[bridge]), header="#E6F5EF")
        dim_top = bridge_top - hb - 0.24
        ax.plot([cx, cx], [bridge_top - hb, dim_top], color=MUTED, lw=1.2)
        _box(ax, x, dim_top, w, dim_rows, dim, thousands(c[dim]))
    save(fig, "a1_10_star_schema.png")


def property_graph(snap: dict) -> None:
    c = snap["row_counts"]
    proj = snap["projections"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    fig.canvas.draw()
    nodes = {
        "Genre": (1.35, 3.55, TEAL, "#E6F5EF", f"{c['dim_genre']} simpul, \"G:genre_key\""),
        "Tag": (6.05, 3.55, TEAL, "#E6F5EF", f"{thousands(c['dim_tag'])} simpul, \"T:tag_key\""),
        "Novel": (3.7, 2.12, PURPLE, "#EEEBFD", f"{thousands(c['dim_novel'])} simpul, id \"N:series_id\""),
        "Author": (3.7, 0.5, AMBER, "#FDF2E2", f"{thousands(c['dim_author'])} simpul, \"A:author_key\""),
    }
    for name, (x, y, edge, fill, sub) in nodes.items():
        ax.add_patch(Ellipse((x, y), 2.6, 0.74, fc=fill, ec=edge, lw=1.8))
        ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=11, weight="bold", color=edge)
        ax.text(x, y - 0.14, sub, ha="center", va="center", fontsize=8, color=MUTED)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.3, mutation_scale=11)
    ax.annotate("", xy=(2.05, 3.19), xytext=(3.0, 2.46), arrowprops=arrow)
    ax.annotate("", xy=(5.35, 3.19), xytext=(4.4, 2.46), arrowprops=arrow)
    ax.annotate("", xy=(3.7, 0.89), xytext=(3.7, 1.74), arrowprops=arrow)
    ax.text(2.45, 2.72, f"HAS_GENRE ({thousands(c['bridge_novel_genre'])} sisi)", fontsize=8.5, color=INK, ha="right")
    ax.text(4.95, 2.72, f"HAS_TAG ({thousands(c['bridge_novel_tag'])} sisi)", fontsize=8.5, color=INK)
    ax.text(3.6, 1.25, f"WRITTEN_BY ({thousands(c['bridge_novel_author'])} sisi)", fontsize=8.5, color=INK, ha="right")
    genre, author = proj["genre_cooccurrence"], proj["author_collaboration"]
    for x, title, kind, size in (
            (0.05, "Proyeksi 1: Genre, Novel, Genre", "graf ko-okurensi terbobot",
             f"{int(genre['n_nodes'])} simpul, {thousands(genre['n_edges'])} sisi"),
            (5.1, "Proyeksi 2: Author, Novel, Author", "graf kolaborasi penulis",
             f"{thousands(author['n_nodes'])} simpul, {thousands(author['n_edges'])} sisi")):
        head = ax.text(x, 0.82, title, fontsize=8.5, color=INK)
        assert _right_edge(ax, head) < 7.38, "judul proyeksi terpotong"
        ax.text(x, 0.6, kind, fontsize=8, color=MUTED)
        ax.text(x, 0.38, size, fontsize=8, color=MUTED)
    save(fig, "a1_11_property_graph.png")


def main() -> None:
    snap = lake_snapshot()
    m, labels, words = load_eda()
    full = m[m["metadata_tier"] == "full"]
    titles_per_year(full)
    votes_buckets(full)
    quality_steps(snap)
    missing_share(full)
    distributions(full, words)
    top_genres(full, labels)
    length_and_pace(full)
    spearman_matrix(full)
    trends(full)
    star_schema(snap)
    property_graph(snap)


if __name__ == "__main__":
    main()
