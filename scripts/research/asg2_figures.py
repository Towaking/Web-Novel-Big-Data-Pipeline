"""
Grafik laporan Assignment II (Bab 5 dan Bab 6).

Semua angka dibaca dari keluaran pipeline, bukan diketik ulang:
  Model/artifacts/metrics.json            Fase 4 (scikit-learn)
  Doc/generated/mllib_comparison.json     Fase 5 (Spark MLlib vs scikit-learn)
  Doc/generated/similarity_graph.json     graf kemiripan novel
  Doc/generated/similarity_graph_edges.parquet, similarity_graph_nodes.parquet

    python scripts/research/asg2_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Doc/Laporan/figures"

# Palet laporan Assignment I: ungu utama, teal pembanding, abu-abu untuk konteks.
PURPLE, TEAL, AMBER, GREY, INK, MUTED = "#5B3FD9", "#13A07A", "#C77A12", "#B5B9C8", "#1F2333", "#5C6178"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#C9CCD8", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ECEDF2", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("ditulis", name)


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 45)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=1.4, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=9.5, weight="bold", color=edge)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.8, color=INK, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, shrinkA=0, shrinkB=0))

    box(0, 32, 28, 12, "Zona clean", "novels_clean (MinIO, Parquet)\n39.660 judul bersinyal teks", PURPLE, "#EEEBFC")
    box(0, 16, 28, 12, "Preprocessing", "pemulihan sensor, token, stopword,\nstemming Snowball, split 80/20", PURPLE, "#EEEBFC")
    box(0, 0, 28, 12, "Feature engineering", "TF-IDF 1-2 gram: corpus, sinopsis, tag\nbobot blok hasil TPE, SVD 200 dimensi", PURPLE, "#EEEBFC")
    box(36, 25, 30, 17, "Tanpa Spark MLlib", "scikit-learn di driver\nHead A: LogReg, Linear SVM,\nRandom Forest, ensemble SVM + RF\nHead B: kNN, Head C: K-Means\nHead D: ridge + HistGradientBoosting", TEAL, "#E6F5F0")
    box(36, 2, 30, 17, "Dengan Spark MLlib", "terdistribusi di worker Spark\nRegexTokenizer, NGram, CountVectorizer,\nIDF, Normalizer, OneHotEncoder,\nLogisticRegression, LinearSVC, GBT", AMBER, "#FCF1E3")
    box(74, 19, 26, 21, "Evaluasi", "fungsi metrik yang sama\nF1 mikro/makro, Hamming loss,\nPrecision@k, MAE/RMSE/R2,\nsilhouette, genre overlap@10", PURPLE, "#EEEBFC")
    box(74, 2, 26, 11, "Penyajian", "Model/artifacts + lake\ndashboard AI Studio", MUTED, "#F2F3F7")

    arrow(14, 31.4, 14, 28.8)
    arrow(14, 15.4, 14, 12.8)
    arrow(28.8, 7, 35.2, 31)
    arrow(28.8, 6, 35.2, 10)
    arrow(66.8, 33, 73.2, 31)
    arrow(66.8, 10, 73.2, 25)
    arrow(87, 18.2, 87, 13.8)
    save(fig, "a2_1_pipeline_ml.png")


def head_a_bars(metrics: dict) -> None:
    models = metrics["head_a_genre_tagger"]["models"]
    names = [("logreg", "Logistic Regression", GREY), ("svm", "Linear SVM", TEAL), ("rf", "Random Forest", AMBER),
             ("ensemble", "Ensemble SVM + RF", PURPLE)]
    names = [n for n in names if n[0] in models]
    keys = [("f1_macro", "F1 makro"), ("f1_micro", "F1 mikro"), ("precision_at_1", "Precision@1"),
            ("precision_at_3", "Precision@3"), ("f1_macro_fixed_threshold", "F1 makro\nambang 0,5")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(keys))
    width = 0.8 / len(names)
    for i, (key, label, color) in enumerate(names):
        vals = [models[key][k] for k, _ in keys]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, fmt(v, 2), ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(x, [lab for _, lab in keys])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Skor pada set uji (7.932 judul)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(names))
    ax.grid(axis="x", visible=False)
    save(fig, "a2_2_head_a.png")


MODEL_NAMES = {"logreg": "Logistic Regression", "svm": "Linear SVM", "rf": "Random Forest",
               "ensemble": "ensemble SVM + Random Forest"}


def f1_per_genre(metrics: dict) -> None:
    head_a = metrics["head_a_genre_tagger"]
    best = head_a["best_model"]
    f1 = pd.Series(head_a["models"][best]["f1_per_label"]).sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    colors = [PURPLE if v >= 0.6 else (TEAL if v >= 0.45 else AMBER) for v in f1.values]
    ax.barh(f1.index, f1.values, color=colors, height=0.7)
    for i, v in enumerate(f1.values):
        ax.text(v + 0.008, i, fmt(v), va="center", fontsize=7.6, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"F1 per genre, {MODEL_NAMES.get(best, best)}, set uji")
    ax.tick_params(axis="y", labelsize=8.4)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PURPLE, label=">= 0,60"), Patch(color=TEAL, label="0,45 - 0,60"),
                       Patch(color=AMBER, label="< 0,45")], frameon=False, loc="lower right", fontsize=8.5)
    save(fig, "a2_3_f1_genre.png")


def mllib_compare(cmp: dict) -> None:
    a = cmp["head_a"]
    rows = [("scikit-learn\nLogReg", a["sklearn"]["logreg"], TEAL), ("MLlib\nLogReg", a["mllib"]["models"]["logreg"], AMBER),
            ("scikit-learn\nLinear SVM", a["sklearn"]["svm"], TEAL), ("MLlib\nLinearSVC", a["mllib"]["models"]["svm"], AMBER)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(rows))
    for key, offset, alpha, label in (("f1_macro", -0.2, 1.0, "F1 makro"), ("f1_micro", 0.2, 0.45, "F1 mikro")):
        vals = [r[key] for _, r, _ in rows]
        bars = ax1.bar(x + offset, vals, 0.38, color=[c for *_, c in rows], alpha=alpha, label=label)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, fmt(v), ha="center", fontsize=7.8, color=INK)
    ax1.set_xticks(x, [n for n, *_ in rows], fontsize=8.4)
    ax1.set_ylim(0, 0.85)
    ax1.set_ylabel("Skor set uji")
    ax1.set_title("Kualitas (batang pekat = makro, pudar = mikro)", fontsize=9.5, color=INK)
    ax1.grid(axis="x", visible=False)

    secs = [r["fit_seconds"] for _, r, _ in rows]
    bars = ax2.bar(x, secs, 0.55, color=[c for *_, c in rows])
    for b, v in zip(bars, secs):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,.0f} dtk".replace(",", "."), ha="center", fontsize=7.8, color=INK)
    ax2.set_xticks(x, [n for n, *_ in rows], fontsize=8.4)
    ax2.set_ylabel("Waktu latih 35 genre (detik)")
    ax2.set_title("Waktu latih", fontsize=9.5, color=INK)
    ax2.grid(axis="x", visible=False)
    save(fig, "a2_4_mllib_vs_sklearn.png")


def loss_head_a(cmp: dict) -> None:
    """Kurva objective MLlib per iterasi dan log loss latih/validasi/uji keempat model Head A."""
    a = cmp["head_a"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    for kind, color, label in (("logreg", PURPLE, "LogisticRegression (log loss)"), ("svm", AMBER, "LinearSVC (hinge loss)")):
        hist = a["mllib"]["models"][kind]["objective_history"]
        length = max(len(h) for h in hist.values())
        mat = np.full((len(hist), length), np.nan)
        for r, h in enumerate(hist.values()):
            mat[r, :len(h)] = h
            mat[r, len(h):] = h[-1]
        x = np.arange(length)
        ax1.plot(x, np.nanmean(mat, axis=0), color=color, lw=1.8, label=label)
        ax1.fill_between(x, np.nanmin(mat, axis=0), np.nanmax(mat, axis=0), color=color, alpha=0.12, lw=0)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iterasi optimisasi")
    ax1.set_ylabel("Nilai objective (skala log)")
    ax1.set_title("Spark MLlib: rata-rata 35 genre, pita = min-maks", fontsize=9.5, color=INK)
    ax1.legend(frameon=False, fontsize=8.4)

    rows = [("scikit-learn\nLogReg", a["sklearn"]["logreg"]), ("MLlib\nLogReg", a["mllib"]["models"]["logreg"]),
            ("scikit-learn\nLinear SVM", a["sklearn"]["svm"]), ("MLlib\nLinearSVC", a["mllib"]["models"]["svm"])]
    x = np.arange(len(rows))
    for key, off, color, label in (("log_loss_train", -0.27, GREY, "Latih"), ("log_loss_val", 0, TEAL, "Validasi"),
                                    ("log_loss_test", 0.27, PURPLE, "Uji")):
        vals = [r[key] for _, r in rows]
        bars = ax2.bar(x + off, vals, 0.26, color=color, label=label)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, fmt(v), ha="center", fontsize=6.8, color=INK, rotation=90, va="bottom")
    ax2.set_xticks(x, [n for n, _ in rows], fontsize=8.2)
    ax2.set_ylabel("Log loss rata-rata 35 genre")
    ax2.set_title("Binary cross-entropy (protokol Fase 5)", fontsize=9.5, color=INK)
    ax2.set_ylim(0, max(r[k] for _, r in rows for k in ("log_loss_train", "log_loss_val", "log_loss_test")) * 1.35)
    ax2.legend(frameon=False, fontsize=8.4, ncol=3, loc="upper left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "a2_5_loss_head_a.png")


def loss_head_d(cmp: dict) -> None:
    d = cmp["head_d"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for key, color, name in (("sklearn_hgb", TEAL, "HistGradientBoosting"), ("mllib_gbt", AMBER, "MLlib GBTRegressor")):
        tr, te = d[key]["mse_train_curve"], d[key]["mse_test_curve"]
        ax.plot(np.arange(1, len(tr) + 1), tr, color=color, lw=1.3, ls="--", label=f"{name}, latih")
        ax.plot(np.arange(1, len(te) + 1), te, color=color, lw=1.9, label=f"{name}, uji")
    ax.axhline(d["baseline_rmse"] ** 2, color=GREY, lw=1, ls=":")
    ax.text(len(d["sklearn_hgb"]["mse_test_curve"]) * 0.99, d["baseline_rmse"] ** 2 + 0.003, "baseline (rata-rata)",
            ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("Jumlah pohon")
    ax.set_ylabel("MSE rating")
    ax.legend(frameon=False, fontsize=8.2, ncol=2)
    save(fig, "a2_9_loss_head_d.png")


def community_graph(catalog: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    comm = nodes["community"].to_numpy()

    top = 14
    e = edges.assign(a=comm[edges["src"].to_numpy()], b=comm[edges["dst"].to_numpy()])
    e = e[(e.a != e.b) & (e.a < top) & (e.b < top)]
    e = e.assign(lo=e[["a", "b"]].min(axis=1), hi=e[["a", "b"]].max(axis=1)).groupby(["lo", "hi"]).size()
    meta = nx.Graph()
    for cid in range(top):
        members = catalog[comm == cid]
        genres = pd.Series([g for row in members["genre_list"] for g in (row if row is not None else [])]).value_counts()
        origin = members["type_origin"].fillna("?").value_counts().index[0]
        label_genres = [g for g in genres.index if g not in ("Romance", "Fantasy", "Drama")][:2] or list(genres.index[:2])
        meta.add_node(cid, size=len(members), label=f"K{cid}\n{' + '.join(label_genres)}\n({origin}, {len(members):,})".replace(",", "."))
    for (a, b), w in e.items():
        if w >= 400:
            meta.add_edge(a, b, weight=w)
    # Urutan melingkar mengikuti vektor Fiedler, supaya komunitas yang banyak
    # bertukar sisi duduk berdekatan dan garisnya tidak menyilang ke seberang lingkaran.
    order = nx.spectral_ordering(meta, weight="weight", seed=7)
    angles = {n: np.pi * (2 * i + 1) / len(order) for i, n in enumerate(order)}
    pos = {n: (np.cos(a), np.sin(a)) for n, a in angles.items()}
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.axis("off")
    ax.set_aspect("equal")
    widths = [0.4 + 5 * meta[u][v]["weight"] / max(w for *_, w in meta.edges(data="weight")) for u, v in meta.edges]
    nx.draw_networkx_edges(meta, pos, width=widths, edge_color=GREY, alpha=0.85, ax=ax)
    sizes = [meta.nodes[n]["size"] / 2.6 for n in meta.nodes]
    nx.draw_networkx_nodes(meta, pos, node_size=sizes, node_color=PURPLE, alpha=0.9, ax=ax)
    for n, a in angles.items():
        c, s = np.cos(a), np.sin(a)
        # Label di sisi kiri/kanan dirata ke luar agar teks panjang tidak menutupi simpulnya;
        # label di busur atas diletakkan terpusat di atas simpul.
        ha = "center" if s > 0.75 else "right" if c < -0.35 else "left" if c > 0.35 else "center"
        r = 1.17 if ha != "center" else 1.36
        ax.text(r * c, r * s, meta.nodes[n]["label"], ha=ha, va="center", multialignment="center",
                fontsize=9.5, color=INK)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.55, 1.55)
    save(fig, "a2_6_komunitas.png")


def homophily_scatter(sim: dict) -> None:
    h = pd.DataFrame(sim["homophily"])
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(h["neighbour_share"], h["f1"], s=18 + 220 * h["prevalence"], color=PURPLE, alpha=0.75, edgecolor="white")
    coef = np.polyfit(h["neighbour_share"], h["f1"], 1)
    xs = np.linspace(0, h["neighbour_share"].max() * 1.03, 50)
    ax.plot(xs, np.polyval(coef, xs), color=TEAL, lw=1.4, ls="--")
    label = {"Romance", "Fantasy", "Gender Bender", "Sports", "Mecha", "Shoujo Ai", "Seinen", "Tragedy", "Comedy",
             "Xianxia", "Psychological", "Yaoi", "Shounen Ai", "Slice of Life", "Drama", "Action"}
    for r in h.itertuples():
        if r.genre in label:
            offset = {"Seinen": (4, -10), "Tragedy": (-38, 5), "Slice of Life": (6, -12),
                      "Shounen Ai": (-6, 6)}.get(r.genre, (4, 3))
            ax.annotate(r.genre, (r.neighbour_share, r.f1), xytext=offset, textcoords="offset points", fontsize=7.8, color=INK)
    rho = sim["spearman_f1"]["neighbour_share"][0]
    ax.text(0.02, 0.96, f"Spearman rho = {fmt(rho, 3)} (35 genre)", transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.set_xlabel("Homofili: porsi tetangga kNN yang bergenre sama")
    ref = sim.get("homophily_reference_model", "svm")
    ax.set_ylabel(f"F1 genre ({MODEL_NAMES.get(ref, ref)}, set uji)")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0.1, 0.92)
    save(fig, "a2_7_homofili_f1.png")


def degree_hist(edges: pd.DataFrame, n: int) -> None:
    deg = np.bincount(np.concatenate([edges["src"].to_numpy(), edges["dst"].to_numpy()]), minlength=n)
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    bins = np.arange(10, deg.max() + 2, 2)
    ax.hist(deg, bins=bins, color=PURPLE)
    ax.set_yscale("log")
    ax.set_xlabel("Derajat simpul (k = 10, sehingga derajat minimum 10)")
    ax.set_ylabel("Jumlah novel (skala log)")
    ax.grid(axis="x", visible=False)
    save(fig, "a2_8_derajat.png")


def main() -> None:
    metrics = json.loads((ROOT / "Model/artifacts/metrics.json").read_text(encoding="utf-8"))
    sim = json.loads((ROOT / "Doc/generated/similarity_graph.json").read_text(encoding="utf-8"))
    catalog = pd.read_parquet(ROOT / "Model/artifacts/catalog.parquet")
    edges = pd.read_parquet(ROOT / "Doc/generated/similarity_graph_edges.parquet")

    pipeline_diagram()
    head_a_bars(metrics)
    f1_per_genre(metrics)
    cmp_path = ROOT / "Doc/generated/mllib_comparison.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
        mllib_compare(cmp)
        loss_head_a(cmp)
        loss_head_d(cmp)
    nodes = pd.read_parquet(ROOT / "Doc/generated/similarity_graph_nodes.parquet")
    assert (nodes["series_id"].to_numpy() == catalog["series_id"].to_numpy()).all()
    community_graph(catalog, nodes, edges)
    homophily_scatter(sim)
    degree_hist(edges, len(catalog))


if __name__ == "__main__":
    main()
