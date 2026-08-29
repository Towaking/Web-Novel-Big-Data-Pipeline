"""
NovelLens — antarmuka produk
============================
Aplikasi Streamlit yang menyajikan hasil pipeline sebagai produk siap pakai,
bukan panel eksperimen. Konsekuensinya pada desain:

  * Tidak ada detail infrastruktur (endpoint storage, jumlah objek per bucket,
    daftar centang fase pipeline). Itu urusan operator, bukan pengguna.
  * Kegagalan backend ditampilkan sebagai pesan manusiawi; jejak teknisnya
    disembunyikan di balik expander untuk pemelihara.
  * Angka performa model tetap ditampilkan — di halaman Metodologi — karena
    pengguna berhak tahu seberapa jauh keluaran AI boleh dipercaya. Yang
    dibuang adalah artefak proses coba-coba, bukan transparansinya.

Data dibaca dari hasil kueri Spark SQL yang sudah dimaterialisasi (Fase 3);
tidak ada Spark yang dijalankan dari sini agar interaksi tetap instan.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.common.config import get_settings
from src.common.model_registry import ModelNotTrainedError, artifacts_available, load_registry
from src.common.storage import get_s3_client, read_parquet_dir

st.set_page_config(
    page_title="NovelLens — Intelijen Katalog Novel",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SETTINGS = get_settings()

# --- Identitas visual -------------------------------------------------------
BRAND = "#7C5CFF"
ACCENT = "#22D3A7"
MUTED = "#8B93A7"
SEQUENCE = ["#7C5CFF", "#22D3A7", "#FFB454", "#FF6B9D", "#4CC9F0", "#B388FF", "#FF8F6B", "#5EEAD4"]
SCALE = [[0.0, "#1E1B4B"], [0.45, "#5B4BD6"], [0.75, "#7C5CFF"], [1.0, "#22D3A7"]]

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px;}}
      header[data-testid="stHeader"] {{background: transparent;}}

      .nl-hero {{
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.6rem;
        background: linear-gradient(135deg, rgba(124,92,255,.22), rgba(34,211,167,.10));
        border: 1px solid rgba(124,92,255,.30);
      }}
      .nl-hero h1 {{margin: 0; font-size: 1.9rem; letter-spacing: -.02em;}}
      .nl-hero p {{margin: .45rem 0 0; color: {MUTED}; font-size: 1rem; max-width: 62ch;}}

      .nl-card {{
        padding: 1.05rem 1.15rem; border-radius: 14px; height: 100%;
        background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08);
      }}
      .nl-card .lbl {{color: {MUTED}; font-size: .78rem; text-transform: uppercase;
                     letter-spacing: .07em; margin-bottom: .35rem;}}
      .nl-card .val {{font-size: 1.75rem; font-weight: 650; line-height: 1.15;}}
      .nl-card .sub {{color: {MUTED}; font-size: .82rem; margin-top: .25rem;}}

      .nl-chip {{
        display: inline-block; padding: .16rem .6rem; margin: .12rem .22rem .12rem 0;
        border-radius: 999px; font-size: .75rem; background: rgba(124,92,255,.16);
        border: 1px solid rgba(124,92,255,.34); color: #C9BDFF;
      }}
      .nl-chip.on {{background: rgba(34,211,167,.16); border-color: rgba(34,211,167,.4); color: #8FF0D6;}}

      .nl-note {{color: {MUTED}; font-size: .84rem; line-height: 1.5; margin-top: .4rem;}}
      .nl-syn {{color: #C3C9D8; font-size: .87rem; line-height: 1.55;}}
      h2, h3 {{letter-spacing: -.01em;}}
      [data-testid="stMetricValue"] {{font-size: 1.5rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Lapisan data
# ---------------------------------------------------------------------------
class DataUnavailable(RuntimeError):
    """Data produk belum siap disajikan (pipeline belum dijalankan)."""


@st.cache_data(ttl=900, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Baca satu tabel hasil analitik dari data lake."""
    try:
        return read_parquet_dir(SETTINGS.minio.bucket_curated, f"analytics/{name}")
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc


@st.cache_data(ttl=120, show_spinner=False)
def backend_online() -> tuple[bool, str]:
    """Cek ketersediaan penyimpanan data tanpa membocorkan detailnya ke UI."""
    try:
        get_s3_client(SETTINGS).list_buckets()
        return True, ""
    except Exception as exc:
        return False, str(exc)


@st.cache_resource(show_spinner=False)
def get_registry():
    return load_registry()


# ---------------------------------------------------------------------------
# Komponen UI
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="nl-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(container, label: str, value: str, sub: str = "") -> None:
    container.markdown(
        f'<div class="nl-card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def chips(values, active: bool = False) -> str:
    cls = "nl-chip on" if active else "nl-chip"
    return "".join(f'<span class="{cls}">{v}</span>' for v in values)


def style_fig(fig, height: int | None = None):
    """Samakan gaya seluruh grafik dengan tema aplikasi."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DAE5", size=13),
        margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor="#161B26", bordercolor="rgba(255,255,255,.15)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    if height:
        fig.update_layout(height=height)
    return fig


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value]
    return [str(value)]


def novel_card(row: pd.Series, show_similarity: bool = True) -> None:
    """Kartu hasil rekomendasi: judul, genre, cuplikan sinopsis, metrik."""
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"**{row['title']}**")
            st.markdown(chips(as_list(row.get("genre_list"))[:5]), unsafe_allow_html=True)
            snippet = row.get("synopsis_snippet")
            if isinstance(snippet, str) and snippet.strip():
                text = snippet.strip()
                st.markdown(
                    f'<div class="nl-syn">{text[:260]}{"…" if len(text) > 260 else ""}</div>',
                    unsafe_allow_html=True,
                )
        with right:
            rating = row.get("rating_score")
            st.metric("Rating", f"{rating:.1f}" if pd.notna(rating) else "—")
            if show_similarity and pd.notna(row.get("similarity")):
                st.caption(f"Kemiripan {row['similarity']:.0%}")

        meta = []
        if pd.notna(row.get("chapter_count")):
            meta.append(f"{int(row['chapter_count']):,} bab")
        if pd.notna(row.get("rating_votes")):
            meta.append(f"{int(row['rating_votes']):,} suara")
        if isinstance(row.get("type_origin"), str):
            meta.append(row["type_origin"])
        cols = st.columns([4, 1])
        cols[0].caption(" · ".join(meta))
        if isinstance(row.get("url"), str) and row["url"].startswith("http"):
            cols[1].link_button("Buka", row["url"], use_container_width=True)


# ---------------------------------------------------------------------------
# HALAMAN — Ringkasan
# ---------------------------------------------------------------------------
def page_overview() -> None:
    hero(
        "📖 NovelLens",
        "Intelijen katalog untuk 4.461 judul web novel: apa yang dibaca orang, "
        "apa yang mereka nilai tinggi, dan judul apa yang sebaiknya direkomendasikan.",
    )

    viz = load_table("viz_novels")
    reliable = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    genres = load_table("genre_frequency")
    ranking = load_table("genre_ranking").sort_values("avg_weighted_rating", ascending=False)

    cols = st.columns(4)
    card(cols[0], "Judul dalam katalog", f"{len(viz):,}", f"{viz['primary_genre'].nunique()} genre utama")
    card(cols[1], "Rating rata-rata", f"{viz['weighted_rating'].mean():.2f}",
         f"{len(reliable):,} judul dengan penilaian mapan")
    card(cols[2], "Median panjang", f"{viz['chapter_count'].median():,.0f} bab",
         f"terpanjang {viz['chapter_count'].max():,.0f} bab")
    card(cols[3], "Genre teratas", genres.iloc[0]["genre"],
         f"{genres.iloc[0]['pct_of_catalog']:.0f}% dari katalog")

    st.write("")
    st.subheader("Yang perlu diketahui")

    corr = load_table("correlation_metrics").set_index("pair")
    richness = load_table("text_richness_vs_rating")
    top_genre, longest = ranking.iloc[0], ranking.sort_values("avg_chapters", ascending=False).iloc[0]
    r_len = corr.loc["chapter_count ~ weighted_rating", "pearson_r"]
    gain = richness.iloc[-1]["avg_weighted_rating"] - richness.iloc[0]["avg_weighted_rating"]

    insight_cols = st.columns(3)
    card(
        insight_cols[0], "Panjang ≠ disukai", f"r = {r_len}",
        "Korelasi jumlah bab dengan rating justru negatif tipis. Menambah bab "
        "tidak membeli penerimaan pembaca.",
    )
    card(
        insight_cols[1], "Genre paling dihargai", top_genre["genre"],
        f"Rata-rata {top_genre['avg_weighted_rating']:.2f} dari {int(top_genre['n_novels'])} judul. "
        f"Terpanjang: {longest['genre']} (~{longest['avg_chapters']:,.0f} bab).",
    )
    card(
        insight_cols[2], "Sinopsis yang kaya menang", f"+{gain:.2f} poin",
        "Judul dengan sinopsis 300+ kata dinilai lebih tinggi dibanding yang "
        "di bawah 50 kata.",
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Genre terpopuler")
        subset = genres.nlargest(10, "n_novels").sort_values("n_novels")
        fig = px.bar(
            subset, x="n_novels", y="genre", orientation="h",
            color="avg_weighted_rating", color_continuous_scale=SCALE,
            labels={"n_novels": "Jumlah judul", "genre": "", "avg_weighted_rating": "Rating"},
        )
        fig.update_coloraxes(colorbar_title_text="Rating")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with right:
        st.subheader("Paling dihargai pembaca")
        top = (
            reliable.nlargest(8, "weighted_rating")[["title", "weighted_rating", "rating_votes"]]
            .rename(columns={"title": "Judul", "weighted_rating": "Rating", "rating_votes": "Suara"})
        )
        st.dataframe(
            top, hide_index=True, use_container_width=True, height=380,
            column_config={
                "Rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=5, format="%.2f"),
            },
        )


# ---------------------------------------------------------------------------
# HALAMAN — Analitik Katalog
# ---------------------------------------------------------------------------
def page_analytics() -> None:
    hero(
        "Analitik katalog",
        "Struktur pasar novel terjemahan: komposisi genre, hubungan panjang cerita "
        "dengan penerimaan pembaca, dan kosakata yang mendominasi ceritanya.",
    )

    viz = load_table("viz_novels")
    tab_genre, tab_length, tab_text, tab_market = st.tabs(
        ["Genre", "Panjang & Rating", "Bahasa Cerita", "Peta Pasar"]
    )

    # --- Genre ---------------------------------------------------------
    with tab_genre:
        genres = load_table("genre_frequency")
        top_n = st.slider("Genre ditampilkan", 5, min(30, len(genres)), 15)
        subset = genres.nlargest(top_n, "n_novels").sort_values("n_novels")

        fig = px.bar(
            subset, x="n_novels", y="genre", orientation="h",
            color="avg_weighted_rating", color_continuous_scale=SCALE,
            labels={"n_novels": "Jumlah judul", "genre": "", "avg_weighted_rating": "Rating"},
            hover_data={"pct_of_catalog": ":.1f", "avg_chapters": ":.0f"},
        )
        fig.update_coloraxes(colorbar_title_text="Rating")
        st.plotly_chart(style_fig(fig, max(360, 26 * top_n)), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Panjang batang = jumlah judul, warna = rating rata-rata. '
            "Genre terbanyak belum tentu genre yang paling disukai — bandingkan keduanya.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Kombinasi genre tersering**")
            combos = load_table("genre_combinations").head(12)
            st.dataframe(
                combos[["genre_combo", "n_novels", "avg_weighted_rating"]].rename(
                    columns={"genre_combo": "Kombinasi", "n_novels": "Judul",
                             "avg_weighted_rating": "Rating"}),
                hide_index=True, use_container_width=True, height=430,
            )
        with col2:
            st.markdown("**Sebaran rating per genre**")
            top_genres = viz["primary_genre"].value_counts().head(8).index.tolist()
            box_df = viz[viz["primary_genre"].isin(top_genres) & viz["weighted_rating"].notna()]
            fig = px.box(
                box_df, x="primary_genre", y="weighted_rating", color="primary_genre",
                points=False, color_discrete_sequence=SEQUENCE,
                labels={"primary_genre": "", "weighted_rating": "Rating"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    # --- Panjang & rating ----------------------------------------------
    with tab_length:
        col_a, col_b = st.columns([3, 2])
        available = sorted(viz["primary_genre"].dropna().unique().tolist())
        default = load_table("genre_frequency").nlargest(5, "n_novels")["genre"].tolist()
        picked = col_a.multiselect("Genre", available, default=[g for g in default if g in available])
        only_reliable = col_b.checkbox("Hanya judul dengan penilaian mapan", value=True)

        scatter = viz.dropna(subset=["chapter_count", "weighted_rating"])
        if picked:
            scatter = scatter[scatter["primary_genre"].isin(picked)]
        if only_reliable:
            scatter = scatter[scatter["is_rated_reliable"] == True]  # noqa: E712

        if scatter.empty:
            st.info("Tidak ada judul yang cocok dengan filter ini.")
        else:
            fig = px.scatter(
                scatter, x="chapter_count", y="weighted_rating", color="primary_genre",
                size="rating_votes", size_max=26, opacity=0.62, log_x=True,
                color_discrete_sequence=SEQUENCE, hover_name="title",
                hover_data={"rating_score": True, "rating_votes": True, "type_origin": True},
                labels={"chapter_count": "Jumlah bab (skala log)",
                        "weighted_rating": "Rating", "primary_genre": ""},
            )
            buckets = load_table("length_vs_rating")
            fig.add_scatter(
                x=buckets["avg_chapters"], y=buckets["avg_weighted_rating"],
                mode="lines+markers", name="Rata-rata per kelompok panjang",
                line=dict(color="#FFB454", width=3, dash="dash"),
            )
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)

            corr = load_table("correlation_metrics")
            r = corr.loc[corr["pair"] == "chapter_count ~ weighted_rating", "pearson_r"].iloc[0]
            st.markdown(
                f'<div class="nl-note">Garis oranye adalah rata-rata rating per kelompok panjang. '
                f"Korelasinya r = {r} — praktis datar lalu menurun di atas 1.000 bab: cerita yang "
                "terlalu panjang cenderung kehilangan pembaca yang puas.</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Rating menurut kelompok panjang**")
        buckets = load_table("length_vs_rating")
        fig = px.bar(
            buckets, x="length_bucket", y="avg_weighted_rating", color="n_novels",
            color_continuous_scale=SCALE,
            labels={"length_bucket": "", "avg_weighted_rating": "Rating", "n_novels": "Judul"},
        )
        fig.update_yaxes(range=[3.3, 3.65])
        fig.update_coloraxes(colorbar_title_text="Judul")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    # --- Bahasa cerita --------------------------------------------------
    with tab_text:
        tokens = load_table("top_tokens")
        st.markdown("**Kosakata yang mendominasi sinopsis**")
        col_wc, col_list = st.columns([3, 1], gap="large")
        with col_wc:
            st.image(render_wordcloud(tokens), use_column_width=True)
        with col_list:
            st.dataframe(
                tokens.head(14)[["token", "n_novels"]].rename(
                    columns={"token": "Kata", "n_novels": "Judul"}),
                hide_index=True, use_container_width=True, height=420,
            )
        st.markdown(
            '<div class="nl-note">Diambil dari sinopsis asli (bukan tag katalog), '
            "setelah pembuangan stopword.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Panjang sinopsis vs rating**")
            richness = load_table("text_richness_vs_rating")
            fig = px.bar(
                richness, x="text_richness", y="avg_weighted_rating", color="n_novels",
                color_continuous_scale=SCALE,
                labels={"text_richness": "", "avg_weighted_rating": "Rating", "n_novels": "Judul"},
            )
            fig.update_yaxes(range=[3.35, 3.8])
            fig.update_layout(xaxis_tickangle=-18)
            fig.update_coloraxes(colorbar_title_text="Judul")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with col2:
            st.markdown("**Elemen cerita dengan rating terbaik**")
            tags = load_table("best_rated_tags").head(14).sort_values("avg_weighted_rating")
            fig = px.bar(
                tags, x="avg_weighted_rating", y="tag", orientation="h", color="n_novels",
                color_continuous_scale=SCALE,
                labels={"avg_weighted_rating": "Rating", "tag": "", "n_novels": "Judul"},
            )
            fig.update_xaxes(range=[3.4, 4.0])
            fig.update_coloraxes(colorbar_title_text="Judul")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    # --- Peta pasar -----------------------------------------------------
    with tab_market:
        st.markdown("**Rating rata-rata per genre × negara asal**")
        matrix = load_table("genre_origin_matrix")
        pivot = matrix.pivot(index="genre", columns="type_origin", values="avg_weighted_rating")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(
            pivot, color_continuous_scale=SCALE, aspect="auto", text_auto=".2f",
            labels=dict(x="Negara asal", y="", color="Rating"),
        )
        st.plotly_chart(style_fig(fig, max(380, 30 * len(pivot))), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Sel kosong = kombinasi dengan kurang dari 25 judul, '
            "sengaja tidak dihitung agar rata-rata tidak dibentuk segelintir judul.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Tren per dekade**")
            decade = load_table("decade_trend")
            fig = px.line(
                decade, x="decade", y="avg_weighted_rating", markers=True,
                labels={"decade": "", "avg_weighted_rating": "Rating"},
            )
            fig.update_traces(line=dict(color=BRAND, width=3), marker=dict(size=10, color=ACCENT))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        with col2:
            st.markdown("**Judul terpanjang di katalog**")
            outliers = load_table("stat_outlier_novels").head(10)
            st.dataframe(
                outliers[["title", "chapter_count", "rating_score"]].rename(
                    columns={"title": "Judul", "chapter_count": "Bab", "rating_score": "Rating"}),
                hide_index=True, use_container_width=True, height=340,
            )


@st.cache_data(ttl=900, show_spinner=False)
def render_wordcloud(tokens: pd.DataFrame) -> bytes:
    """Render word cloud dengan latar transparan agar menyatu dengan tema."""
    from wordcloud import WordCloud

    cloud = WordCloud(
        width=1400, height=560, mode="RGBA", background_color=None,
        colormap="cool", max_words=180, prefer_horizontal=0.92,
    ).generate_from_frequencies(dict(zip(tokens["token"], tokens["freq"])))

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# HALAMAN — AI Studio
# ---------------------------------------------------------------------------
CONTOH_SINOPSIS = (
    "Setelah dikhianati sekutunya dan tewas di puncak menara, seorang pendekar "
    "terbangun kembali di tubuh dirinya sepuluh tahun lebih muda. Kali ini ia "
    "berniat menguasai teknik pedang terlarang sebelum musuh lamanya bangkit."
)
CONTOH_TAG = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"


def page_studio() -> None:
    hero(
        "AI Studio",
        "Empat asisten yang bekerja dari isi cerita: mencari judul serupa, "
        "melabeli genre, memperkirakan penerimaan pembaca, dan mengenali arketipe naratif.",
    )

    if not artifacts_available():
        st.info("Fitur AI sedang disiapkan. Coba lagi setelah proses pelatihan model selesai.")
        return

    try:
        registry = get_registry()
    except ModelNotTrainedError as exc:
        st.info("Fitur AI belum tersedia.")
        with st.expander("Detail teknis"):
            st.code(str(exc))
        return

    tab_rec, tab_genre, tab_rating, tab_arch = st.tabs(
        ["Rekomendasi", "Pelabelan Genre", "Perkiraan Rating", "Arketipe Cerita"]
    )

    with tab_rec:
        _studio_recommend(registry)
    with tab_genre:
        _studio_genre(registry)
    with tab_rating:
        _studio_rating(registry)
    with tab_arch:
        _studio_archetype(registry)


def _studio_recommend(registry) -> None:
    st.markdown("Temukan judul yang secara isi paling dekat dengan yang Anda sukai.")
    mode = st.radio("Cari dari", ["Judul di katalog", "Deskripsi bebas"], horizontal=True)
    k = st.slider("Jumlah rekomendasi", 3, 12, 6)

    try:
        if mode == "Judul di katalog":
            titles = registry.catalog.sort_values("rating_votes", ascending=False)["title"].tolist()
            choice = st.selectbox("Judul acuan", titles)
            series_id = int(registry.catalog.loc[registry.catalog["title"] == choice, "series_id"].iloc[0])
            results = registry.recommend_by_series(series_id, k=k)
        else:
            text = st.text_area("Cerita seperti apa yang Anda cari?", CONTOH_SINOPSIS, height=110)
            results = registry.recommend_by_text(text, k=k)
    except (ValueError, KeyError):
        st.warning("Masukan belum cukup untuk dicarikan padanan. Tambahkan detail cerita.")
        return

    st.write("")
    for _, row in results.iterrows():
        novel_card(row)


def _studio_genre(registry) -> None:
    st.markdown(
        "Tempel sinopsis dan tag sebuah novel, lalu pilih model yang ingin dipakai. "
        "Setiap model punya cara berbeda dalam menimbang kata."
    )

    names = registry.genre_model_names
    info = {name: registry.genre_model_info(name) for name in names}

    choice = st.radio(
        "Model prediksi",
        names,
        horizontal=True,
        format_func=lambda n: f"{info[n].get('label', n)} · F1 {info[n].get('f1_macro', 0):.2f}",
    )
    selected = info[choice]
    st.markdown(
        f'<div class="nl-note"><b>{selected.get("family", "")}</b> — {selected.get("description", "")}</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        synopsis = st.text_area("Sinopsis", CONTOH_SINOPSIS, height=150)
        tags = st.text_input("Tag (opsional, pisahkan dengan koma)", CONTOH_TAG)
    with col2:
        origin = st.selectbox("Negara asal", ["CN", "KR", "JP", "EN", "Other"])
        novel_type = st.selectbox("Tipe", ["Web Novel", "Light Novel", "Published Novel"])
        sensitivity = st.slider(
            "Ketelitian", 0.6, 1.6, 1.0, 0.05,
            help="Di bawah 1 menampilkan lebih banyak genre; di atas 1 hanya yang paling yakin.",
        )

    if st.button("Prediksi genre", type="primary"):
        try:
            result = registry.predict_genres(
                synopsis, tags=tags, model=choice, top_k=12,
                type_origin=origin, novel_type=novel_type, sensitivity=sensitivity,
            )
        except ValueError:
            st.warning("Teks belum cukup untuk dianalisis. Tambahkan sinopsis atau tag.")
            return

        predicted = result[result["diprediksi"]]["genre"].tolist()
        if predicted:
            st.markdown("**Genre terprediksi**")
            st.markdown(chips(predicted, active=True), unsafe_allow_html=True)
        else:
            st.info("Tidak ada genre yang cukup meyakinkan. Turunkan ketelitian atau tambah detail.")

        st.write("")
        plot = result.sort_values("probability")
        fig = px.bar(
            plot, x="probability", y="genre", orientation="h",
            color="diprediksi", color_discrete_map={True: ACCENT, False: "rgba(255,255,255,.18)"},
            labels={"probability": "Keyakinan model", "genre": "", "diprediksi": ""},
        )
        fig.add_scatter(
            x=plot["threshold"], y=plot["genre"], mode="markers",
            marker=dict(symbol="line-ns", size=16, line=dict(color="#FFB454", width=2)),
            name="Ambang genre",
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Garis oranye adalah ambang khusus tiap genre. '
            "Genre langka punya ambang lebih rendah, sehingga tetap bisa terdeteksi "
            "tanpa membanjiri hasil dengan genre umum.</div>",
            unsafe_allow_html=True,
        )


def _studio_rating(registry) -> None:
    head_d = registry.metrics["head_d_rating_estimator"]
    if not head_d.get("trained"):
        st.info("Perkiraan rating belum tersedia.")
        return

    st.markdown(
        "Perkirakan bagaimana pembaca akan menilai sebuah cerita — berguna sebelum "
        "judul baru punya rating sungguhan."
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        text = st.text_area("Sinopsis atau tag", CONTOH_TAG, height=130, key="rating_text")
    with col2:
        chapters = st.number_input("Rencana jumlah bab", 1, 12000, 300, 10)
        year = st.number_input("Tahun rilis", 1990, 2030, 2024)
        origin = st.selectbox("Negara asal", ["CN", "KR", "JP", "EN", "Other"], key="rating_origin")
        completed = st.checkbox("Sudah tamat")

    if st.button("Perkirakan rating", type="primary"):
        try:
            predicted = registry.estimate_rating(
                text, chapter_count=float(chapters), year=int(year),
                is_completed=completed, type_origin=origin,
            )
        except (ValueError, ModelNotTrainedError):
            st.warning("Masukan belum cukup untuk diperkirakan.")
            return

        average = registry.catalog["rating_score"].mean()
        cols = st.columns(3)
        card(cols[0], "Perkiraan rating", f"{predicted:.2f}", "skala 1–5")
        card(cols[1], "Dibanding katalog", f"{predicted - average:+.2f}", f"rata-rata {average:.2f}")
        card(cols[2], "Rentang kesalahan", f"±{head_d['mae']:.2f}", "berdasarkan uji tertahan")

        st.markdown(
            '<div class="nl-note">Perkiraan ini hanya membaca isi cerita, bukan kualitas '
            "penulisan atau promosi. Konten menjelaskan sebagian kecil variasi rating, jadi "
            "perlakukan angkanya sebagai indikasi arah — bukan vonis.</div>",
            unsafe_allow_html=True,
        )


def _studio_archetype(registry) -> None:
    head_c = registry.metrics["head_c_clustering"]
    profiles = pd.DataFrame(head_c["profiles"])

    st.markdown(
        f"Katalog terbagi menjadi {head_c['best_k']} arketipe naratif. "
        "Tempel sebuah cerita untuk melihat ia masuk keluarga mana."
    )
    text = st.text_area("Sinopsis atau tag", CONTOH_TAG, height=110, key="arch_text")

    if st.button("Kenali arketipe", type="primary"):
        try:
            assignment = registry.assign_cluster(text)
        except ValueError:
            st.warning("Teks belum cukup untuk dikenali.")
            return

        profile = assignment["profile"]
        with st.container(border=True):
            st.markdown(f"### Arketipe #{assignment['cluster_id']}")
            st.markdown(chips(profile.get("top_terms", [])[:8], active=True), unsafe_allow_html=True)
            cols = st.columns(3)
            card(cols[0], "Judul sekeluarga", f"{profile.get('size', 0):,}")
            card(cols[1], "Rating rata-rata", f"{profile.get('avg_weighted_rating', 0):.2f}")
            card(cols[2], "Panjang rata-rata", f"{profile.get('avg_chapters', 0):,.0f} bab")
            st.write("")
            st.caption("Contoh judul: " + " · ".join(profile.get("example_titles", [])[:4]))

    st.write("")
    st.markdown("**Seluruh arketipe**")
    overview = profiles.assign(
        istilah=profiles["top_terms"].apply(lambda t: ", ".join(t[:5])),
        genre=profiles["top_genres"].apply(lambda g: ", ".join(list(g.keys())[:3])),
    )[["cluster_id", "size", "avg_weighted_rating", "avg_chapters", "istilah", "genre"]]
    st.dataframe(
        overview.rename(columns={
            "cluster_id": "#", "size": "Judul", "avg_weighted_rating": "Rating",
            "avg_chapters": "Bab", "istilah": "Ciri khas", "genre": "Genre dominan"}),
        hide_index=True, use_container_width=True,
    )


# ---------------------------------------------------------------------------
# HALAMAN — Metodologi
# ---------------------------------------------------------------------------
def page_method() -> None:
    hero(
        "Metodologi & keterbatasan",
        "Bagaimana angka di aplikasi ini dihitung, seberapa akurat modelnya, "
        "dan di titik mana hasilnya tidak boleh dipercaya mentah-mentah.",
    )

    st.subheader("Dari mana datanya")
    st.markdown(
        """
Katalog berisi **4.461 judul** web novel terjemahan beserta metadata, rating
pembaca, dan sinopsis. Sebelum dipakai, data melewati pembersihan:

- Nilai kosong yang disamarkan sebagai angka `0` atau teks `"Unknown"`
  dikembalikan menjadi kosong sungguhan — tanpa ini rata-rata jumlah bab
  meleset sekitar 20%.
- Rating ditampilkan dalam bentuk **rating tertimbang**: judul dengan 2 suara
  tidak diperlakukan sama dengan judul dengan 500 suara. Peringkat apa pun di
  aplikasi ini hanya melibatkan judul dengan minimal 10 suara.
- Sinopsis digabungkan dari sumber terpisah lewat pencocokan tautan, mencakup
  **98,45%** katalog.
        """
    )

    if not artifacts_available():
        return

    metrics = get_registry().metrics
    head_a, head_b = metrics["head_a_genre_tagger"], metrics["head_b_recommender"]
    head_c, head_d = metrics["head_c_clustering"], metrics["head_d_rating_estimator"]

    st.subheader("Performa model pelabelan genre")
    st.markdown(
        "Ketiga model dilatih dan diuji pada pembagian data yang sama "
        f"({head_a['n_train']:,} judul untuk pengembangan · {head_a['n_test']:,} untuk pengujian), "
        "sehingga perbandingannya adil. Hyperparameter dan ambang keputusan tiap genre "
        "dipilih memakai validasi silang di dalam set pengembangan — set uji baru "
        "disentuh sekali, di akhir, agar angka di bawah ini tidak dilebih-lebihkan."
    )

    rows = []
    for name, entry in head_a["models"].items():
        rows.append({
            "Model": entry.get("label", name),
            "Pendekatan": entry.get("family", ""),
            "F1 makro": entry["f1_macro"],
            "F1 mikro": entry["f1_micro"],
            "F1 per judul": entry["f1_samples"],
            "Tanpa penyetelan ambang": entry["f1_macro_fixed_threshold"],
        })
    table = pd.DataFrame(rows).sort_values("F1 makro", ascending=False)
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "F1 makro": st.column_config.ProgressColumn("F1 makro", min_value=0, max_value=1, format="%.3f"),
            "F1 mikro": st.column_config.ProgressColumn("F1 mikro", min_value=0, max_value=1, format="%.3f"),
        },
    )
    st.markdown(
        '<div class="nl-note">Kolom terakhir memperlihatkan skor model yang sama bila memakai '
        "ambang seragam 0,5. Selisihnya adalah kontribusi penyetelan ambang per genre.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Fitur lain")
    cols = st.columns(3)
    card(cols[0], "Rekomendasi", f"{head_b['genre_overlap_at_k']:.0%}",
         f"dari {head_b['k']} saran berbagi genre dengan judul acuan")
    card(cols[1], "Arketipe", f"{head_c['best_k']} kelompok",
         "jumlah kelompok dipilih otomatis lewat silhouette score")
    card(cols[2], "Perkiraan rating", f"±{head_d['mae']:.2f}",
         f"lebih baik {head_d['improvement_over_baseline_pct']:.0f}% dari tebakan rata-rata")

    st.write("")
    st.subheader("Keterbatasan yang perlu diingat")
    st.markdown(
        """
- Model membaca **deskripsi**, bukan naskah. Novel dengan sinopsis buruk akan
  dinilai buruk meski isinya bagus.
- Rekomendasi murni berbasis kemiripan konten; dua judul bertema sama dianggap
  mirip walau mutu penulisannya berbeda jauh.
- Katalog didominasi novel Tiongkok (55%), sehingga akurasi untuk judul
  berbahasa Inggris atau Jepang cenderung lebih rendah.
- Rating berasal dari satu komunitas pembaca terjemahan, bukan populasi umum.
- Arketipe adalah kelompok yang saling bersinggungan, bukan sekat tegas.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Ringkasan": page_overview,
    "Analitik Katalog": page_analytics,
    "AI Studio": page_studio,
    "Metodologi": page_method,
}


def main() -> None:
    with st.sidebar:
        st.markdown("## 📖 NovelLens")
        st.caption("Intelijen katalog web novel")
        st.write("")
        choice = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")
        st.write("")
        st.caption("Data: 4.461 judul · NovelUpdates")

    online, detail = backend_online()
    if not online:
        st.error("Layanan data sedang tidak dapat dijangkau. Silakan coba beberapa saat lagi.")
        with st.expander("Detail teknis"):
            st.code(detail)
        st.stop()

    try:
        PAGES[choice]()
    except DataUnavailable as exc:
        st.warning("Data untuk halaman ini belum tersedia.")
        with st.expander("Detail teknis"):
            st.code(str(exc))


main()
