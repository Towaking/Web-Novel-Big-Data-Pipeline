"""
NovelLens — product interface
=============================
A Streamlit app that presents the pipeline output as a finished product rather
than an experiment console. That choice drives the design:

  * No infrastructure details (storage endpoints, per-bucket object counts,
    pipeline-phase checklists). That is the operator's concern, not the user's.
  * Backend failures surface as plain-language messages; the technical trace is
    tucked behind an expander for maintainers.
  * Model performance numbers stay visible — on the Methodology page — because
    users deserve to know how far the AI output can be trusted. What was cut is
    the trial-and-error residue, not the transparency.

Data is read from materialised Spark SQL results (Phase 3); no Spark job runs
from here, so interaction stays instant.
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
    page_title="NovelLens — Novel Catalogue Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SETTINGS = get_settings()

# --- Visual identity -------------------------------------------------------
BRAND = "#7C5CFF"
ACCENT = "#22D3A7"
MUTED = "#8B93A7"
SEQUENCE = ["#7C5CFF", "#22D3A7", "#FFB454", "#FF6B9D", "#4CC9F0", "#B388FF", "#FF8F6B", "#5EEAD4"]
SCALE = [[0.0, "#1E1B4B"], [0.45, "#5B4BD6"], [0.75, "#7C5CFF"], [1.0, "#22D3A7"]]

st.markdown(
    f"""
    <style>
      /* --- Page rhythm ------------------------------------------------ */
      .block-container {{padding-top: 1.4rem; padding-bottom: 5rem; max-width: 1380px;}}
      header[data-testid="stHeader"] {{background: transparent; height: 0;}}
      #MainMenu, footer {{visibility: hidden;}}

      /* A consistent type scale, not a per-element size free-for-all. */
      h1 {{font-size: 1.95rem; letter-spacing: -.025em; font-weight: 680;}}
      h2 {{font-size: 1.28rem; letter-spacing: -.015em; font-weight: 640; margin-top: .3rem;}}
      h3 {{font-size: 1.02rem; letter-spacing: -.01em; font-weight: 620;}}

      /* --- Hero ---------------------------------------------------------- */
      .nl-hero {{
        position: relative; overflow: hidden;
        padding: 1.75rem 1.9rem; border-radius: 20px; margin-bottom: 1.5rem;
        background:
          radial-gradient(1200px 260px at 8% -40%, rgba(124,92,255,.34), transparent 62%),
          radial-gradient(900px 240px at 92% -30%, rgba(34,211,167,.22), transparent 60%),
          linear-gradient(160deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 18px 42px -28px rgba(0,0,0,.9);
      }}
      .nl-hero h1 {{margin: 0;}}
      .nl-hero p {{margin: .5rem 0 0; color: {MUTED}; font-size: .97rem; max-width: 66ch; line-height: 1.55;}}
      .nl-eyebrow {{
        display: inline-block; margin-bottom: .55rem; padding: .2rem .62rem;
        border-radius: 999px; font-size: .68rem; font-weight: 600;
        letter-spacing: .1em; text-transform: uppercase;
        color: #C9BDFF; background: rgba(124,92,255,.16); border: 1px solid rgba(124,92,255,.32);
      }}

      /* --- Metric cards --------------------------------------------------- */
      .nl-card {{
        position: relative; height: 100%;
        padding: 1.05rem 1.15rem; border-radius: 15px;
        background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
        border: 1px solid rgba(255,255,255,.085);
        transition: border-color .18s ease, transform .18s ease;
      }}
      .nl-card:hover {{border-color: rgba(124,92,255,.42); transform: translateY(-2px);}}
      .nl-card .lbl {{color: {MUTED}; font-size: .72rem; text-transform: uppercase;
                     letter-spacing: .09em; font-weight: 600; margin-bottom: .4rem;}}
      .nl-card .val {{font-size: 1.72rem; font-weight: 660; line-height: 1.1;
                     background: linear-gradient(92deg, #FFFFFF, #C9BDFF);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;}}
      .nl-card .sub {{color: {MUTED}; font-size: .8rem; margin-top: .35rem; line-height: 1.45;}}

      /* --- Genre chips ----------------------------------------------------- */
      .nl-chip {{
        display: inline-block; padding: .18rem .62rem; margin: .14rem .24rem .14rem 0;
        border-radius: 999px; font-size: .735rem; font-weight: 500;
        background: rgba(124,92,255,.14); border: 1px solid rgba(124,92,255,.30); color: #C9BDFF;
      }}
      .nl-chip.on {{background: rgba(34,211,167,.15); border-color: rgba(34,211,167,.40); color: #8FF0D6;}}

      /* --- Supporting text ------------------------------------------------- */
      .nl-note {{
        color: {MUTED}; font-size: .83rem; line-height: 1.6; margin-top: .55rem;
        padding-left: .7rem; border-left: 2px solid rgba(124,92,255,.32);
      }}
      .nl-syn {{color: #BFC6D6; font-size: .865rem; line-height: 1.6;}}

      /* --- Tabs, buttons, tables --------------------------------------------- */
      .stTabs [data-baseweb="tab-list"] {{gap: .35rem; border-bottom: 1px solid rgba(255,255,255,.08);}}
      .stTabs [data-baseweb="tab"] {{
        font-size: .9rem; font-weight: 560; padding: .55rem .95rem;
        border-radius: 9px 9px 0 0;
      }}
      .stTabs [aria-selected="true"] {{background: rgba(124,92,255,.12);}}
      .stButton > button {{border-radius: 10px; font-weight: 580; padding: .5rem 1.05rem;}}
      div[data-testid="stDataFrame"] {{border-radius: 12px; overflow: hidden;}}
      [data-testid="stMetricValue"] {{font-size: 1.42rem; font-weight: 640;}}
      [data-testid="stMetricLabel"] {{color: {MUTED};}}
      hr {{border-color: rgba(255,255,255,.08);}}

      /* --- Sidebar -------------------------------------------------------- */
      section[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.07);}}
      section[data-testid="stSidebar"] .stRadio label {{font-size: .92rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
class DataUnavailable(RuntimeError):
    """Product data is not ready to serve yet (the pipeline has not run)."""


@st.cache_data(ttl=900, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Read one analytics result table from the data lake."""
    try:
        return read_parquet_dir(SETTINGS.minio.bucket_curated, f"analytics/{name}")
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc


@st.cache_data(ttl=120, show_spinner=False)
def backend_online() -> tuple[bool, str]:
    """Check storage availability without leaking its details into the UI."""
    try:
        get_s3_client(SETTINGS).list_buckets()
        return True, ""
    except Exception as exc:
        return False, str(exc)


@st.cache_resource(show_spinner=False)
def get_registry():
    return load_registry()


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    """Page header: eyebrow label, title, one explanatory sentence."""
    tag = f'<span class="nl-eyebrow">{eyebrow}</span>' if eyebrow else ""
    st.markdown(
        f'<div class="nl-hero">{tag}<h1>{title}</h1><p>{subtitle}</p></div>',
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
    """Match every chart to the app theme."""
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
    """Recommendation card: title, genres, synopsis snippet, metrics."""
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
                st.caption(f"Similarity {row['similarity']:.0%}")

        meta = []
        if pd.notna(row.get("chapter_count")):
            meta.append(f"{int(row['chapter_count']):,} chapters")
        if pd.notna(row.get("rating_votes")):
            meta.append(f"{int(row['rating_votes']):,} votes")
        if isinstance(row.get("type_origin"), str):
            meta.append(row["type_origin"])
        cols = st.columns([4, 1])
        cols[0].caption(" · ".join(meta))
        if isinstance(row.get("url"), str) and row["url"].startswith("http"):
            cols[1].link_button("Open", row["url"], use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE — Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    hero(
        "NovelLens",
        "Catalogue intelligence for 4,461 web novel titles: what people read, "
        "what they rate highly, and which titles are worth recommending.",
        eyebrow="Catalogue overview",
    )

    viz = load_table("viz_novels")
    reliable = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    genres = load_table("genre_frequency")
    ranking = load_table("genre_ranking").sort_values("avg_weighted_rating", ascending=False)

    cols = st.columns(4)
    card(cols[0], "Titles in catalogue", f"{len(viz):,}", f"{viz['primary_genre'].nunique()} primary genres")
    card(cols[1], "Average rating", f"{viz['weighted_rating'].mean():.2f}",
         f"{len(reliable):,} titles with an established score")
    card(cols[2], "Median length", f"{viz['chapter_count'].median():,.0f} chapters",
         f"longest {viz['chapter_count'].max():,.0f} chapters")
    card(cols[3], "Top genre", genres.iloc[0]["genre"],
         f"{genres.iloc[0]['pct_of_catalog']:.0f}% of the catalogue")

    st.write("")
    st.subheader("What you need to know")

    corr = load_table("correlation_metrics").set_index("pair")
    richness = load_table("text_richness_vs_rating")
    top_genre, longest = ranking.iloc[0], ranking.sort_values("avg_chapters", ascending=False).iloc[0]
    r_len = corr.loc["chapter_count ~ weighted_rating", "pearson_r"]
    gain = richness.iloc[-1]["avg_weighted_rating"] - richness.iloc[0]["avg_weighted_rating"]

    insight_cols = st.columns(3)
    card(
        insight_cols[0], "Length ≠ acclaim", f"r = {r_len}",
        "Chapter count correlates slightly negatively with rating. Adding more "
        "chapters does not buy reader approval.",
    )
    card(
        insight_cols[1], "Best-rated genre", top_genre["genre"],
        f"Averages {top_genre['avg_weighted_rating']:.2f} across {int(top_genre['n_novels'])} titles. "
        f"Longest: {longest['genre']} (~{longest['avg_chapters']:,.0f} chapters).",
    )
    card(
        insight_cols[2], "Rich synopses win", f"+{gain:.2f} points",
        "Titles with 300+ word synopses score higher than those under "
        "50 words.",
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Most common genres")
        subset = genres.nlargest(10, "n_novels").sort_values("n_novels")
        fig = px.bar(
            subset, x="n_novels", y="genre", orientation="h",
            color="avg_weighted_rating", color_continuous_scale=SCALE,
            labels={"n_novels": "Number of titles", "genre": "", "avg_weighted_rating": "Rating"},
        )
        fig.update_coloraxes(colorbar_title_text="Rating")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with right:
        st.subheader("Highest reader ratings")
        top = (
            reliable.nlargest(8, "weighted_rating")[["title", "weighted_rating", "rating_votes"]]
            .rename(columns={"title": "Title", "weighted_rating": "Rating", "rating_votes": "Votes"})
        )
        st.dataframe(
            top, hide_index=True, use_container_width=True, height=380,
            column_config={
                "Rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=5, format="%.2f"),
            },
        )


# ---------------------------------------------------------------------------
# PAGE — Catalogue Analytics
# ---------------------------------------------------------------------------
def page_analytics() -> None:
    hero(
        "Catalogue analytics",
        "The structure of the translated-novel market: genre mix, how story length "
        "relates to reader reception, and the vocabulary that dominates the stories.",
        eyebrow="Business intelligence",
    )

    viz = load_table("viz_novels")
    tab_genre, tab_length, tab_text, tab_market = st.tabs(
        ["Genres", "Length & Rating", "Story Language", "Market Map"]
    )

    # --- Genres ---------------------------------------------------------
    with tab_genre:
        genres = load_table("genre_frequency")
        top_n = st.slider("Genres shown", 5, min(30, len(genres)), 15)
        subset = genres.nlargest(top_n, "n_novels").sort_values("n_novels")

        fig = px.bar(
            subset, x="n_novels", y="genre", orientation="h",
            color="avg_weighted_rating", color_continuous_scale=SCALE,
            labels={"n_novels": "Number of titles", "genre": "", "avg_weighted_rating": "Rating"},
            hover_data={"pct_of_catalog": ":.1f", "avg_chapters": ":.0f"},
        )
        fig.update_coloraxes(colorbar_title_text="Rating")
        st.plotly_chart(style_fig(fig, max(360, 26 * top_n)), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Bar length = number of titles, colour = average rating. '
            "The most common genre is not necessarily the best-liked one — compare both.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Most frequent genre combinations**")
            combos = load_table("genre_combinations").head(12)
            st.dataframe(
                combos[["genre_combo", "n_novels", "avg_weighted_rating"]].rename(
                    columns={"genre_combo": "Combination", "n_novels": "Titles",
                             "avg_weighted_rating": "Rating"}),
                hide_index=True, use_container_width=True, height=430,
            )
        with col2:
            st.markdown("**Rating spread by genre**")
            top_genres = viz["primary_genre"].value_counts().head(8).index.tolist()
            box_df = viz[viz["primary_genre"].isin(top_genres) & viz["weighted_rating"].notna()]
            fig = px.box(
                box_df, x="primary_genre", y="weighted_rating", color="primary_genre",
                points=False, color_discrete_sequence=SEQUENCE,
                labels={"primary_genre": "", "weighted_rating": "Rating"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    # --- Length & rating ----------------------------------------------
    with tab_length:
        col_a, col_b = st.columns([3, 2])
        available = sorted(viz["primary_genre"].dropna().unique().tolist())
        default = load_table("genre_frequency").nlargest(5, "n_novels")["genre"].tolist()
        picked = col_a.multiselect("Genres", available, default=[g for g in default if g in available])
        only_reliable = col_b.checkbox("Only titles with an established score", value=True)

        scatter = viz.dropna(subset=["chapter_count", "weighted_rating"])
        if picked:
            scatter = scatter[scatter["primary_genre"].isin(picked)]
        if only_reliable:
            scatter = scatter[scatter["is_rated_reliable"] == True]  # noqa: E712

        if scatter.empty:
            st.info("No titles match these filters.")
        else:
            fig = px.scatter(
                scatter, x="chapter_count", y="weighted_rating", color="primary_genre",
                size="rating_votes", size_max=26, opacity=0.62, log_x=True,
                color_discrete_sequence=SEQUENCE, hover_name="title",
                hover_data={"rating_score": True, "rating_votes": True, "type_origin": True},
                labels={"chapter_count": "Chapter count (log scale)",
                        "weighted_rating": "Rating", "primary_genre": ""},
            )
            buckets = load_table("length_vs_rating")
            fig.add_scatter(
                x=buckets["avg_chapters"], y=buckets["avg_weighted_rating"],
                mode="lines+markers", name="Average per length bucket",
                line=dict(color="#FFB454", width=3, dash="dash"),
            )
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)

            corr = load_table("correlation_metrics")
            r = corr.loc[corr["pair"] == "chapter_count ~ weighted_rating", "pearson_r"].iloc[0]
            st.markdown(
                f'<div class="nl-note">The orange line is the average rating per length bucket. '
                f"The correlation is r = {r} — essentially flat, then falling past 1,000 chapters: "
                "stories that run too long tend to lose satisfied readers.</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Rating by length bucket**")
        buckets = load_table("length_vs_rating")
        fig = px.bar(
            buckets, x="length_bucket", y="avg_weighted_rating", color="n_novels",
            color_continuous_scale=SCALE,
            labels={"length_bucket": "", "avg_weighted_rating": "Rating", "n_novels": "Titles"},
        )
        fig.update_yaxes(range=[3.3, 3.65])
        fig.update_coloraxes(colorbar_title_text="Titles")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        st.write("")
        st.markdown("**Rating by release pace**")
        pace = load_table("release_pace_vs_rating")
        fig = px.bar(
            pace, x="release_pace", y="avg_weighted_rating", color="n_novels",
            color_continuous_scale=SCALE,
            labels={"release_pace": "", "avg_weighted_rating": "Rating", "n_novels": "Titles"},
        )
        fig.update_layout(xaxis_tickangle=-12)
        fig.update_coloraxes(colorbar_title_text="Titles")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        pace_r = load_table("correlation_metrics")
        pace_r = pace_r.loc[pace_r["pair"] == "log_days_per_chapter ~ weighted_rating", "pearson_r"]
        st.markdown(
            '<div class="nl-note">Release pace is parsed from the catalogue&apos;s '
            '"Every N Day(s)" field, which had never been used before. Its correlation '
            f"with rating (r = {pace_r.iloc[0] if len(pace_r) else '—'}) is about as strong as "
            "the chapter-count one — slow-updating titles score slightly lower.</div>",
            unsafe_allow_html=True,
        )

    # --- Story language --------------------------------------------------
    with tab_text:
        tokens = load_table("top_tokens")
        st.markdown("**Vocabulary that dominates the synopses**")
        col_wc, col_list = st.columns([3, 1], gap="large")
        with col_wc:
            st.image(render_wordcloud(tokens), use_column_width=True)
        with col_list:
            st.dataframe(
                tokens.head(14)[["token", "n_novels"]].rename(
                    columns={"token": "Word", "n_novels": "Titles"}),
                hide_index=True, use_container_width=True, height=420,
            )
        st.markdown(
            '<div class="nl-note">Taken from the original synopses (not the catalogue tags), '
            "after stopword removal.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Synopsis length vs rating**")
            richness = load_table("text_richness_vs_rating")
            fig = px.bar(
                richness, x="text_richness", y="avg_weighted_rating", color="n_novels",
                color_continuous_scale=SCALE,
                labels={"text_richness": "", "avg_weighted_rating": "Rating", "n_novels": "Titles"},
            )
            fig.update_yaxes(range=[3.35, 3.8])
            fig.update_layout(xaxis_tickangle=-18)
            fig.update_coloraxes(colorbar_title_text="Titles")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with col2:
            st.markdown("**Best-rated story elements**")
            tags = load_table("best_rated_tags").head(14).sort_values("avg_weighted_rating")
            fig = px.bar(
                tags, x="avg_weighted_rating", y="tag", orientation="h", color="n_novels",
                color_continuous_scale=SCALE,
                labels={"avg_weighted_rating": "Rating", "tag": "", "n_novels": "Titles"},
            )
            fig.update_xaxes(range=[3.4, 4.0])
            fig.update_coloraxes(colorbar_title_text="Titles")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    # --- Market map -----------------------------------------------------
    with tab_market:
        st.markdown("**Average rating by genre × country of origin**")
        matrix = load_table("genre_origin_matrix")
        pivot = matrix.pivot(index="genre", columns="type_origin", values="avg_weighted_rating")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(
            pivot, color_continuous_scale=SCALE, aspect="auto", text_auto=".2f",
            labels=dict(x="Country of origin", y="", color="Rating"),
        )
        st.plotly_chart(style_fig(fig, max(380, 30 * len(pivot))), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Empty cells = combinations with fewer than 25 titles, '
            "deliberately left uncomputed so no average rests on a handful of titles.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Trend by decade**")
            decade = load_table("decade_trend")
            fig = px.line(
                decade, x="decade", y="avg_weighted_rating", markers=True,
                labels={"decade": "", "avg_weighted_rating": "Rating"},
            )
            fig.update_traces(line=dict(color=BRAND, width=3), marker=dict(size=10, color=ACCENT))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        with col2:
            st.markdown("**Longest titles in the catalogue**")
            outliers = load_table("stat_outlier_novels").head(10)
            st.dataframe(
                outliers[["title", "chapter_count", "rating_score"]].rename(
                    columns={"title": "Title", "chapter_count": "Chapters", "rating_score": "Rating"}),
                hide_index=True, use_container_width=True, height=340,
            )


@st.cache_data(ttl=900, show_spinner=False)
def render_wordcloud(tokens: pd.DataFrame) -> bytes:
    """Render the word cloud on a transparent ground so it blends with the theme."""
    from wordcloud import WordCloud

    cloud = WordCloud(
        width=1400, height=560, mode="RGBA", background_color=None,
        colormap="cool", max_words=180, prefer_horizontal=0.92,
    ).generate_from_frequencies(dict(zip(tokens["token"], tokens["freq"])))

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# PAGE — AI Studio
# ---------------------------------------------------------------------------
SAMPLE_SYNOPSIS = (
    "After being betrayed by their allies and dying at the top of the tower, a "
    "swordsman wakes up in their own body ten years younger. This time they intend "
    "to master the forbidden sword techniques before their old enemy rises again."
)
SAMPLE_TAGS = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"


def page_studio() -> None:
    hero(
        "AI Studio",
        "Four assistants that work from story content: finding similar titles, "
        "tagging genres, estimating reader reception, and recognising narrative archetypes.",
        eyebrow="Predictive models",
    )

    if not artifacts_available():
        st.info("AI features are still being prepared. Try again once model training has finished.")
        return

    try:
        registry = get_registry()
    except ModelNotTrainedError as exc:
        st.info("AI features are not available yet.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return

    tab_rec, tab_genre, tab_rating, tab_arch = st.tabs(
        ["Recommendations", "Genre Tagging", "Rating Estimate", "Story Archetypes"]
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
    st.markdown("Find the titles whose content sits closest to one you already like.")
    mode = st.radio("Search by", ["Catalogue title", "Free-text description"], horizontal=True)
    k = st.slider("How many recommendations", 3, 12, 6)

    try:
        if mode == "Catalogue title":
            titles = registry.catalog.sort_values("rating_votes", ascending=False)["title"].tolist()
            choice = st.selectbox("Reference title", titles)
            series_id = int(registry.catalog.loc[registry.catalog["title"] == choice, "series_id"].iloc[0])
            results = registry.recommend_by_series(series_id, k=k)
        else:
            text = st.text_area("What kind of story are you looking for?", SAMPLE_SYNOPSIS, height=110)
            results = registry.recommend_by_text(text, k=k)
    except (ValueError, KeyError):
        st.warning("Not enough input to find a match. Add more story detail.")
        return

    st.write("")
    for _, row in results.iterrows():
        novel_card(row)


def _studio_genre(registry) -> None:
    st.markdown(
        "Paste a novel's synopsis and tags, then pick the model to use. "
        "Each model weighs words differently."
    )

    names = registry.genre_model_names
    info = {name: registry.genre_model_info(name) for name in names}

    choice = st.radio(
        "Prediction model",
        names,
        horizontal=True,
        format_func=lambda n: f"{info[n].get('label', n)} · F1 {info[n].get('f1_macro', 0):.2f}",
    )
    selected = info[choice]
    reference = selected.get("reference")
    st.markdown(
        f'<div class="nl-note"><b>{selected.get("family", "")}</b> — {selected.get("description", "")}'
        + (f'<br><span style="opacity:.75">Architecture follows {reference}.</span>' if reference else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        synopsis = st.text_area("Synopsis", SAMPLE_SYNOPSIS, height=150)
        tags = st.text_input("Tags (optional, comma-separated)", SAMPLE_TAGS)
    with col2:
        origin = st.selectbox("Country of origin", ["CN", "KR", "JP", "EN", "Other"])
        novel_type = st.selectbox("Format", ["Web Novel", "Light Novel", "Published Novel"])
        sensitivity = st.slider(
            "Strictness", 0.6, 1.6, 1.0, 0.05,
            help="Below 1 surfaces more genres; above 1 keeps only the most confident ones.",
        )

    if st.button("Predict genres", type="primary"):
        try:
            # The hybrid model loads its sentence encoder on first use — about
            # five seconds, once per session. Every later call is ~50 ms, so the
            # spinner only ever shows up on that first prediction.
            with st.spinner("Running the model…"):
                result = registry.predict_genres(
                    synopsis, tags=tags, model=choice, top_k=12,
                    type_origin=origin, novel_type=novel_type, sensitivity=sensitivity,
                )
        except ValueError:
            st.warning("Not enough text to analyse. Add a synopsis or some tags.")
            return

        predicted = result[result["predicted"]]["genre"].tolist()
        if predicted:
            st.markdown("**Predicted genres**")
            st.markdown(chips(predicted, active=True), unsafe_allow_html=True)
        else:
            st.info("No genre is confident enough. Lower the strictness or add more detail.")

        st.write("")
        plot = result.sort_values("probability")
        fig = px.bar(
            plot, x="probability", y="genre", orientation="h",
            color="predicted", color_discrete_map={True: ACCENT, False: "rgba(255,255,255,.18)"},
            labels={"probability": "Model confidence", "genre": "", "predicted": ""},
        )
        fig.add_scatter(
            x=plot["threshold"], y=plot["genre"], mode="markers",
            marker=dict(symbol="line-ns", size=16, line=dict(color="#FFB454", width=2)),
            name="Genre threshold",
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        st.markdown(
            '<div class="nl-note">The orange marks are each genre&apos;s own threshold. '
            "Rare genres get a lower one, so they can still be detected without "
            "flooding the results with common genres.</div>",
            unsafe_allow_html=True,
        )


def _studio_rating(registry) -> None:
    head_d = registry.metrics["head_d_rating_estimator"]
    if not head_d.get("trained"):
        st.info("Rating estimation is not available yet.")
        return

    st.markdown(
        "Estimate how readers will score a story — useful before a new title has "
        "any real ratings."
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        text = st.text_area("Synopsis or tags", SAMPLE_TAGS, height=130, key="rating_text")
    with col2:
        chapters = st.number_input("Planned chapter count", 1, 12000, 300, 10)
        year = st.number_input("Release year", 1990, 2030, 2024)
        origin = st.selectbox("Country of origin", ["CN", "KR", "JP", "EN", "Other"], key="rating_origin")
        pace_days = st.number_input(
            "Days between chapters", 0.5, 365.0, 7.0, 0.5,
            help="How often a new chapter lands. Slow-updating titles score slightly lower.",
        )
        completed = st.checkbox("Already completed")

    if st.button("Estimate rating", type="primary"):
        try:
            predicted = registry.estimate_rating(
                text, chapter_count=float(chapters), year=int(year),
                is_completed=completed, type_origin=origin,
                days_per_chapter=float(pace_days),
            )
        except (ValueError, ModelNotTrainedError):
            st.warning("Not enough input to produce an estimate.")
            return

        average = registry.catalog["rating_score"].mean()
        cols = st.columns(3)
        card(cols[0], "Estimated rating", f"{predicted:.2f}", "on a 1–5 scale")
        card(cols[1], "Vs catalogue", f"{predicted - average:+.2f}", f"average {average:.2f}")
        card(cols[2], "Error margin", f"±{head_d['mae']:.2f}", "measured on the held-out test set")

        st.markdown(
            '<div class="nl-note">This estimate reads story content only — not writing quality '
            "or marketing. Content explains a small share of rating variance, so treat the "
            "number as a direction, not a verdict.</div>",
            unsafe_allow_html=True,
        )


def _studio_archetype(registry) -> None:
    head_c = registry.metrics["head_c_clustering"]
    profiles = pd.DataFrame(head_c["profiles"])

    st.markdown(
        f"The catalogue splits into {head_c['best_k']} narrative archetypes. "
        "Paste a story to see which family it belongs to."
    )
    text = st.text_area("Synopsis or tags", SAMPLE_TAGS, height=110, key="arch_text")

    if st.button("Identify archetype", type="primary"):
        try:
            assignment = registry.assign_cluster(text)
        except ValueError:
            st.warning("Not enough text to identify an archetype.")
            return

        profile = assignment["profile"]
        with st.container(border=True):
            st.markdown(f"### Archetype #{assignment['cluster_id']}")
            st.markdown(chips(profile.get("top_terms", [])[:8], active=True), unsafe_allow_html=True)
            cols = st.columns(3)
            card(cols[0], "Titles in this family", f"{profile.get('size', 0):,}")
            card(cols[1], "Average rating", f"{profile.get('avg_weighted_rating', 0):.2f}")
            card(cols[2], "Average length", f"{profile.get('avg_chapters', 0):,.0f} chapters")
            st.write("")
            st.caption("Example titles: " + " · ".join(profile.get("example_titles", [])[:4]))

    st.write("")
    st.markdown("**All archetypes**")
    overview = profiles.assign(
        terms=profiles["top_terms"].apply(lambda t: ", ".join(t[:5])),
        genre=profiles["top_genres"].apply(lambda g: ", ".join(list(g.keys())[:3])),
    )[["cluster_id", "size", "avg_weighted_rating", "avg_chapters", "terms", "genre"]]
    st.dataframe(
        overview.rename(columns={
            "cluster_id": "#", "size": "Titles", "avg_weighted_rating": "Rating",
            "avg_chapters": "Chapters", "terms": "Signature terms", "genre": "Dominant genres"}),
        hide_index=True, use_container_width=True,
    )


# ---------------------------------------------------------------------------
# PAGE — Methodology
# ---------------------------------------------------------------------------
def page_method() -> None:
    hero(
        "Methodology & limitations",
        "How the numbers in this app are computed, how accurate the models are, "
        "and where the output should not be taken at face value.",
        eyebrow="Transparency",
    )

    st.subheader("Where the data comes from")
    st.markdown(
        """
The catalogue holds **4,461 translated web novel titles** with their metadata,
reader ratings, and synopses. Before use, the data goes through cleaning:

- Missing values disguised as the number `0` or the text `"Unknown"` are
  restored to genuinely empty — without this the average chapter count is off
  by roughly 20%.
- Ratings are shown as a **weighted rating**: a title with 2 votes is not
  treated like one with 500. Every ranking in this app uses only titles with at
  least 10 votes.
- Synopses are joined in from a separate source by URL matching, covering
  **98.45%** of the catalogue.
        """
    )

    if not artifacts_available():
        return

    metrics = get_registry().metrics
    head_a, head_b = metrics["head_a_genre_tagger"], metrics["head_b_recommender"]
    head_c, head_d = metrics["head_c_clustering"], metrics["head_d_rating_estimator"]

    st.subheader("Genre tagging model performance")
    st.markdown(
        "Every model is trained and tested on the same split "
        f"({head_a['n_train']:,} titles for development · {head_a['n_test']:,} for testing), "
        "so the comparison is fair. Hyperparameters and each genre's decision threshold "
        "are chosen by cross-validation inside the development set — the test set is "
        "touched once, at the very end, so the numbers below are not inflated."
    )

    rows = []
    for name, entry in head_a["models"].items():
        rows.append({
            "Model": entry.get("label", name),
            "Approach": entry.get("family", ""),
            "Macro F1": entry["f1_macro"],
            "Micro F1": entry["f1_micro"],
            "P@3": entry.get("precision_at_3"),
            "Hamming loss": entry.get("hamming_loss"),
            "Macro F1 without threshold tuning": entry["f1_macro_fixed_threshold"],
        })
    table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "Macro F1": st.column_config.ProgressColumn("Macro F1", min_value=0, max_value=1, format="%.3f"),
            "Micro F1": st.column_config.ProgressColumn("Micro F1", min_value=0, max_value=1, format="%.3f"),
            "P@3": st.column_config.NumberColumn("P@3", format="%.3f"),
            "Hamming loss": st.column_config.NumberColumn("Hamming loss", format="%.4f"),
        },
    )
    st.markdown(
        """
<div class="nl-note">
<b>Macro F1</b> weights every genre equally, so it drops sharply when rare genres are
ignored — this is the model selection metric. <b>Micro F1</b> is dominated by common
genres. <b>P@3</b> measures how accurate the top three guesses are without depending on a
threshold at all, which is closest to what users actually see. <b>Hamming loss</b> is the
share of binary decisions that are wrong (lower is better). The last column shows the same
model scored with a uniform 0.5 threshold — the gap is the contribution of per-genre
threshold tuning.
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**Architecture references.** The two feature-based models (TF-IDF + linear) follow the
baselines common in genre-classification-from-synopsis research. The sequence model
replicates a BiLSTM with self-attention (Ertugrul & Karagoz, 2018; Zhou et al., 2016),
with multi-label handling via weighted per-label *binary cross-entropy*
(Liu et al., SIGIR 2017). Per-genre probability threshold tuning follows Hoang (2018).
        """
    )

    st.write("")
    st.subheader("Other features")
    cols = st.columns(3)
    card(cols[0], "Recommendations", f"{head_b['genre_overlap_at_k']:.0%}",
         f"of {head_b['k']} suggestions share a genre with the reference title")
    card(cols[1], "Archetypes", f"{head_c['best_k']} clusters",
         "cluster count chosen automatically by silhouette score")
    card(cols[2], "Rating estimate", f"±{head_d['mae']:.2f}",
         f"{head_d['improvement_over_baseline_pct']:.0f}% better than guessing the average")

    st.write("")
    st.subheader("Limitations worth remembering")
    st.markdown(
        """
- The models read the **description**, not the manuscript. A novel with a poor
  synopsis will be judged poorly even if the story itself is good.
- Recommendations rest purely on content similarity; two titles on the same theme
  count as similar even when their writing quality differs sharply.
- The catalogue is dominated by Chinese novels (55%), so accuracy on English or
  Japanese titles tends to be lower.
- Ratings come from a single translated-fiction reader community, not the general
  population.
- Archetypes are overlapping groups, not hard partitions.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Catalogue Analytics": page_analytics,
    "AI Studio": page_studio,
    "Methodology": page_method,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .1rem">'
            '<div style="width:34px;height:34px;border-radius:10px;display:grid;place-items:center;'
            'background:linear-gradient(135deg,#7C5CFF,#22D3A7);font-size:1.05rem">📖</div>'
            '<div><div style="font-size:1.06rem;font-weight:680;letter-spacing:-.02em">NovelLens</div>'
            f'<div style="font-size:.72rem;color:{MUTED}">Novel catalogue intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.markdown(
            f'<div style="font-size:.75rem;color:{MUTED};line-height:1.6">'
            "<b>4,461</b> titles · NovelUpdates<br>"
            "Bayesian weighted rating<br>"
            "Models selected by cross-validation</div>",
            unsafe_allow_html=True,
        )

    online, detail = backend_online()
    if not online:
        st.error("The data service is unreachable right now. Please try again shortly.")
        with st.expander("Technical details"):
            st.code(detail)
        st.stop()

    try:
        PAGES[choice]()
    except DataUnavailable as exc:
        st.warning("The data for this page is not available yet.")
        with st.expander("Technical details"):
            st.code(str(exc))


main()
