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

import html
import io
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.common.config import get_settings
from src.common.model_registry import (
    ARTIFACT_DIR,
    ModelNotTrainedError,
    artifacts_available,
    load_registry,
)
from src.common.storage import get_s3_client, list_objects, read_parquet_dir
from src.common.title_generator import TitleGenerator, description_keywords

st.set_page_config(
    page_title="NovelLens: Novel Catalogue Intelligence",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="auto",
)

SETTINGS = get_settings()
#: NOVELLENS_PUBLIC=1 for a published deployment: operator pages (full dataset,
#: lake inventory) and raw error text, which names hosts and paths, stay hidden.
PUBLIC = os.environ.get("NOVELLENS_PUBLIC", "0") == "1"

# Visual identity
BRAND = "#7C5CFF"
ACCENT = "#22D3A7"
# Secondary text: #A3ABBF is 7.45:1 on the card surface (WCAG AA needs 4.5).
MUTED = "#A3ABBF"
# Focus ring: lighter than BRAND so it clears 3:1 against input surfaces (5.72:1).
FOCUS = "#9D86FF"
# Plus Jakarta Sans: an Indonesian typeface that stays legible at small label sizes.
# The system stack covers an unreachable Google Fonts.
FONT_STACK = ('"Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", '
              "Roboto, sans-serif")
SEQUENCE = ["#7C5CFF", "#22D3A7", "#FFB454", "#FF6B9D", "#4CC9F0", "#B388FF", "#FF8F6B", "#5EEAD4"]
SCALE = [[0.0, "#1E1B4B"], [0.45, "#5B4BD6"], [0.75, "#7C5CFF"], [1.0, "#22D3A7"]]

st.markdown(
    f"""
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400..800&display=swap");

      /* Streamlit sets Source Sans on p/div/span through its own classes, so a
         container rule loses. Icons (Material Symbols ligatures) and code keep
         their own fonts: overriding them turns icons into words. */
      html, body, .stApp,
      .stApp *:not([data-testid="stIconMaterial"]):not(code):not(code *):not(pre *) {{
        font-family: {FONT_STACK};
      }}

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
        padding: 1.6rem 1.9rem; border-radius: 14px; margin-bottom: 1.5rem;
        background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.10);
      }}
      .nl-hero h1 {{margin: 0;}}
      .nl-hero p {{margin: .5rem 0 0; color: {MUTED}; font-size: .97rem; max-width: 66ch; line-height: 1.55;}}

      /* --- Metric cards --------------------------------------------------- */
      .nl-card {{
        position: relative; height: 100%;
        padding: 1.05rem 1.15rem; border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
        border: 1px solid rgba(255,255,255,.085);
      }}
      .nl-card .lbl {{color: {MUTED}; font-size: .8rem; font-weight: 600; margin-bottom: .4rem;}}
      /* Solid colour first: browsers without background-clip:text, print/PDF,
         and forced-colours mode keep a readable number. */
      .nl-card .val {{font-size: 1.72rem; font-weight: 680; line-height: 1.1; color: #F1EDFF;}}
      @supports ((-webkit-background-clip: text) or (background-clip: text)) {{
        .nl-card .val {{
          background: linear-gradient(92deg, #FFFFFF, #C9BDFF);
          -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }}
      }}
      @media print, (forced-colors: active) {{
        .nl-card .val {{background: none; -webkit-text-fill-color: currentColor; color: CanvasText;}}
      }}
      .nl-card .sub {{color: {MUTED}; font-size: .8rem; font-weight: 500; margin-top: .35rem; line-height: 1.45;}}

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
      div[data-testid="stDataFrame"] {{border-radius: 10px; overflow: hidden;}}
      [data-testid="stMetricValue"] {{font-size: 1.42rem; font-weight: 640;}}
      [data-testid="stMetricLabel"] {{color: {MUTED};}}
      hr {{border-color: rgba(255,255,255,.08);}}

      /* --- Inputs ---------------------------------------------------------
         Same translucent surface and hairline border as the metric cards, so
         controls and the numbers they drive read as one system. No blur. */
      [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
        background: rgba(255,255,255,.035) !important;
        border: 1px solid rgba(255,255,255,.12) !important; border-radius: 10px !important;
        transition: border-color .15s ease, box-shadow .15s ease;
      }}
      [data-baseweb="input"]:hover, [data-baseweb="textarea"]:hover,
      [data-baseweb="select"] > div:hover {{border-color: rgba(255,255,255,.22) !important;}}
      /* Focus: a 2px ring in the lighter brand tone (5.72:1 against the input). */
      [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
      [data-baseweb="select"] > div:focus-within {{
        border-color: {FOCUS} !important; box-shadow: 0 0 0 2px {FOCUS}55 !important;
      }}
      .stButton > button:focus-visible, a.nl-open:focus-visible {{
        outline: 2px solid {FOCUS}; outline-offset: 2px;
      }}

      /* --- Recommendation card ------------------------------------------- */
      .nl-rec {{
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: .4rem 1.4rem;
        padding: 1.1rem 1.2rem; margin-bottom: .75rem; border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
        border: 1px solid rgba(255,255,255,.085);
      }}
      .nl-rec .title {{font-size: 1rem; font-weight: 680; line-height: 1.35; margin-bottom: .35rem;}}
      .nl-rec .nl-syn {{
        margin-top: .55rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
        overflow: hidden; min-height: calc(3 * 1.6em);
      }}
      .nl-rec .stats {{display: flex; flex-direction: column; align-items: flex-end; gap: .7rem; text-align: right;}}
      .nl-rec .num {{font-size: 1.45rem; font-weight: 720; line-height: 1;}}
      .nl-rec .num-lbl {{color: {MUTED}; font-size: .74rem; font-weight: 600; margin-top: .2rem;}}
      .nl-rec .meta {{
        grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center;
        gap: .8rem; padding-top: .7rem; margin-top: .25rem; border-top: 1px solid rgba(255,255,255,.07);
        color: {MUTED}; font-size: .8rem; font-weight: 500;
      }}
      a.nl-open {{
        flex: none; padding: .45rem .95rem; min-height: 44px; display: inline-flex; align-items: center;
        border-radius: 10px; border: 1px solid rgba(255,255,255,.16); color: #E6E9EF !important;
        text-decoration: none !important; font-weight: 600; font-size: .85rem;
      }}
      a.nl-open:hover {{border-color: {FOCUS};}}
      .nl-rec.compact .stats {{flex-direction: row; gap: 1.6rem; align-items: flex-start;}}
      @media (max-width: 640px) {{
        .nl-rec {{grid-template-columns: 1fr;}}
        .nl-rec .stats {{flex-direction: row; justify-content: flex-start; align-items: baseline;
                        gap: 1.6rem; text-align: left;}}
      }}

      /* --- Sidebar -------------------------------------------------------- */
      section[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.07);}}
      section[data-testid="stSidebar"] .stRadio label {{font-size: .92rem;}}
      .nl-brand {{padding: .2rem .2rem 1rem; margin-bottom: .4rem; border-bottom: 1px solid rgba(255,255,255,.08);}}
      .nl-brand .name {{font-size: 1.15rem; font-weight: 720; letter-spacing: -.015em; color: #F1EDFF;}}
      .nl-brand .sub {{font-size: .78rem; color: {MUTED}; margin-top: .15rem;}}
      /* Section groups: a plain header row, no card. */
      section[data-testid="stSidebar"] [data-testid="stExpander"] details {{
        border: none; background: transparent; border-radius: 0;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        padding: .45rem .25rem; border-radius: 8px;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        font-size: .74rem; font-weight: 700; color: {MUTED}; letter-spacing: .02em;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p {{color: #E6E9EF;}}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible {{
        outline: 2px solid {FOCUS}; outline-offset: 1px;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{padding: 0 0 .35rem .1rem;}}
      section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {{gap: .15rem;}}
      /* Folding animation: the links slide up into the header and back down.
         ::details-content + interpolate-size animate height:auto (Chrome 131+);
         older browsers simply open and close without motion. */
      section[data-testid="stSidebar"] {{interpolate-size: allow-keywords;}}
      section[data-testid="stSidebar"] details::details-content {{
        height: 0; opacity: 0; overflow: clip;
        transition: height .26s ease, opacity .2s ease, content-visibility .26s allow-discrete;
      }}
      section[data-testid="stSidebar"] details[open]::details-content {{height: auto; opacity: 1;}}
      @media (prefers-reduced-motion: reduce) {{
        section[data-testid="stSidebar"] details::details-content {{transition: none;}}
      }}
      /* Streamlit paints the current page's link itself; only the overview's card look is undone. */
      section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
        border-color: transparent; min-height: 36px; padding: .3rem .55rem;
      }}
      .nl-status {{display: flex; align-items: center; gap: .45rem; margin-top: 1rem; padding-top: .8rem;
                  border-top: 1px solid rgba(255,255,255,.08);
                  font-size: .75rem; color: {MUTED}; font-weight: 500;}}
      .nl-status .dot {{width: 8px; height: 8px; border-radius: 50%; flex: none;}}

      /* --- Change against a reference (KPI cards) --------------------------
         Arrow + sign carry the direction; colour only repeats it. */
      .nl-card .delta {{display: inline-block; margin-top: .4rem; font-size: .78rem; font-weight: 640;}}
      .nl-card .delta.up {{color: #5EEAD4;}}
      .nl-card .delta.down {{color: #FF9BBB;}}
      .nl-card .delta.flat {{color: {MUTED};}}

      /* --- Title ideas ----------------------------------------------------- */
      .nl-idea {{
        padding: .95rem 1.1rem; margin-bottom: .7rem; border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
        border: 1px solid rgba(255,255,255,.085);
      }}
      .nl-idea .rank {{color: {MUTED}; font-size: .74rem; font-weight: 640; margin-bottom: .25rem;}}
      .nl-idea .t {{font-size: 1.06rem; font-weight: 700; line-height: 1.35;}}
      .nl-idea .src {{color: {MUTED}; font-size: .78rem; line-height: 1.5; margin-top: .45rem;}}

      /* --- Ranked list (overview) --------------------------------------- */
      .nl-rank {{list-style: none; margin: 0; padding: 0;}}
      .nl-rank li {{display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) auto 5.6rem; gap: .7rem;
                   align-items: baseline; padding: .62rem .2rem; border-bottom: 1px solid rgba(255,255,255,.07);}}
      .nl-rank .r {{color: {MUTED}; font-variant-numeric: tabular-nums;}}
      .nl-rank .t {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 560;}}
      .nl-rank .v {{font-weight: 700; font-variant-numeric: tabular-nums;}}
      .nl-rank .n {{color: {MUTED}; font-size: .8rem; text-align: right; font-variant-numeric: tabular-nums;}}

      /* --- Genre confidence bars (AI Studio) ------------------------------ */
      .nl-bars {{display: grid; gap: .45rem; margin-top: .3rem;}}
      .nl-bar {{display: grid; grid-template-columns: 8.5rem minmax(0, 1fr) 2.6rem 4.8rem; gap: .7rem;
               align-items: center; font-size: .85rem;}}
      .nl-bar .g {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
      .nl-bar .track {{position: relative; height: 10px; border-radius: 5px; background: rgba(255,255,255,.07);}}
      .nl-bar .fill {{position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: rgba(163,171,191,.45);}}
      .nl-bar.on .fill {{background: {ACCENT};}}
      .nl-bar .tick {{position: absolute; top: -3px; width: 2px; height: 16px; background: #FFB454;}}
      .nl-bar .p {{text-align: right; font-variant-numeric: tabular-nums; color: #E6E9EF;}}
      .nl-bar .s {{color: {MUTED}; font-size: .76rem;}}
      .nl-bar.on .g, .nl-bar.on .s {{color: #8FF0D6; font-weight: 600;}}
      @media (max-width: 640px) {{.nl-bar {{grid-template-columns: 6.5rem minmax(0, 1fr) 2.4rem;}} .nl-bar .s {{display: none;}}}}

      /* --- Page links on the overview ------------------------------------- */
      [data-testid="stMain"] [data-testid="stPageLink"] a {{
        width: 100%; box-sizing: border-box;
        border: 1px solid rgba(255,255,255,.10); border-radius: 10px; padding: .55rem .8rem;
        background: rgba(255,255,255,.025); min-height: 44px;
      }}
      [data-testid="stMain"] [data-testid="stPageLink"] a:hover {{border-color: rgba(124,92,255,.5);}}
      [data-testid="stPageLink"] a:focus-visible {{outline: 2px solid {FOCUS}; outline-offset: 2px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# Data layer
class DataUnavailable(RuntimeError):
    """Product data is not ready to serve yet (the pipeline has not run)."""


@st.cache_data(ttl=60, show_spinner=False)
def lake_version() -> str:
    """
    Version stamp of the curated zone: the newest object's modification time.

    Every lake loader takes this stamp as a cache key, so all tables switch to
    a new pipeline run together (within a minute) instead of each one expiring
    on its own 15-minute clock and a page mixing old and new tables.
    """
    try:
        objects = list_objects(SETTINGS.minio.bucket_curated, "")
    except Exception:
        return ""
    return str(max((o["LastModified"] for o in objects), default=""))


@st.cache_data(show_spinner=False, max_entries=64)
def _load_lake(prefix: str, version: str) -> pd.DataFrame:
    try:
        return read_parquet_dir(SETTINGS.minio.bucket_curated, prefix)
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc


def load_lake(prefix: str) -> pd.DataFrame:
    """Read one table from the curated zone (`analytics/`, `model/`, `governance/`)."""
    return _load_lake(prefix, lake_version())


def load_table(name: str) -> pd.DataFrame:
    """Read one analytics result table from the data lake."""
    return load_lake(f"analytics/{name}")


@st.cache_data(show_spinner=False, max_entries=64)
def _table_profile(prefix: str, group_col: str | None, version: str) -> tuple[int, dict]:
    frame = _load_lake(prefix, version)
    return len(frame), (frame[group_col].value_counts().to_dict() if group_col else {})


def table_profile(prefix: str, group_col: str | None = None) -> tuple[int, dict]:
    """Row count (and optional value counts) of a lake table without copying the table on every rerun."""
    return _table_profile(prefix, group_col, lake_version())


@st.cache_data(ttl=120, show_spinner=False)
def _probe_backend() -> bool:
    # Raises on failure, so only a healthy answer is cached: an outage does not
    # keep the dashboard locked for two minutes after storage comes back.
    get_s3_client(SETTINGS).list_buckets()
    return True


def backend_online() -> tuple[bool, str]:
    """Check storage availability without leaking its details into the UI."""
    try:
        return _probe_backend(), ""
    except Exception as exc:
        return False, str(exc)


def _artifact_stamp() -> int:
    # metrics.json is written last by Phase 4, so its mtime marks a complete set.
    path = ARTIFACT_DIR / "metrics.json"
    return path.stat().st_mtime_ns if path.is_file() else 0


@st.cache_resource(show_spinner=False, max_entries=1)
def _registry(stamp: int):
    return load_registry()


def get_registry():
    """Model registry, reloaded when Phase 4 writes a new set of artefacts."""
    return _registry(_artifact_stamp())


@st.cache_data(show_spinner=False, max_entries=8)
def _load_labels(bridge: str, dim: str, key: str, label: str, version: str) -> pd.DataFrame:
    """A bridge table resolved to (series_id, label); labels as categories to keep 600k rows small."""
    pairs = _load_lake(f"model/{bridge}", version)
    names = _load_lake(f"model/{dim}", version)[[key, label]]
    out = pairs.merge(names, on=key)[["series_id", label]]
    out[label] = out[label].astype("category")
    return out


def genre_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", lake_version())


def tag_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_tag", "dim_tag", "tag_key", "tag", lake_version())


def author_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_author", "dim_author", "author_key", "author", lake_version())


@st.cache_data(show_spinner=False, max_entries=2)
def _genre_lists(version: str) -> pd.Series:
    pairs = _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", version)
    return pairs.assign(genre=pairs["genre"].astype(str)).groupby("series_id")["genre"].agg(list)


def genre_lists() -> pd.Series:
    """series_id -> list of genres, for cards that show genre chips."""
    return _genre_lists(lake_version())


@st.cache_resource(show_spinner=False, max_entries=1)
def _title_generator(version: str) -> TitleGenerator:
    viz = _load_lake("analytics/viz_novels", version)[["series_id", "title", "type_origin"]]
    return TitleGenerator(viz.assign(genre_list=viz["series_id"].map(_genre_lists(version))))


def title_generator() -> TitleGenerator:
    """Title pattern model over the whole lake catalogue, rebuilt when the lake changes (~3 s)."""
    return _title_generator(lake_version())


def last_full_year(years: pd.Series) -> int:
    """Latest release year that is over: the current year is still filling up."""
    return int(min(years.max(), pd.Timestamp.now(tz="UTC").year - 1))


#: Fixed colour per origin so a filter never repaints the survivors.
ORIGIN_COLORS = {"CN": BRAND, "KR": ACCENT, "JP": "#FFB454", "EN": "#4CC9F0", "Other": "#8A90A6"}


def origin_group(origins: pd.Series) -> pd.Series:
    return origins.where(origins.isin(["CN", "KR", "JP", "EN"]), "Other")


# UI components
def hero(title: str, subtitle: str) -> None:
    """Page header: title and one explanatory sentence."""
    st.markdown(
        f'<div class="nl-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(container, label: str, value: str, sub: str = "", delta: tuple[float, str] | None = None) -> None:
    """Metric card; `delta` is (change, text): the sign picks the arrow, the text names the baseline."""
    change = ""
    if delta is not None and pd.notna(delta[0]):
        cls, arrow = ("up", "▲") if delta[0] > 0 else ("down", "▼") if delta[0] < 0 else ("flat", "■")
        change = f'<div class="delta {cls}">{arrow} {delta[1]}</div>'
    container.markdown(
        f'<div class="nl-card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>{change}'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def chips(values, active: bool = False) -> str:
    cls = "nl-chip on" if active else "nl-chip"
    return "".join(f'<span class="{cls}">{html.escape(str(v))}</span>' for v in values)


def style_fig(fig, height: int | None = None):
    """Match every chart to the app theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D5DAE5", size=13, family=FONT_STACK),
        # 4px rounded data ends, matching the card radius language; plotly only
        # rounds the outer end of a stacked bar, so segments still meet flush.
        barcornerradius=4,
        margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor="#161B26", bordercolor="rgba(255,255,255,.15)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    if height:
        fig.update_layout(height=height)
    return fig


#: Bucket labels come out of Spark in Indonesian ("3. Panjang (200-499)").
BUCKET_WORDS = [
    ("Sangat Panjang", "Very long"), ("Sangat panjang", "Very long"), ("Sangat ringkas", "Very brief"),
    ("Pendek", "Short"), ("Sedang", "Medium"), ("Panjang", "Long"), ("Epik", "Epic"),
    ("Ringkas", "Brief"), ("Harian", "Daily"), ("Mingguan", "Weekly"), ("Bulanan", "Monthly"),
    ("Lambat", "Slow"), ("Terhenti", "Stalled"), (" kata", " words"), (" hari", " days"),
    (" bulan", " months"), ("<=", "≤"),
]


def en_label(value) -> str:
    """English display label for a pipeline bucket; the order prefix is dropped."""
    text = re.sub(r"^\d+\.\s*", "", str(value))
    for old, new in BUCKET_WORDS:
        text = text.replace(old, new)
    return text


def dot_rating_chart(frame: pd.DataFrame, category: str, horizontal: bool = False, height: int = 340,
                     translate: bool = True):
    """Average rating per group as dots: the means differ by tenths, and a bar on a
    cut axis would overstate the gap where a dot does not."""
    plot = frame.assign(label=frame[category].map(en_label) if translate else frame[category])
    xy = dict(x="avg_weighted_rating", y="label") if horizontal else dict(x="label", y="avg_weighted_rating")
    fig = px.scatter(
        plot, **xy, color="n_novels", color_continuous_scale=SCALE,
        hover_data={"n_novels": ":,", "label": False},
        labels={"label": "", "avg_weighted_rating": "Average rating", "n_novels": "Titles"},
    )
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color="#0D1117")))
    lo, hi = plot["avg_weighted_rating"].min(), plot["avg_weighted_rating"].max()
    pad = max(hi - lo, 0.1) * 0.35
    (fig.update_xaxes if horizontal else fig.update_yaxes)(range=[lo - pad, hi + pad])
    fig.update_coloraxes(colorbar_title_text="Titles")
    return style_fig(fig, height)


def rating_text(values: pd.Series, empty: str = "no rated titles") -> pd.Series:
    """Averages as fixed two-decimal text; Streamlit 1.39 prints NaN in number columns as None."""
    return values.map(lambda v: f"{v:.2f}" if pd.notna(v) else empty)


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value]
    return [str(value)]


def novel_card(row: pd.Series, show_similarity: bool = True, similarity_label: str = "match",
               show_synopsis: bool = True) -> None:
    """
    Recommendation card: identity on the left (title, genres, synopsis), the two
    numbers a reader compares across cards on the right, metadata and the link
    in a footer row. One HTML block so every card has the same structure and
    the synopsis clamps to three lines, which keeps the cards an even height.
    """
    title = html.escape(str(row.get("title") or "Untitled"))
    snippet = row.get("synopsis_snippet")
    synopsis = (
        "" if not show_synopsis
        else f'<div class="nl-syn">{html.escape(snippet.strip())}</div>'
        if isinstance(snippet, str) and snippet.strip()
        else '<div class="nl-syn">No synopsis available.</div>'
    )

    rating, votes = row.get("rating_score"), row.get("rating_votes")
    stats = [
        f'<div><div class="num">{rating:.1f}</div><div class="num-lbl">'
        f'{"rating" if pd.notna(votes) and votes >= 10 else "rating, few votes"}</div></div>'
        if pd.notna(rating) else
        '<div><div class="num">n/a</div><div class="num-lbl">not rated yet</div></div>'
    ]
    if show_similarity and pd.notna(row.get("similarity")):
        stats.append(
            f'<div><div class="num">{row["similarity"]:.0%}</div>'
            f'<div class="num-lbl">{similarity_label}</div></div>'
        )

    meta = []
    if pd.notna(row.get("chapter_count")):
        meta.append(f"{int(row['chapter_count']):,} chapters")
    if pd.notna(votes):
        meta.append(f"{int(votes):,} vote{'' if int(votes) == 1 else 's'}")
    if isinstance(row.get("type_origin"), str):
        meta.append(html.escape(row["type_origin"]))
    url = row.get("url")
    link = (
        f'<a class="nl-open" href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">Open on NovelUpdates</a>'
        if isinstance(url, str) and url.startswith("http") else ""
    )

    st.markdown(
        f'<div class="nl-rec{"" if show_synopsis else " compact"}"><div><div class="title">{title}</div>'
        f'{chips(as_list(row.get("genre_list"))[:5])}{synopsis}</div>'
        f'<div class="stats">{"".join(stats)}</div>'
        f'<div class="meta"><span>{" · ".join(meta)}</span>{link}</div></div>',
        unsafe_allow_html=True,
    )


# PAGE — Overview
def page_overview() -> None:
    viz = load_table("viz_novels")
    hero(
        "NovelLens",
        f"Catalogue intelligence for {len(viz):,} web novel titles: what people read, "
        "what they rate highly, and which titles are worth recommending.",
    )

    reliable = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    genres = load_table("genre_frequency")
    ranking = load_table("genre_ranking").sort_values("avg_weighted_rating", ascending=False)

    cols = st.columns(4)
    card(cols[0], "Titles in catalogue", f"{len(viz):,}", f"{viz['primary_genre'].nunique()} primary genres")
    card(cols[1], "Average rating", f"{viz['weighted_rating'].mean():.2f}",
         f"{len(reliable):,} titles with an established score")
    card(cols[2], "Median length", f"{viz['chapter_count'].median():,.0f} chapters",
         f"longest {viz['chapter_count'].max():,.0f} chapters")
    top_freq = genres.nlargest(1, "n_novels").iloc[0]
    card(cols[3], "Top genre", top_freq["genre"],
         f"{top_freq['pct_of_catalog']:.0f}% of the catalogue")

    st.write("")
    st.subheader("What you need to know")

    corr = load_table("correlation_metrics").set_index("pair")
    richness = load_table("text_richness_vs_rating").sort_values("text_richness")
    top_genre, longest = ranking.iloc[0], ranking.sort_values("avg_chapters", ascending=False).iloc[0]
    r_len = corr.loc["chapter_count ~ weighted_rating", "pearson_r"]
    gain = richness.iloc[-1]["avg_weighted_rating"] - richness.iloc[0]["avg_weighted_rating"]

    strength = "barely" if abs(r_len) < 0.05 else "slightly" if abs(r_len) < 0.2 else "clearly"
    direction = "negatively" if r_len < 0 else "positively"
    insight_cols = st.columns(3)
    card(
        insight_cols[0], "Length ≠ acclaim" if r_len < 0.2 else "Longer rates higher", f"r = {r_len}",
        f"Chapter count correlates {strength} {direction} with rating."
        + (" Adding more chapters does not buy reader approval." if r_len < 0.2 else ""),
    )
    card(
        insight_cols[1], "Best-rated genre", top_genre["genre"],
        f"Averages {top_genre['avg_weighted_rating']:.2f} across {int(top_genre['n_novels'])} titles. "
        f"Longest: {longest['genre']} (~{longest['avg_chapters']:,.0f} chapters).",
    )
    card(
        insight_cols[2], "Rich synopses win" if gain > 0 else "Synopsis length barely matters",
        f"{gain:+.2f} points",
        f"Titles with 300+ word synopses score {'higher' if gain > 0 else 'no higher'} than those "
        "under 50 words.",
    )

    st.write("")
    st.subheader("Go deeper")
    links = st.columns(5)
    for col, (key, text) in zip(links, [
        ("trends", "Trends by year"), ("genre", "One genre in depth"),
        ("tags", "Story element gaps"), ("lookup", "Look up a title"),
        ("studio", "Analyse a story"),
    ]):
        with col:
            st.page_link(PAGE[key], label=text)

    st.write("")
    left, right = st.columns([1, 1], gap="large")

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
        top = reliable.nlargest(8, "weighted_rating")
        items = "".join(
            f'<li><span class="r">{i}</span><span class="t" title="{html.escape(str(r.title), quote=True)}">'
            f'{html.escape(str(r.title))}</span><span class="v">{r.weighted_rating:.2f}</span>'
            f'<span class="n">{int(r.rating_votes):,} votes</span></li>'
            for i, r in enumerate(top.itertuples(index=False), 1)
        )
        st.markdown(f'<ol class="nl-rank">{items}</ol>', unsafe_allow_html=True)
        st.caption("Weighted rating, titles with 10+ votes.")


# PAGE — Catalogue Analytics
def page_analytics() -> None:
    hero(
        "Catalogue analytics",
        "The structure of the translated-novel market: genre mix, how story length "
        "relates to reader reception, and the vocabulary that dominates the stories.",
    )

    viz = load_table("viz_novels")
    tab_genre, tab_length, tab_text, tab_market = st.tabs(
        ["Genres", "Length & Rating", "Story Language", "Market Map"]
    )

    # Genres
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
            "The most common genre is not necessarily the best-liked one, so compare both.</div>",
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
                column_config={"Rating": st.column_config.NumberColumn(format="%.2f")},
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

    # Length & rating
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
            st.info("No titles match these filters. Pick more genres or include titles without an established score.")
        else:
            # render_mode="webgl": 20k+ points stay responsive to hover and zoom.
            fig = px.scatter(
                scatter, x="chapter_count", y="weighted_rating", color="primary_genre",
                size="rating_votes", size_max=26, opacity=0.62, log_x=True, render_mode="webgl",
                color_discrete_sequence=SEQUENCE, hover_name="title",
                hover_data={"rating_score": True, "rating_votes": True, "type_origin": True},
                labels={"chapter_count": "Chapter count (log scale)",
                        "weighted_rating": "Rating", "primary_genre": ""},
            )
            buckets = load_table("length_vs_rating").sort_values("length_bucket")
            fig.add_scatter(
                x=buckets["avg_chapters"], y=buckets["avg_weighted_rating"],
                mode="lines+markers", name="Average per length bucket",
                line=dict(color="#FFB454", width=3, dash="dash"),
            )
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
            st.caption(f"{len(scatter):,} titles plotted.")

            corr = load_table("correlation_metrics")
            r = corr.loc[corr["pair"] == "chapter_count ~ weighted_rating", "pearson_r"].iloc[0]
            best = buckets.loc[buckets["avg_weighted_rating"].idxmax()]
            worst = buckets.loc[buckets["avg_weighted_rating"].idxmin()]
            st.markdown(
                f'<div class="nl-note">The orange line is the average rating per length bucket. '
                f"The correlation is r = {r}. Ratings are highest for "
                f"{en_label(best['length_bucket']).lower()} chapters ({best['avg_weighted_rating']:.2f}) and "
                f"lowest for {en_label(worst['length_bucket']).lower()} ({worst['avg_weighted_rating']:.2f})"
                + (": stories that run long tend to lose satisfied readers." if worst["avg_chapters"]
                   > best["avg_chapters"] else ".")
                + "</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        rate_col, pace_col = st.columns(2, gap="large")
        with rate_col:
            st.markdown("**Rating by length bucket**")
            buckets = load_table("length_vs_rating").sort_values("length_bucket")
            st.plotly_chart(dot_rating_chart(buckets, "length_bucket"), use_container_width=True)
        with pace_col:
            st.markdown("**Rating by release pace**")
            pace = load_table("release_pace_vs_rating").sort_values("release_pace")
            st.plotly_chart(dot_rating_chart(pace, "release_pace"), use_container_width=True)
        pace_r = load_table("correlation_metrics")
        pace_r = pace_r.loc[pace_r["pair"] == "log_days_per_chapter ~ weighted_rating", "pearson_r"]
        st.markdown(
            '<div class="nl-note">Release pace is parsed from the catalogue&apos;s '
            '"Every N Day(s)" field, which had never been used before. Its correlation '
            f"with rating (r = {pace_r.iloc[0] if len(pace_r) else 'n/a'}) is about as strong as "
            "the chapter-count one: slow-updating titles score slightly lower.</div>",
            unsafe_allow_html=True,
        )

    # Story language
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
            richness = load_table("text_richness_vs_rating").sort_values("text_richness")
            fig = dot_rating_chart(richness, "text_richness", height=380)
            fig.update_layout(xaxis_tickangle=-18)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Best-rated story elements**")
            tags = load_table("best_rated_tags").head(14).sort_values("avg_weighted_rating")
            st.plotly_chart(dot_rating_chart(tags, "tag", horizontal=True, height=380, translate=False),
                            use_container_width=True)

    # Market map
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
            st.markdown("**Average rating by decade**")
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


# PAGE — AI Studio
SAMPLE_SYNOPSIS = (
    "After being betrayed by their allies and dying at the top of the tower, a "
    "swordsman wakes up in their own body ten years younger. This time they intend "
    "to master the forbidden sword techniques before their old enemy rises again."
)
SAMPLE_TAGS = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"


def page_studio() -> None:
    hero(
        "AI Studio",
        "Describe a story once and get its likely genres, reader reception, narrative family, "
        "closest titles in the catalogue and working-title ideas, all from the same brief.",
    )
    if not artifacts_available():
        st.info("AI features are still being prepared. They appear once model training has finished.")
        return
    try:
        registry = get_registry()
    except ModelNotTrainedError as exc:
        st.info("AI features are not available yet. They appear once the models finish training.")
        technical_details(str(exc))
        return

    tab_story, tab_similar = st.tabs(["Analyse a story", "Similar to a catalogue title"])
    with tab_story:
        _studio_story(registry)
    with tab_similar:
        _studio_similar(registry)


TITLE_STYLES = {
    "Any": "Any origin", "CN": "Chinese web novel", "KR": "Korean web novel",
    "JP": "Japanese light novel", "EN": "English original",
}

#: English copy for each genre model; metrics.json carries the pipeline's
#: Indonesian wording, which stays there for the written report.
MODEL_TEXT = {
    "logreg": ("Linear · sparse TF-IDF",
               "A linear model over TF-IDF features. The fastest, with the best-calibrated probabilities."),
    "svm": ("Margin-based · sparse TF-IDF",
            "Maximises the margin between classes; probabilities come from Platt calibration."),
    "hybrid": ("Sparse + dense · pretrained encoder",
               "TF-IDF plus pretrained sentence embeddings (BGE): phrases typical of a genre and "
               "paraphrases that use different words."),
    "bilstm": ("Deep learning · bidirectional recurrent",
               "Reads the synopsis as a sequence in both directions; attention weights the words that decide the genre."),
}


def model_text(name: str, info: dict) -> tuple[str, str]:
    """(family, description) in English, falling back to the stored label for a new model."""
    return MODEL_TEXT.get(name, (info.get("label", name), ""))


def reference_options(registry) -> tuple[list[int], dict[int, str]]:
    """Selectbox options keyed by series_id; duplicate titles get their year so both stay pickable."""
    catalog = registry.catalog.sort_values("rating_votes", ascending=False)
    duplicated = catalog["title"].duplicated(keep=False)
    year = pd.to_numeric(catalog["year"], errors="coerce")
    suffix = np.where(duplicated & year.notna(), " (" + year.fillna(0).astype(int).astype(str) + ")", "")
    labels = dict(zip(catalog["series_id"].astype(int), catalog["title"] + suffix))
    return list(labels), labels


def trained_origins(registry) -> list[str]:
    """Origins the rating model has a feature for, most common first."""
    learned = {
        name[len("origin_"):] for name in registry.metrics.get("numeric_features", [])
        if name.startswith("origin_")
    }
    ranked = registry.catalog["type_origin"].value_counts().index.tolist()
    ordered = [o for o in ranked if o in learned] + sorted(learned - set(ranked))
    return ordered or ["CN", "KR", "JP"]


def genre_bars_html(result: pd.DataFrame) -> str:
    """Confidence per genre as bars; the tick is that genre's own threshold, the label says predicted or not."""
    rows = []
    for r in result.itertuples(index=False):
        pct, tick = min(r.probability, 1) * 100, min(r.threshold, 1) * 100
        rows.append(
            f'<div class="nl-bar{" on" if r.predicted else ""}">'
            f'<span class="g">{html.escape(str(r.genre))}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.1f}%"></span>'
            f'<span class="tick" style="left:{tick:.1f}%"></span></span>'
            f'<span class="p">{pct:.0f}%</span>'
            f'<span class="s">{"predicted" if r.predicted else ""}</span></div>'
        )
    return '<div class="nl-bars">' + "".join(rows) + "</div>"


def _studio_story(registry) -> None:
    names = registry.genre_model_names
    info = {name: registry.genre_model_info(name) for name in names}
    head_d = registry.metrics.get("head_d_rating_estimator", {})
    origins = trained_origins(registry)

    with st.form("story_brief", border=True):
        st.markdown("**Story brief**")
        left, right = st.columns([3, 2], gap="large")
        with left:
            synopsis = st.text_area("Story idea or synopsis", SAMPLE_SYNOPSIS, height=168)
            tags = st.text_input("Tags (optional, comma-separated)", SAMPLE_TAGS)
        with right:
            a, b = st.columns(2)
            origin = a.selectbox("Country of origin", origins)
            novel_type = b.selectbox("Format", ["Web Novel", "Light Novel", "Published Novel"])
            a, b = st.columns(2)
            chapters = a.number_input("Planned chapters", 1, 12000, 300, 10)
            year = b.number_input("Release year", 1990, 2030, 2025)
            a, b = st.columns(2)
            pace = a.number_input("Days between chapters", 0.5, 365.0, 7.0, 0.5, format="%.1f")
            status = b.selectbox("Status", ["Ongoing", "Completed"])
            completed = status == "Completed"
        with st.expander("Model settings"):
            m1, m2 = st.columns([3, 2])
            choice = m1.radio(
                "Genre model", names, horizontal=True,
                format_func=lambda n: f"{info[n].get('label', n)} · F1 {info[n].get('f1_macro', 0):.2f}",
            )
            sensitivity = m2.slider(
                "Genre strictness", 0.6, 1.6, 1.0, 0.05,
                help="Below 1 surfaces more genres; above 1 keeps only the most confident ones.",
            )
            family, description = model_text(choice, info[choice])
            st.caption(f"{family}: {description}")
        submitted = st.form_submit_button("Analyse story", type="primary")

    if submitted:
        text = f"{synopsis} {tags}".strip()
        try:
            # The hybrid model loads its sentence encoder on first use (about five
            # seconds, once per session); later calls take ~50 ms.
            with st.spinner("Reading the story…"):
                genres = registry.predict_genres(
                    synopsis, tags=tags, model=choice, top_k=10,
                    type_origin=origin, novel_type=novel_type, sensitivity=sensitivity,
                )
                similar = registry.recommend_by_text(text, k=12)
                archetype = registry.assign_cluster(text)
                rating = registry.estimate_rating(
                    synopsis, chapter_count=float(chapters), year=int(year), is_completed=completed,
                    type_origin=origin, novel_type=novel_type, days_per_chapter=float(pace), tags=tags,
                ) if head_d.get("trained") else None
        except ValueError:
            st.warning("Not enough story detail to work from. Add a sentence or two about the plot.")
            return
        st.session_state["story"] = {
            "synopsis": synopsis, "tags": tags, "origin": origin, "genres": genres, "similar": similar,
            "archetype": archetype, "rating": rating, "model": choice,
        }
        st.session_state.pop("title_ideas", None)
        st.session_state["title_seen"] = set()

    story = st.session_state.get("story")
    if story is None:
        st.caption("Fill in the brief and press Analyse story. Only the story idea is required.")
        return
    _story_report(registry, story, info)


def _story_report(registry, story: dict, info: dict) -> None:
    genres, archetype = story["genres"], story["archetype"]
    predicted = genres[genres["predicted"]]["genre"].tolist()
    profile = archetype["profile"]
    catalog = registry.catalog
    average = catalog.loc[catalog["rating_votes"] >= 10, "rating_score"].mean()
    head_d = registry.metrics.get("head_d_rating_estimator", {})

    st.write("")
    st.subheader("What the models see")
    cards = st.columns(3)
    card(cards[0], "Likely genre", predicted[0] if predicted else "None confident",
         (f"also {', '.join(predicted[1:3])} · " if len(predicted) > 1 else "")
         + f"{len(predicted)} pass their threshold")
    if story["rating"] is not None:
        diff = story["rating"] - average
        card(cards[1], "Expected rating", f"{story['rating']:.2f}",
             f"±{head_d.get('mae', 0):.2f} typical error",
             delta=(diff, f"{diff:+.2f} vs catalogue"))
    else:
        card(cards[1], "Expected rating", "n/a", "the rating model is not trained on this data")
    card(cards[2], "Narrative family", f"Archetype {archetype['cluster_id']}",
         f"{profile.get('size', 0):,} titles · average rating {profile.get('avg_weighted_rating', 0):.2f}")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Genre confidence**")
        st.markdown(genre_bars_html(genres), unsafe_allow_html=True)
        st.caption("The bar is the model's confidence; the tick is that genre's own threshold, tuned so "
                   "rare genres can still be detected.")
    with right:
        st.markdown(f"**Archetype {archetype['cluster_id']}: signature terms**")
        st.markdown(chips(profile.get("top_terms", [])[:10], active=True), unsafe_allow_html=True)
        top = profile.get("top_genres", {})
        if top:
            st.caption("Dominant genres: " + ", ".join(list(top)[:4]))
        examples = profile.get("example_titles", [])[:4]
        if examples:
            st.caption("Typical titles: " + " · ".join(examples))
        st.caption(f"Typical length {profile.get('avg_chapters', 0):,.0f} chapters.")

    st.write("")
    st.subheader("Working-title ideas")
    _title_ideas(registry, story, predicted)

    st.write("")
    st.subheader("Closest titles in the catalogue")
    similar = story["similar"]
    shown = st.radio("Show", [6, 12], horizontal=True, format_func=lambda n: f"{n} titles", key="similar_n")
    for _, row in similar.head(shown).iterrows():
        novel_card(row)


def _title_ideas(registry, story: dict, predicted: list[str]) -> None:
    with st.spinner("Learning title patterns from the catalogue…"):
        generator = title_generator()
    origin = story["origin"] if story["origin"] in TITLE_STYLES else "Any"
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    style = c1.selectbox("Title style", list(TITLE_STYLES), index=list(TITLE_STYLES).index(origin),
                         format_func=TITLE_STYLES.get, key="title_style",
                         help="Chinese titles run short and formulaic; Japanese light-novel titles are often whole sentences.")
    more = c2.button("New ideas", use_container_width=True, key="title_go")

    signature = (story["synopsis"], style)
    if more or st.session_state.get("title_signature") != signature or "title_ideas" not in st.session_state:
        if st.session_state.get("title_signature") != signature:
            st.session_state["title_signature"], st.session_state["title_seen"] = signature, set()
        st.session_state["title_batch"] = st.session_state.get("title_batch", 0) + 1
        similarity = pd.Series(registry.embeddings @ registry.embed(f"{story['synopsis']} {story['tags']}"),
                               index=registry.catalog["series_id"].to_numpy())
        weights = generator.title_weights(
            generator.catalog["series_id"].map(similarity).to_numpy(dtype=float),
            predicted[:3], None if style == "Any" else style,
        )
        keywords = description_keywords(story["synopsis"]) | description_keywords(" ".join(predicted[:3]))
        ideas = generator.generate(weights, keywords, n=8, seed=st.session_state["title_batch"],
                                   exclude=st.session_state.get("title_seen", set()))
        st.session_state["title_seen"] = st.session_state.get("title_seen", set()) | {i.title.lower() for i in ideas}
        st.session_state["title_ideas"] = ideas

    ideas = st.session_state["title_ideas"]
    if not ideas:
        st.info("No new title survived the filters. Add more story detail or switch the title style.")
        return
    cols = st.columns(2, gap="medium")
    for i, idea in enumerate(ideas):
        sources = " · ".join(html.escape(s if len(s) <= 70 else s[:67].rstrip() + "…") for s in idea.sources[:2])
        cols[i % 2].markdown(
            f'<div class="nl-idea"><div class="t">{html.escape(idea.title)}</div>'
            f'<div class="src">Phrasing from: {sources}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("Every three-word run comes from a catalogue title; the combinations are new, and ideas that copy "
               "a title or a phrase only one title uses are filtered out. No language model is involved, so check "
               "that a title is free before using it.")


def _studio_similar(registry) -> None:
    st.markdown("Pick a title you already like to find the ones whose story content sits closest to it.")
    options, labels = reference_options(registry)
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    series_id = int(c1.selectbox("Reference title", options, format_func=labels.get, key="ref_title"))
    k = c2.selectbox("Results", [6, 9, 12], key="ref_k")
    try:
        results = registry.recommend_by_series(series_id, k=k)
    except KeyError:
        st.warning("That title is not in the model's catalogue yet.")
        return
    st.write("")
    for _, row in results.iterrows():
        novel_card(row)


# PAGE — Methodology
def page_method() -> None:
    hero(
        "Methodology & limitations",
        "How the numbers in this app are computed, how accurate the models are, "
        "and where the output should not be taken at face value.",
    )

    viz = load_table("viz_novels")
    synopsis_share = viz["has_synopsis"].mean() if "has_synopsis" in viz else None
    chapter_gap = viz["chapter_count"].isna().mean()
    st.subheader("Where the data comes from")
    st.markdown(
        f"""
The catalogue holds **{len(viz):,} translated web novel titles** with their metadata,
reader ratings, and synopses. Before use, the data goes through cleaning:

- Missing values disguised as the number `0` or the text `"Unknown"` / `"N/A"` are
  restored to genuinely empty. {chapter_gap:.0%} of titles have no known chapter count;
  counting them as 0 would drag the average length down by the same share.
- Ratings are shown as a **weighted rating**: a title with 2 votes is not
  treated like one with 500. Every ranking in this app uses only titles with at
  least 10 votes.
""".rstrip()
        + "\n"
        + (
            f"- **{synopsis_share:.1%}** of titles carry a synopsis, which is what the "
            "models read.\n"
            if synopsis_share is not None
            else ""
        )
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
        "are chosen by cross-validation inside the development set. The test set is "
        "touched once, at the very end, so the numbers below are not inflated."
    )

    rows = []
    best = head_a.get("best_model")
    for name, entry in head_a["models"].items():
        rows.append({
            "Model": entry.get("label", name) + (" (selected)" if name == best else ""),
            "Approach": model_text(name, entry)[0],
            "Macro F1 (cross-validation)": entry.get("f1_macro_oof"),
            "Macro F1": entry["f1_macro"],
            "Micro F1": entry["f1_micro"],
            "P@3": entry.get("precision_at_3"),
            "Hamming loss": entry.get("hamming_loss"),
            "Macro F1 at a 0.5 threshold": entry["f1_macro_fixed_threshold"],
        })
    table = pd.DataFrame(rows).sort_values("Macro F1 (cross-validation)", ascending=False)
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "Macro F1 (cross-validation)": st.column_config.NumberColumn(format="%.3f"),
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
ignored. The model is selected on its cross-validated macro F1 inside the development set;
the test-set column then confirms the choice without having influenced it. <b>Micro F1</b> is dominated by common
genres. <b>P@3</b> measures how accurate the top three guesses are without depending on a
threshold at all, which is closest to what users actually see. <b>Hamming loss</b> is the
share of binary decisions that are wrong (lower is better). The last column shows the same
model scored with a uniform 0.5 threshold; the gap is the contribution of per-genre
threshold tuning.
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**Architecture references.** The two feature-based models (TF-IDF + linear) follow the
baselines common in genre-classification-from-synopsis research. The hybrid model adds
pretrained sentence embeddings (BGE, Xiao et al., 2023; Sentence-BERT, Reimers &
Gurevych, 2019) next to the TF-IDF block. The sequence model replicates a BiLSTM with self-attention
(Ertugrul & Karagoz, 2018; Zhou et al., 2016), with multi-label handling via weighted
per-label *binary cross-entropy* (Liu et al., SIGIR 2017). Per-genre probability threshold
tuning follows Hoang (2018).
        """
    )

    st.write("")
    st.subheader("Other features")
    cols = st.columns(3)
    card(cols[0], "Recommendations", f"{head_b['genre_overlap_at_k']:.0%}",
         f"of {head_b['k']} suggestions share a genre with the reference title")
    card(cols[1], "Archetypes", f"{head_c['best_k']} clusters",
         "cluster count chosen automatically by silhouette score")
    if head_d.get("trained"):
        card(cols[2], "Rating estimate", f"±{head_d['mae']:.2f}",
             f"{head_d['improvement_over_baseline_pct']:.0f}% better than guessing the average")
    else:
        card(cols[2], "Rating estimate", "n/a", "not trained on this data")

    st.write("")
    cn_share = (viz["type_origin"] == "CN").mean()
    st.subheader("Limitations worth remembering")
    st.markdown(
        f"""
- The models read the **description**, not the manuscript. A novel with a poor
  synopsis will be judged poorly even if the story itself is good.
- Recommendations rest purely on content similarity; two titles on the same theme
  count as similar even when their writing quality differs sharply.
- The catalogue is dominated by Chinese novels ({cn_share:.0%}), so accuracy on English or
  Japanese titles tends to be lower.
- Ratings come from a single translated-fiction reader community, not the general
  population.
- Archetypes are overlapping groups, not hard partitions.
        """
    )


# Data platform pages
#
# The three pages below serve an analyst rather than a reader: they expose the
# star schema, the property graph and the governance audit that the pipeline
# materialises under curated-data/model, analytics and governance. Everything
# is read from those Parquet tables; nothing is recomputed with different
# rules, so a number here matches the number in the pipeline log.

#: Groups smaller than this are hidden for non-additive measures, the same
#: HAVING COUNT(*) >= 10 rule data_model.py uses for its OLAP roll-up.
MIN_GROUP = 10

#: Status labels come out of governance_audit.py in Indonesian.
STATUS_LABEL = {"LULUS": "Pass", "PERINGATAN": "Warning", "GAGAL": "Fail", "INFO": "Info"}
STATUS_COLOR = {"Pass": ACCENT, "Warning": "#FFB454", "Fail": "#FF6B9D", "Info": MUTED}

DIMENSIONS: dict[str, dict] = {
    "Genre": {"kind": "bridge", "bridge": "bridge_novel_genre", "dim": "dim_genre",
              "key": "genre_key", "label": "genre", "alias": "g", "bridge_alias": "bg"},
    "Tag (top 25)": {"kind": "bridge", "bridge": "bridge_novel_tag", "dim": "dim_tag",
                     "key": "tag_key", "label": "tag", "alias": "t", "bridge_alias": "bt",
                     "top": 25},
    "Country of origin": {"kind": "column", "col": "type_origin", "sql": "n.type_origin", "table": "n"},
    "Format": {"kind": "column", "col": "type", "sql": "n.type", "table": "n"},
    "Status": {"kind": "column", "col": "status", "table": "n",
               "sql": "CASE WHEN n.is_completed = 1 THEN 'Completed' ELSE 'Ongoing' END"},
    "Decade": {"kind": "column", "col": "decade", "sql": "f.decade", "ordinal": True},
    "Year": {"kind": "column", "col": "year", "sql": "f.year", "ordinal": True},
    "Length": {"kind": "column", "col": "length_bucket", "sql": "n.length_bucket",
               "table": "n", "ordinal": True},
    "Release pace": {"kind": "column", "col": "release_pace", "sql": "n.release_pace",
                     "table": "n", "ordinal": True},
    "Original publisher (top 15)": {"kind": "column", "col": "publisher", "sql": "p.publisher",
                                    "table": "p", "top": 15},
}

MEASURES: dict[str, dict] = {
    "Titles": {"col": "series_id", "agg": "nunique", "sql": "COUNT(DISTINCT f.series_id)",
               "additive": True, "fmt": ",.0f"},
    "Total votes": {"col": "rating_votes", "agg": "sum", "sql": "SUM(f.rating_votes)",
                    "additive": True, "fmt": ",.0f"},
    "Total reading-list adds": {"col": "reading_list_count", "agg": "sum",
                                "sql": "SUM(f.reading_list_count)", "additive": True, "fmt": ",.0f"},
    "Average weighted rating": {"col": "weighted_rating", "agg": "mean",
                                "sql": "ROUND(AVG(f.weighted_rating), 3)", "additive": False,
                                "fmt": ".3f"},
    "Average chapters": {"col": "chapter_count", "agg": "mean",
                         "sql": "ROUND(AVG(f.chapter_count), 1)", "additive": False, "fmt": ",.1f"},
    "Median days between chapters": {"col": "days_per_chapter", "agg": "median",
                                     "sql": "PERCENTILE_APPROX(f.days_per_chapter, 0.5)",
                                     "additive": False, "fmt": ",.1f"},
}


def load_star() -> pd.DataFrame:
    return _load_star(lake_version())


@st.cache_data(show_spinner=False, max_entries=2)
def _load_star(version: str) -> pd.DataFrame:
    """Fact table joined to its single-valued dimensions: one row per title."""
    fact = load_lake("model/fact_novel")
    novel = load_lake("model/dim_novel")[
        ["series_id", "title", "type", "type_origin", "is_completed", "length_bucket", "release_pace"]
    ]
    publisher = load_lake("model/dim_publisher")[["publisher_key", "publisher"]]
    star = fact.merge(novel, on="series_id", how="left").merge(publisher, on="publisher_key", how="left")
    star["status"] = np.where(star["is_completed"] == 1, "Completed", "Ongoing")
    # publisher_key is ranked by title count, so key <= N is exactly "top N".
    star["publisher_rank"] = star["publisher_key"]
    return star


def load_bridge(name: str) -> pd.DataFrame:
    return _load_bridge(name, lake_version())


@st.cache_data(show_spinner=False, max_entries=8)
def _load_bridge(name: str, version: str) -> pd.DataFrame:
    """Bridge table resolved to labels: (series_id, label), one row per pair."""
    spec = DIMENSIONS[name]
    bridge = load_lake(f"model/{spec['bridge']}")
    dim = load_lake(f"model/{spec['dim']}")
    if "top" in spec:
        dim = dim[dim[spec["key"]] <= spec["top"]]
    return bridge.merge(dim[[spec["key"], spec["label"]]], on=spec["key"])[
        ["series_id", spec["label"]]
    ].rename(columns={spec["label"]: name})


def dimension_frame(star: pd.DataFrame, name: str) -> pd.DataFrame:
    """(series_id, <name>) for the titles in `star`, exploded for bridge dimensions."""
    spec = DIMENSIONS[name]
    if spec["kind"] == "bridge":
        return load_bridge(name).merge(star[["series_id"]], on="series_id")
    frame = star[["series_id", spec["col"]]].rename(columns={spec["col"]: name})
    if "top" in spec:
        frame = frame[star["publisher_rank"] <= spec["top"]]
    frame = frame.dropna(subset=[name])
    if spec.get("ordinal") and name in ("Decade", "Year"):
        frame[name] = frame[name].astype(int)
    return frame


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_sql(rows: str, cols: str | None, measure: str, filters: dict) -> str:
    """The Spark SQL that computes the current view over the star schema tables."""
    dims = [d for d in (rows, cols) if d]
    joins = ["FROM        fact_novel f"]
    needs_n = any(DIMENSIONS[d].get("table") == "n" for d in dims) or any(
        filters.get(k) for k in ("origin", "format", "status")
    )
    if needs_n:
        joins.append("JOIN        dim_novel n       ON n.series_id = f.series_id")
    if any(DIMENSIONS[d].get("table") == "p" for d in dims):
        joins.append("JOIN        dim_publisher p   ON p.publisher_key = f.publisher_key")
    for d in dims:
        spec = DIMENSIONS[d]
        if spec["kind"] == "bridge":
            b, a = spec["bridge_alias"], spec["alias"]
            joins.append(f"JOIN        {spec['bridge']} {b} ON {b}.series_id = f.series_id")
            joins.append(f"JOIN        {spec['dim']} {a} ON {a}.{spec['key']} = {b}.{spec['key']}")

    def select_expr(d: str) -> str:
        spec = DIMENSIONS[d]
        return f"{spec['alias']}.{spec['label']}" if spec["kind"] == "bridge" else spec["sql"]

    where = []
    if filters.get("origin"):
        where.append("n.type_origin IN (" + ", ".join(map(sql_literal, filters["origin"])) + ")")
    if filters.get("format"):
        where.append("n.type IN (" + ", ".join(map(sql_literal, filters["format"])) + ")")
    if filters.get("status"):
        where.append("n.is_completed = 1" if filters["status"] == "Completed"
                     else "COALESCE(n.is_completed, 0) = 0")
    if filters.get("years"):
        lo, hi = filters["years"]
        where.append(f"f.year BETWEEN {lo} AND {hi}")
    if filters.get("reliable"):
        where.append("f.is_rated_reliable")
    if filters.get("decade") is not None:
        where.append(f"f.decade = {filters['decade']}")
    for d in dims:
        spec = DIMENSIONS[d]
        if spec["kind"] == "bridge" and "top" in spec:
            where.append(f"{spec['alias']}.{spec['key']} <= {spec['top']}")
        elif spec.get("table") == "p" and "top" in spec:
            where.append(f"f.publisher_key <= {spec['top']}")
        elif spec["kind"] == "column":
            where.append(f"{spec['sql']} IS NOT NULL")

    m = MEASURES[measure]
    select = [f"{select_expr(d):<40} AS {d.split(' (')[0].lower().replace(' ', '_')}" for d in dims]
    select.append(f"{m['sql']:<40} AS measure")
    select.append(f"{'COUNT(DISTINCT f.series_id)':<40} AS n_titles")
    lines = ["SELECT      " + ",\n            ".join(select), *joins]
    if where:
        lines.append("WHERE       " + "\n  AND       ".join(where))
    lines.append("GROUP BY    " + ", ".join(select_expr(d) for d in dims))
    if not m["additive"]:
        lines.append(f"HAVING      COUNT(f.{m['col']}) >= {MIN_GROUP}")
    lines.append("ORDER BY    measure DESC")
    return "\n".join(lines)


def aggregate(frame: pd.DataFrame, star: pd.DataFrame, dims: list[str], measure: str) -> pd.DataFrame:
    """Group `frame` (series_id + dims) by dims and compute `measure` from the fact columns."""
    m = MEASURES[measure]
    data = frame.merge(star[["series_id", m["col"]]] if m["col"] != "series_id" else star[["series_id"]],
                       on="series_id")
    grouped = data.groupby(dims, observed=True)
    out = grouped[m["col"]].agg(m["agg"]).rename("measure").to_frame()
    out["n_titles"] = grouped["series_id"].nunique()
    if not m["additive"]:
        out = out[grouped[m["col"]].count() >= MIN_GROUP]
    return out.reset_index()


def fmt_value(value: float, measure: str) -> str:
    return format(value, MEASURES[measure]["fmt"])


def olap_operations(rows: str, cols: str | None, filters: dict, drilled: bool) -> list[str]:
    """Name the OLAP operations the current controls perform, in plain words."""
    ops = []
    fixed = []
    if filters.get("origin"):
        fixed.append("origin in " + ", ".join(filters["origin"]))
    if filters.get("format"):
        fixed.append("format in " + ", ".join(filters["format"]))
    if filters.get("status"):
        fixed.append(f"status = {filters['status']}")
    if filters.get("years"):
        fixed.append(f"year {filters['years'][0]} to {filters['years'][1]}")
    if filters.get("reliable"):
        fixed.append("10+ votes only")
    if len(fixed) == 1:
        ops.append(f"**Slice** ({fixed[0]})")
    elif fixed:
        ops.append("**Dice** (" + "; ".join(fixed) + ")")
    if rows == "Decade" or cols == "Decade":
        ops.append("**Roll-up** (year to decade)")
    if drilled:
        ops.append("**Drill-down** (decade to year)")
    if cols:
        ops.append(f"**Pivot** ({rows} by {cols})")
    return ops


#: Power BI model-view layout, (x, y, width, role) in SVG units: single-valued
#: dimensions left, fact centre, bridges and many-valued dimensions right.
SCHEMA_LAYOUT = {
    "dim_novel": (20, 24, 220, "dim"),
    "dim_time": (20, 300, 220, "dim"),
    "dim_publisher": (20, 452, 220, "dim"),
    "fact_novel": (345, 110, 245, "fact"),
    "bridge_novel_genre": (690, 46, 200, "bridge"),
    "bridge_novel_tag": (690, 236, 200, "bridge"),
    "bridge_novel_author": (690, 426, 200, "bridge"),
    "dim_genre": (985, 24, 215, "dim"),
    "dim_tag": (985, 214, 215, "dim"),
    "dim_author": (985, 404, 215, "dim"),
}
#: (one side, its key, many side, its key, cardinality at the many end, filters both ways)
SCHEMA_RELATIONS = [
    ("dim_novel", "series_id", "fact_novel", "series_id", "1", False),
    ("dim_time", "year", "fact_novel", "year", "*", False),
    ("dim_publisher", "publisher_key", "fact_novel", "publisher_key", "*", False),
    ("fact_novel", "series_id", "bridge_novel_genre", "series_id", "*", True),
    ("fact_novel", "series_id", "bridge_novel_tag", "series_id", "*", True),
    ("fact_novel", "series_id", "bridge_novel_author", "series_id", "*", True),
    ("dim_genre", "genre_key", "bridge_novel_genre", "genre_key", "*", False),
    ("dim_tag", "tag_key", "bridge_novel_tag", "tag_key", "*", False),
    ("dim_author", "author_key", "bridge_novel_author", "author_key", "*", False),
]
SCHEMA_PK = {"dim_novel": "series_id", "dim_time": "year", "dim_publisher": "publisher_key",
             "fact_novel": "series_id", "dim_genre": "genre_key", "dim_tag": "tag_key",
             "dim_author": "author_key"}
#: Same split as the measures table under the diagram.
SUMMABLE = {"chapter_count", "rating_votes", "reading_list_count"}
SCHEMA_HEADER, SCHEMA_ROW, SCHEMA_MAX_ROWS = 46, 21, 7


def _schema_rows(name: str, columns: list[str]) -> list[tuple[str, str]]:
    """(column, marker) in model-view order: keys first, then the rest."""
    keys = {r[1] for r in SCHEMA_RELATIONS if r[0] == name} | {r[3] for r in SCHEMA_RELATIONS if r[2] == name}

    def marker(col: str) -> str:
        if SCHEMA_PK.get(name) == col:
            return "PK"
        if col in keys:
            return "FK"
        return "Σ" if name == "fact_novel" and col in SUMMABLE else ""

    ordered = sorted(columns, key=lambda c: (marker(c) not in ("PK", "FK"), columns.index(c)))
    return [(c, marker(c)) for c in ordered]


def star_schema_svg(tables: dict[str, list[str]], counts: dict[str, int]) -> str:
    """Star schema as inline SVG in Power BI's model-view style: table cards with
    PK / FK / Σ markers, cardinality at each line end, filter direction mid-line."""
    surface, line, ink, muted = "#161B26", "#6B7390", "#E6E9EF", MUTED
    header_fill = {"fact": "#2A2152", "dim": "#1E2433", "bridge": "#1B2230"}
    border = {"fact": BRAND, "dim": "rgba(255,255,255,.16)", "bridge": "rgba(255,255,255,.24)"}
    anchors: dict[tuple[str, str], float] = {}
    boxes: dict[str, tuple[float, float, float, float]] = {}
    cards = []

    for name, (x, y, w, role) in SCHEMA_LAYOUT.items():
        rows = _schema_rows(name, tables.get(name, []))
        limit = len(rows) if role == "fact" else SCHEMA_MAX_ROWS
        shown, hidden = rows[:limit], len(rows) - min(limit, len(rows))
        h = SCHEMA_HEADER + SCHEMA_ROW * (len(shown) + (1 if hidden else 0)) + 10
        boxes[name] = (x, y, w, h)
        dash = ' stroke-dasharray="5 4"' if role == "bridge" else ""
        kind = {"fact": "fact table", "dim": "dimension", "bridge": "bridge table"}[role]
        parts = [
            f"<g><title>{name}: {counts.get(name, 0):,} rows, {len(rows)} columns ({kind})</title>",
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{surface}" '
            f'stroke="{border[role]}" stroke-width="{1.6 if role == "fact" else 1.1}"{dash}/>',
            f'<path d="M{x + 1},{y + SCHEMA_HEADER} V{y + 8} Q{x + 1},{y + 1} {x + 8},{y + 1} '
            f'H{x + w - 8} Q{x + w - 1},{y + 1} {x + w - 1},{y + 8} V{y + SCHEMA_HEADER} Z" '
            f'fill="{header_fill[role]}"/>',
            f'<line x1="{x}" y1="{y + SCHEMA_HEADER}" x2="{x + w}" y2="{y + SCHEMA_HEADER}" '
            f'stroke="{border[role]}" stroke-width="1"/>',
            f'<text x="{x + 12}" y="{y + 20}" fill="#F1EDFF" font-size="13.5" font-weight="700">{name}</text>',
            f'<text x="{x + 12}" y="{y + 37}" fill="{muted}" font-size="11">'
            f"{counts.get(name, 0):,} rows · {kind}</text>",
        ]
        for i, (col, mark) in enumerate(shown):
            cy = y + SCHEMA_HEADER + 8 + SCHEMA_ROW * i + SCHEMA_ROW / 2
            anchors[(name, col)] = cy
            weight = "650" if mark in ("PK", "FK") else "450"
            parts.append(f'<text x="{x + 12}" y="{cy + 4}" fill="{ink if mark else "#C3C9D6"}" '
                         f'font-size="12" font-weight="{weight}">{col}</text>')
            if mark:
                parts.append(f'<text x="{x + w - 12}" y="{cy + 4}" fill="{muted}" font-size="10.5" '
                             f'font-weight="650" text-anchor="end">{mark}</text>')
        if hidden:
            cy = y + SCHEMA_HEADER + 8 + SCHEMA_ROW * len(shown) + SCHEMA_ROW / 2
            parts.append(f'<text x="{x + 12}" y="{cy + 4}" fill="{muted}" font-size="11">'
                         f'+{hidden} more column{"s" if hidden > 1 else ""}</text>')
        parts.append("</g>")
        cards.append("".join(parts))

    links = []
    for one, one_key, many, many_key, many_card, both in SCHEMA_RELATIONS:
        (ax, ay, aw, _), (bx, by, bw, _) = boxes[one], boxes[many]
        y1 = anchors.get((one, one_key), ay + SCHEMA_HEADER / 2)
        y2 = anchors.get((many, many_key), by + SCHEMA_HEADER / 2)
        rightward = ax < bx
        x1, x2 = (ax + aw, bx) if rightward else (ax, bx + bw)
        bend = max(40.0, abs(x2 - x1) * 0.45) * (1 if rightward else -1)
        c1, c2 = (x1 + bend, y1), (x2 - bend, y2)
        links.append(f'<path d="M{x1},{y1} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {x2},{y2}" '
                     f'fill="none" stroke="{line}" stroke-width="1.4"/>')
        # Cardinality just outside each card, as in Power BI.
        step = 7 if rightward else -7
        anchor_a, anchor_b = ("start", "end") if rightward else ("end", "start")
        links.append(f'<text x="{x1 + step}" y="{y1 - 6}" fill="{ink}" font-size="12" '
                     f'font-weight="700" text-anchor="{anchor_a}">1</text>')
        links.append(f'<text x="{x2 - step}" y="{y2 - 6}" fill="{ink}" font-size="12" '
                     f'font-weight="700" text-anchor="{anchor_b}">{many_card}</text>')
        # Filter direction at the curve's midpoint, from the "1" side to the many side.
        mx = (x1 + 3 * c1[0] + 3 * c2[0] + x2) / 8
        my = (y1 + 3 * c1[1] + 3 * c2[1] + y2) / 8
        dx = 0.75 * (c1[0] - x1) + 1.5 * (c2[0] - c1[0]) + 0.75 * (x2 - c2[0])
        dy = 0.75 * (c1[1] - y1) + 1.5 * (c2[1] - c1[1]) + 0.75 * (y2 - c2[1])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        for offset, flip in ([(0, 0)] if not both else [(7, 0), (-7, 180)]):
            links.append(
                f'<g transform="translate({mx:.1f},{my:.1f}) rotate({angle:.1f}) translate({offset},0) '
                f'rotate({flip})"><polygon points="-5,-4.5 5,0 -5,4.5" fill="#C3C9D6"/></g>'
            )

    width = max(x + w for x, _, w, _ in boxes.values()) + 20
    height = max(y + h for _, y, _, h in boxes.values()) + 20
    font = FONT_STACK.replace('"', "&quot;")
    return (
        '<div style="overflow-x:auto;padding-bottom:.3rem">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        'aria-label="Star schema: fact_novel joined to dim_novel, dim_time and dim_publisher, and '
        'through bridge tables to dim_genre, dim_tag and dim_author" '
        f'style="width:100%;min-width:900px;height:auto;font-family:{font}">'
        + "".join(links) + "".join(cards) + "</svg></div>"
    )


def page_warehouse() -> None:
    hero(
        "Data warehouse",
        "The cleaned catalogue modelled as a star schema. Pick a dimension and a measure "
        "below to slice, dice, roll up, drill down or pivot it.",
    )

    with st.spinner("Loading the star schema tables…"):
        star = load_star()
        counts = {
            name: table_profile(f"model/{name}")[0]
            for name in ("fact_novel", "dim_novel", "dim_time", "dim_publisher", "dim_genre",
                         "dim_tag", "dim_author", "bridge_novel_genre", "bridge_novel_tag",
                         "bridge_novel_author")
        }

    columns = {name: list(load_lake(f"model/{name}").columns) for name in counts}
    st.markdown(star_schema_svg(columns, counts), unsafe_allow_html=True)
    st.caption("Read it like Power BI's model view: 1 and * mark each side of a relationship, the "
               "arrow is the filter direction, and a double arrow means filters cross both ways. "
               "The fact-to-bridge links need that, or a genre, tag or author filter could not "
               "reach the fact table. Σ marks the summable measures; dashed cards are bridge tables.")
    notes, measures = st.columns([2, 3], gap="large")
    with notes:
        st.markdown("**Why this shape**")
        st.markdown(
            f"The fact table holds **one row per title** ({counts['fact_novel']:,} rows). "
            "Genres, tags and authors are many-valued, so they hang off bridge tables "
            "instead of being copied into the fact. A title with five genres keeps its "
            "vote count once, not five times. Single-valued dimensions sit on the left, "
            "many-valued ones reach the fact through a bridge on the right."
        )
    with measures:
        st.dataframe(
            pd.DataFrame([
                ("chapter_count", "Additive", "SUM, AVG"),
                ("rating_votes", "Additive", "SUM"),
                ("reading_list_count", "Additive", "SUM"),
                ("weighted_rating", "Non-additive", "AVG only"),
                ("rating_score", "Non-additive", "AVG only"),
                ("days_per_chapter", "Non-additive", "AVG, median"),
            ], columns=["Measure in fact_novel", "Type", "Valid aggregation"]),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    st.subheader("Ask the warehouse")

    c1, c2, c3 = st.columns(3)
    rows = c1.selectbox("Break down by", list(DIMENSIONS), index=0)
    col_options = ["Nothing (single breakdown)"] + [d for d in DIMENSIONS if d != rows]
    cols_pick = c2.selectbox("Compare across", col_options, index=0)
    cols = None if cols_pick == col_options[0] else cols_pick
    measure = c3.selectbox("Measure", list(MEASURES), index=3)

    origins = star["type_origin"].value_counts().index.tolist()
    formats = star["type"].value_counts().index.tolist()
    with st.expander("Slice and dice filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        pick_origin = f1.multiselect("Country of origin", origins, placeholder="All origins")
        pick_format = f2.multiselect("Format", formats, placeholder="All formats")
        pick_status = f3.radio("Status", ["All", "Completed", "Ongoing"], horizontal=True)
        g1, g2 = st.columns([1, 2])
        use_years = g1.checkbox("Limit release year", help="Titles with no known year drop out when this is on.")
        valid_years = star["year"].dropna().astype(int)
        years = None
        if use_years:
            years = g2.slider("Release year", int(valid_years.min()), int(valid_years.max()),
                              (2015, int(valid_years.max())))
        reliable = g1.checkbox("Only titles with 10+ votes")

    filters = {
        "origin": pick_origin, "format": pick_format,
        "status": None if pick_status == "All" else pick_status,
        "years": years, "reliable": reliable,
    }
    view = star
    if pick_origin:
        view = view[view["type_origin"].isin(pick_origin)]
    if pick_format:
        view = view[view["type"].isin(pick_format)]
    if filters["status"]:
        view = view[view["status"] == filters["status"]]
    if years:
        view = view[view["year"].between(*years)]
    if reliable:
        view = view[view["is_rated_reliable"] == True]  # noqa: E712

    if cols and DIMENSIONS[rows]["kind"] == "bridge" and DIMENSIONS[cols]["kind"] == "bridge":
        st.info("Genre and tag are both many-valued. Compare one of them against a "
                "single-valued dimension such as origin or decade.")
        return
    if view.empty:
        st.info("No titles match these filters. Clear a filter or widen the year range.")
        return

    frame = dimension_frame(view, rows)
    if cols:
        frame = frame.merge(dimension_frame(view, cols), on="series_id")
    dims = [rows] + ([cols] if cols else [])
    result = aggregate(frame, view, dims, measure)
    if result.empty:
        st.info(f"No group reaches {MIN_GROUP} titles with a value for this measure. "
                "Loosen the filters or choose a coarser breakdown.")
        return

    ordinal = DIMENSIONS[rows].get("ordinal", False)
    drill = None
    if rows == "Decade" and not cols:
        decades = sorted(result["Decade"].unique().tolist(), reverse=True)
        drill_pick = st.selectbox(
            "Drill down into a decade", ["Stay at decade level"] + [str(d) for d in decades]
        )
        drill = None if drill_pick == "Stay at decade level" else int(drill_pick)

    ops = olap_operations(rows, cols, filters, drill is not None)
    # .at keeps each column's dtype; a row Series would turn decade 1970 into 1970.0.
    best_idx = result["measure"].idxmax()
    label = " / ".join(str(result.at[best_idx, d]) for d in dims)
    st.markdown(
        f"#### Highest {measure.lower()}: {label} "
        f"({fmt_value(result.at[best_idx, 'measure'], measure)})"
    )
    st.caption(
        f"{view['series_id'].nunique():,} titles in view"
        + (" · operations: " + ", ".join(ops) if ops else "")
    )

    if cols:
        pivot = result.pivot(index=rows, columns=cols, values="measure")
        if ordinal:
            pivot = pivot.sort_index()
        else:
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        if DIMENSIONS[cols].get("ordinal"):
            pivot = pivot[sorted(pivot.columns)]
        pivot.columns = [str(c) for c in pivot.columns]
        pivot.index = [str(i) for i in pivot.index]
        text_fmt = ".2f" if measure == "Average weighted rating" else ".3s"
        fig = px.imshow(pivot, color_continuous_scale=SCALE, aspect="auto", text_auto=text_fmt,
                        labels=dict(x=cols, y="", color=measure))
        st.plotly_chart(style_fig(fig, max(360, 28 * len(pivot))), use_container_width=True)
        st.caption("Empty cells: the combination does not exist or, for averages, has fewer "
                   f"than {MIN_GROUP} titles.")
    else:
        plot = result.copy()
        plot[rows] = plot[rows].astype(str)
        top_label = str(result.at[best_idx, rows])
        plot["lead"] = np.where(plot[rows] == top_label, "Highest", "Other")
        # Averages differ by tenths, so they get dots on a zoomed axis (see
        # dot_rating_chart); additive measures keep bars that start at zero.
        chart = px.bar if MEASURES[measure]["additive"] else px.scatter
        if ordinal:
            plot = plot.sort_values(rows)
            fig = chart(plot, x=rows, y="measure", color="lead",
                         color_discrete_map={"Highest": ACCENT, "Other": BRAND},
                         hover_data={"n_titles": ":,", "lead": False},
                         labels={"measure": measure, rows: "", "n_titles": "Titles"})
            height = 380
        else:
            plot = plot.sort_values("measure")
            fig = chart(plot, x="measure", y=rows, color="lead",
                        color_discrete_map={"Highest": ACCENT, "Other": BRAND},
                        hover_data={"n_titles": ":,", "lead": False},
                        labels={"measure": measure, rows: "", "n_titles": "Titles"},
                        **({"orientation": "h"} if MEASURES[measure]["additive"] else {}))
            height = max(340, 24 * len(plot))
        fig.update_layout(showlegend=False)
        if not MEASURES[measure]["additive"]:
            fig.update_traces(marker=dict(size=12, line=dict(width=2, color="#0D1117")))
            lo, hi = plot["measure"].min(), plot["measure"].max()
            pad = max(hi - lo, 0.1) * 0.1
            (fig.update_yaxes if ordinal else fig.update_xaxes)(range=[lo - pad, hi + pad])
        st.plotly_chart(style_fig(fig, height), use_container_width=True)

    multi = [d for d in dims if DIMENSIONS[d]["kind"] == "bridge"]
    if multi and MEASURES[measure]["additive"]:
        total = result["measure"].sum()
        distinct = frame["series_id"].nunique()
        st.warning(
            f"Summing this column gives {fmt_value(total, measure)}, but a title with several "
            f"{multi[0].split(' (')[0].lower()}s is counted once in each. That is why the fact "
            f"table keeps one row per title: the view itself holds {distinct:,} distinct titles. "
            "Read the bars as reach per group, not as shares of one total."
        )

    if drill is not None:
        st.markdown(f"**Drill-down: {drill}s by year**")
        in_decade = view[view["decade"] == drill]
        yearly = aggregate(dimension_frame(in_decade, "Year"), in_decade, ["Year"], measure)
        if yearly.empty:
            st.info(f"No year in the {drill}s reaches {MIN_GROUP} titles for this measure.")
        else:
            yearly["Year"] = yearly["Year"].astype(str)
            fig = px.bar(yearly, x="Year", y="measure", hover_data={"n_titles": ":,"},
                         labels={"measure": measure, "Year": "", "n_titles": "Titles"})
            fig.update_traces(marker_color=BRAND)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    table, sql = st.tabs(["Result table", "SQL for this view"])
    with table:
        shown = result.sort_values(dims if ordinal else "measure", ascending=ordinal)
        if measure == "Titles":
            shown = shown.drop(columns="n_titles")
        shown = shown.rename(columns={"measure": measure, "n_titles": "Titles"})
        st.dataframe(shown, hide_index=True, use_container_width=True, height=320)
        st.download_button(
            "Download this result as CSV", shown.to_csv(index=False).encode("utf-8"),
            file_name=f"warehouse_{rows}_{measure}.csv".replace(" ", "_").lower(),
            mime="text/csv",
        )
    with sql:
        st.caption("The same aggregation written against the star schema tables in "
                   "curated-data/model. The dashboard computes it from those Parquet files "
                   "with pandas; the pipeline runs this form in Spark SQL.")
        st.code(build_sql(rows, cols, measure, filters), language="sql")


@st.cache_data(ttl=900, show_spinner=False)
def genre_graph_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fixed positions so the map does not jump when the edge filter changes."""
    import networkx as nx

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        # Distance = 1 - Jaccard: genres that share many titles sit close together.
        graph.add_edge(row.source, row.target, dist=1.0 - row.jaccard)
    return {k: (float(v[0]), float(v[1])) for k, v in
            nx.kamada_kawai_layout(graph, weight="dist").items()}


def page_graph() -> None:
    hero(
        "Relationship graph",
        "Titles, genres, tags and authors as one property graph. The questions here are "
        "about connections, which the star schema cannot answer with a GROUP BY.",
    )

    import networkx as nx

    with st.spinner("Loading graph tables…"):
        n_nodes, node_counts = table_profile("model/graph_nodes", "node_type")
        n_edges, edge_counts = table_profile("model/graph_edges", "rel_type")
        summary = load_table("graph_summary").set_index("label")
        pagerank = load_table("graph_genre_pagerank")
        genre_dim = load_lake("model/dim_genre")[["genre", "n_novels"]]
        try:
            genre_edges = load_table("graph_genre_edges")
        except DataUnavailable:
            genre_edges = None

    st.markdown(
        f"**{n_nodes:,} nodes** "
        f"({', '.join(f'{v:,} {k}' for k, v in node_counts.items())}) and "
        f"**{n_edges:,} edges** "
        f"({', '.join(f'{v:,} {k}' for k, v in edge_counts.items())}), "
        "stored as graph_nodes / graph_edges so they load straight into Neo4j or GraphFrames."
    )

    genre_s, author_s = summary.loc["genre_cooccurrence"], summary.loc["author_collaboration"]
    compare = pd.DataFrame({
        "Genres": [genre_s["n_nodes"], genre_s["n_edges"], genre_s["n_components"],
                                genre_s["largest_component_size"], genre_s["n_communities"],
                                genre_s["modularity"]],
        "Authors": [author_s["n_nodes"], author_s["n_edges"], author_s["n_components"],
                                 author_s["largest_component_size"], author_s["n_communities"],
                                 author_s["modularity"]],
    }, index=["Nodes", "Edges", "Connected components", "Largest component",
              "Communities", "Modularity"])
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Two projections, opposite shapes**")
        st.dataframe(compare.map(lambda v: f"{v:,.4f}" if isinstance(v, float) and v < 1 else f"{v:,.0f}"),
                     use_container_width=True)
    with right:
        st.markdown("**What the numbers say**")
        st.markdown(
            f"- Genres form **one component** with modularity {genre_s['modularity']:.3f}: they "
            "mix freely, so a genre label alone separates titles poorly.\n"
            f"- Authors split into **{int(author_s['n_components']):,} separate teams**, the "
            f"largest only {int(author_s['largest_component_size'])} people, modularity "
            f"{author_s['modularity']:.3f}. Co-writing teams almost never cross over.\n"
            "- The warehouse can count pairs; ranking influence (PageRank), finding teams "
            "(components) and routing between genres (shortest path) need graph algorithms."
        )

    st.divider()
    st.subheader("Genre map")
    if genre_edges is None:
        st.info("The genre edge list has not been written yet. Run "
                "`python -m src.phases.graph_analysis` in the pipeline container.")
        return

    sizes = dict(zip(genre_dim["genre"], genre_dim["n_novels"]))
    genre_edges = genre_edges.assign(
        jaccard=lambda d: d["weight"] / (d["source"].map(sizes) + d["target"].map(sizes) - d["weight"])
    )
    positions = genre_graph_layout(genre_edges)

    ctrl, _ = st.columns([2, 3])
    threshold = ctrl.slider(
        "Show links with Jaccard similarity of at least", 0.05, 0.40, 0.15, 0.01,
        help="Jaccard = titles sharing both genres / titles with either. Raw counts would "
             "make big genres look linked to everything.",
    )
    shown = genre_edges[genre_edges["jaccard"] >= threshold]
    has_community = "community" in pagerank.columns
    info = pagerank.set_index("node")
    palette = [BRAND, ACCENT, "#FFB454", "#FF6B9D", "#4CC9F0"]

    map_col, rank_col = st.columns([3, 2], gap="large")
    with map_col:
        import plotly.graph_objects as go

        fig = go.Figure()
        for lo, hi, width in ((0, 0.2, 0.8), (0.2, 0.3, 1.8), (0.3, 1.01, 3.2)):
            band = shown[(shown["jaccard"] >= lo) & (shown["jaccard"] < hi)]
            xs, ys = [], []
            for row in band.itertuples(index=False):
                (x0, y0), (x1, y1) = positions[row.source], positions[row.target]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(width=width, color="rgba(174,181,198,.35)")))
        names = [n for n in positions]
        pr = info["pagerank"].reindex(names).fillna(0)
        community = info["community"].reindex(names).fillna(-1).astype(int) if has_community else pd.Series(0, index=names)
        fig.add_trace(go.Scatter(
            x=[positions[n][0] for n in names], y=[positions[n][1] for n in names],
            mode="markers+text", text=names, textposition="top center",
            textfont=dict(size=11, color="#D5DAE5"),
            marker=dict(size=(12 + 260 * pr).clip(upper=46),
                        color=[palette[c % len(palette)] if c >= 0 else MUTED for c in community],
                        line=dict(width=1, color="#0D1117")),
            customdata=np.stack([pr.values, community.values, [sizes.get(n, 0) for n in names]], axis=1),
            hovertemplate="<b>%{text}</b><br>PageRank %{customdata[0]:.4f}<br>"
                          "Community %{customdata[1]}<br>%{customdata[2]:,} titles<extra></extra>",
            showlegend=False,
        ))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig, 560), use_container_width=True)
        st.caption(
            f"{len(shown)} of {len(genre_edges)} links shown. Dot size = PageRank, colour = "
            "community found by greedy modularity in graph_analysis.py."
        )
    with rank_col:
        st.markdown("**Most central genres (PageRank)**")
        top = pagerank.nlargest(12, "pagerank").sort_values("pagerank")
        fig = px.bar(top, x="pagerank", y="node", orientation="h",
                     labels={"pagerank": "PageRank", "node": ""})
        fig.update_traces(marker_color=BRAND)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        if has_community:
            groups = pagerank.sort_values("pagerank", ascending=False).groupby("community")["node"]
            for cid, members in groups:
                st.markdown(
                    f'<span style="color:{palette[int(cid) % len(palette)]};font-weight:600">'
                    f"Community {int(cid)}</span> · " + ", ".join(members.head(8))
                    + (f" and {len(members) - 8} more" if len(members) > 8 else ""),
                    unsafe_allow_html=True,
                )

    st.write("")
    st.subheader("Route between two genres")
    graph = nx.Graph()
    graph.add_nodes_from(positions)
    for row in shown.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight, jaccard=row.jaccard)
    genres = sorted(positions)
    a, b = st.columns(2)
    src = a.selectbox("From", genres, index=genres.index("Yaoi") if "Yaoi" in genres else 0)
    dst = b.selectbox("To", genres, index=genres.index("Martial Arts") if "Martial Arts" in genres else 1)
    if src == dst:
        st.info("Pick two different genres.")
    elif not nx.has_path(graph, src, dst):
        st.info(f"No route between {src} and {dst} at Jaccard {threshold:.2f}. Lower the "
                "threshold on the map to let weaker links through.")
    else:
        path = nx.shortest_path(graph, src, dst)
        hops = [
            f"{u} to {v}: {graph[u][v]['weight']:,} shared titles (Jaccard {graph[u][v]['jaccard']:.2f})"
            for u, v in zip(path, path[1:])
        ]
        st.markdown(f"**{len(path) - 1} hop{'s' if len(path) > 2 else ''}:** " + chips(path, active=True),
                    unsafe_allow_html=True)
        st.markdown("\n".join(f"- {h}" for h in hops))

    st.divider()
    st.subheader("Authors")
    reach = load_lake("model/op_graph_author_reach")
    authors_pr = load_table("graph_author_pagerank")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Widest genre range** (Author, Novel, Genre: two hops)")
        min_titles = st.slider("Minimum titles per author", 3, 20, 8)
        subset = reach[reach["n_novels"] >= min_titles].nlargest(12, "n_distinct_genres")
        if subset.empty:
            st.info("No author has that many titles. Lower the minimum.")
        else:
            st.dataframe(
                subset[["author", "n_novels", "n_distinct_genres", "avg_weighted_rating"]].rename(
                    columns={"author": "Author", "n_novels": "Titles",
                             "n_distinct_genres": "Genres covered", "avg_weighted_rating": "Rating"}),
                hide_index=True, use_container_width=True, height=380,
            )
    with col_b:
        st.markdown("**Most connected co-authors** (collaboration graph)")
        top_auth = authors_pr.nlargest(12, "degree_centrality")[["node", "degree_centrality", "pagerank"]]
        n_auth = int(author_s["n_nodes"])
        top_auth = top_auth.assign(collaborators=(top_auth["degree_centrality"] * (n_auth - 1)).round().astype(int))
        st.dataframe(
            top_auth[["node", "collaborators", "pagerank"]].rename(
                columns={"node": "Author", "collaborators": "Co-authors", "pagerank": "PageRank"}),
            hide_index=True, use_container_width=True, height=380,
            column_config={"PageRank": st.column_config.NumberColumn(format="%.5f")},
        )


#: English, publish-safe audit text rebuilt from the numbers in the pipeline's evidence.
#: Security lines keep their status but never name credentials, accounts, endpoints or ports.
def _num(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text or "")
    return match.group(1) if match else default


def _int(text: str) -> str:
    try:
        return f"{int(str(text).replace(',', '').replace('.', '')):,}"
    except ValueError:
        return str(text)


def _zone(check_id: str) -> str:
    bucket = check_id.split("/", 1)[-1]
    return {"raw-data": "raw", "clean-data": "clean", "curated-data": "curated"}.get(bucket, "storage")


def _cmp01(ev: str) -> str:
    files = re.findall(r"([\w.]+\.csv): md5=\w+\.*, (\d+) baris", ev or "")
    if not files:
        return "Every source file carries a checksum and ingestion time."
    listed = ", ".join(f"{name} ({_int(rows)} rows)" for name, rows in files)
    return f"{len(files)} source files carry a checksum, row count and ingestion time: {listed}."


def _cmp03(ev: str) -> str:
    pairs = re.findall(r"([\w-]+)=(\w+)", ev or "")
    on = [_zone("x/" + b) for b, state in pairs if state.lower() == "enabled"]
    off = [_zone("x/" + b) for b, state in pairs if state.lower() != "enabled"]
    if pairs and not off:
        names = on[0] if len(on) == 1 else ", ".join(on[:-1]) + " and " + on[-1]
        return f"Versioning is on for the {names} zones."
    return f"Versioning on: {', '.join(on) or 'none'}; off: {', '.join(off) or 'none'}."


AUDIT_PUBLIC: dict[str, tuple[str, object, str]] = {
    # check_id: (name, evidence(ev, status) -> str, follow-up shown on the page)
    "SEC-01": ("Storage credentials changed from the defaults",
               lambda ev, s: "Every storage account uses its own long secret, none of them a default "
               "or example value." if s == "LULUS" else
               "The storage still runs with its default credentials. That is acceptable only while it "
               "cannot be reached from any network.", ""),
    "SEC-02": ("Encryption in transit (TLS)",
               lambda ev, s: ("Storage traffic is encrypted: " + _num(r"^(TLSv[\d.]+)", ev, "TLS")
                              + ", with the server certificate verified against the stack's own CA.")
               if s == "LULUS" else
               "Storage traffic is not encrypted. It stays on a private container network, which is "
               "acceptable only while that traffic never leaves the machine.", ""),
    "SEC-03": ("Storage zone closed to anonymous access",
               lambda ev, s: "No anonymous access policy." if s == "LULUS" else "Anonymous access is possible.", ""),
    "SEC-04": ("Separate read and write permissions",
               lambda ev, s: "This dashboard reads through a read-only account; only the pipeline can write.", ""),
    "SEC-05": ("Services isolated on a private network",
               lambda ev, s: "All services share one private container network; only user interfaces are "
               "published to the host.", ""),
    "CMP-01": ("Provenance recorded (checksum and time)", lambda ev, s: _cmp01(ev), ""),
    "CMP-02": ("Raw zone never overwritten by transformations",
               lambda ev, s: "The raw zone holds only registered sources and their manifests ("
               + _num(r"(\d+) objek", ev, "?") + " objects), each still the size recorded at ingestion.", ""),
    "CMP-03": ("Bucket versioning (recovery and audit)", lambda ev, s: _cmp03(ev), ""),
    "CMP-04": ("Data source documented",
               lambda ev, s: "The data comes from the public NovelUpdates catalogue: work metadata plus scraped "
               "synopses. The metadata is factual; synopses remain the copyright of their owners and are used "
               "here for non-commercial analytics only.",
               "Redistribution rights must be confirmed before the dataset itself is republished."),
    "PRV-01": ("Direct PII scan (email and phone)",
               lambda ev, s: "No match in any text column of the source files." if s == "LULUS" else
               "Possible matches found; they are reviewed before release.", ""),
    "PRV-02": ("Identity-like number patterns (manual review)",
               lambda ev, s: "None found." if (ev or "").strip() in ("tidak ada", "") else
               "Some number runs look like identifiers; in prose they are usually ordinary numbers.", ""),
    "PRV-03": ("No person-level data",
               lambda ev, s: "No column has one row per reader. Reader signals exist only as aggregates "
               "(vote counts, reading-list counts), so no individual can be re-identified.", ""),
    "PRV-04": ("Personal names that are processed",
               lambda ev, s: "The authors column holds pen names the platform publishes as credits, not private "
               "identities. The artists column is dropped during cleaning (" + _num(r"([\d,]+%)", ev, "mostly").replace(",", ".")
               + " Unknown).",
               "Pen names can still be personal data when linked to a real person; the report states the basis "
               "for processing them."),
    "QLT-01": ("Clean catalogue keeps every source row (at least 99.9%)",
               lambda ev, s: _num(r"^([\d,]+) dari", ev, "?") + " of " + _num(r"dari ([\d,]+)", ev, "?")
               + " source rows kept (" + _num(r"\(([\d.]+%)\)", ev, "?") + "); only duplicate series are removed. "
               + _int(_num(r"full=([\d,]+)", ev, "0")) + " titles have full metadata, "
               + _int(_num(r"listing=([\d,]+)", ev, "0")) + " listing-only.", ""),
    "QLT-01b": ("Listing-only titles are flagged, not hidden",
                lambda ev, s: _int(_num(r"^([\d,]+)", ev, "0")) + " titles carry only text and genres; their "
                "rating, chapters, year and popularity are empty in every row (checked, not assumed). Metrics "
                "from the detail pages filter them out.", ""),
    "QLT-02": ("series_id is unique",
               lambda ev, s: _num(r"^([\d,]+)", ev, "?") + " unique values across "
               + _num(r"dari ([\d,]+)", ev, "?") + " rows.", ""),
    "QLT-03": ("Placeholder values turned into empty values",
               lambda ev, s: "No placeholder value (0 chapters, \"N/A\" and the like) is left in the clean zone."
               if s == "LULUS" else "Some placeholder values remain in the clean zone.", ""),
    "QLT-04": ("Synopsis coverage of the full-metadata core ≥ 98%",
               lambda ev, s: _num(r"^([\d.]+%)", ev, "?") + " (" + _num(r"\(([\d,/]+)\)", ev, "?")
               + ") of full-metadata titles have a synopsis.", ""),
    "QLT-04b": ("Synopsis coverage of listing-only titles",
                lambda ev, s: _num(r"^([\d.]+%)", ev, "?") + " (" + _num(r"\(([\d,/]+)\)", ev, "?") + ").", ""),
    "QLT-05": ("Reliable-rating population reported",
               lambda ev, s: _num(r"^([\d,]+)", ev, "?") + " titles have 10+ votes out of "
               + _num(r"dari ([\d,]+)", ev, "?") + ". Rating averages use this population or the weighted "
               "rating, never the raw mean.", ""),
    "QLT-06": ("Ratings within the valid range (0 to 5)",
               lambda ev, s: "All values are in range." if s == "LULUS" else "Some ratings fall outside 0 to 5.",
               ""),
}


def public_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """English, publish-safe name / evidence / follow-up for every audit row."""
    def one(row) -> tuple[str, str, str]:
        base = row.check_id.split("/", 1)[0]
        spec = AUDIT_PUBLIC.get(row.check_id) or AUDIT_PUBLIC.get(base)
        if spec is None:
            # A check added to the pipeline later: show its id, never its raw text.
            return row.check_id, "Details are in the pipeline's audit log.", ""
        name, evidence, follow = spec
        if base == "SEC-03":
            name = f"{_zone(row.check_id).capitalize()} zone closed to anonymous access"
        return name, evidence(row.evidence, row.status), follow
    text = [one(r) for r in audit.itertuples(index=False)]
    return audit.assign(name_en=[t[0] for t in text], evidence_en=[t[1] for t in text],
                        follow_en=[t[2] for t in text])


#: Funnel notes, by step id.
FUNNEL_NOTES = {
    "1_raw_loaded": lambda n: "novel_catalogue.csv read from the raw zone",
    "2_identified": lambda n: "Rows with a URL and a title",
    "3_sentinels_to_null": lambda n: "Placeholder values emptied: " + ", ".join(
        f"{col} {_int(v)}" for col, v in re.findall(r"(\w+)=([\d,]+)", n)),
    "4_deduplicated": lambda n: "Duplicates removed on series_id",
    "5_with_synopsis": lambda n: "Synopsis of at least 10 words (" + _num(r"([\d.]+%)", n, "?") + " coverage)",
    "6_enriched": lambda n: "Weighted rating added (catalogue prior C = " + _num(r"=([\d.]+)", n, "?") + ")",
    "7_with_text_signal": lambda n: "At least 3 words of text to model",
    "8_reliable_rating": lambda n: "10+ votes: enters rating averages",
}

ELEMENT_TERMS = {
    "Terstruktur": "Structured", "Semi-terstruktur": "Semi-structured", "Tidak terstruktur": "Unstructured",
    "Identitas": "Identity", "Label": "Label", "Deskripsi": "Description", "Dimensi": "Dimension",
    "Temporal": "Temporal", "Metrik": "Metric", "Popularitas": "Popularity", "Flag": "Flag",
    "Tidak ada": "None", "Publik": "Public", "Nama orang (publik)": "Person name (published credit)",
    "Nama entitas (publik)": "Organisation name (public)", "Agregat (bukan PII)": "Aggregate (not PII)",
    "Agregat": "Aggregate",
}
#: Why each source column sits in its format class. Percentages live in the
#: "% placeholder" column next to it, so they are not repeated here.
ELEMENT_RATIONALE = {
    "series_id": "Natural primary key: a unique, atomic integer. Used for de-duplication because titles are not unique.",
    "url": "Hierarchical path (/series/<slug>/) that can be split into segments. The key that unifies the scrape, not an attribute.",
    "title": "Free natural-language text with no internal schema; tokenised before it enters the model corpus.",
    "genres": "Comma-separated list in one cell: fixed schema, non-atomic content. Split into an array before counting; the genre model's target.",
    "tags": "Like genres but much denser (median 12 tags): community-curated semantic labels.",
    "synopsis": "Free prose synopsis, the core unstructured data here; useful only after tokenisation and TF-IDF or embeddings.",
    "type": "Closed, low-cardinality category, one value per row.",
    "type_origin": "Country of origin: an atomic category and an aggregation axis.",
    "language": "Atomic category with very low cardinality.",
    "status_coo": "Mixes status text with number-and-unit patterns (\"182 Chapters (Completed)\"); the chapter count is recovered from it with a regex.",
    "authors": "Comma-separated list, as a title can have several authors. Holds published pen names, not private identities.",
    "original_publisher": "Organisation name, one per row; an unknown publisher becomes empty during cleaning.",
    "artists": "Dropped during cleaning: almost entirely Unknown.",
    "english_publisher": "Dropped during cleaning: almost entirely Unknown.",
    "year": "Release year, an ordinal scalar. The 0 placeholder cannot be recovered from other columns, so it becomes empty and leaves time analyses.",
    "release_frequency": "Sentence pattern (\"Every 0.9 Day(s)\") with number and unit inside the text; every row parses into days per chapter.",
    "chapter_count": "Additive numeric measure. The 0 placeholder becomes empty first, or the average length would drop falsely.",
    "rating_score": "Continuous 0 to 5 score. 0 means never rated, not rated badly, so it becomes empty.",
    "rating_votes": "Vote count, used as the confidence weight (v) in the Bayesian weighted rating.",
    "reading_list_count": "Aggregate count of readers who saved the title; no per-user rows, so no PII.",
    "activity_week_rank": "Weekly activity rank: ordinal, lower is more popular.",
    "activity_month_rank": "Monthly activity rank.",
    "activity_all_rank": "All-time activity rank.",
    "reading_list_month_rank": "Monthly reading-list rank.",
    "reading_list_all_rank": "All-time reading-list rank.",
    "is_completed": "Boolean stored as a number.",
    "licensed": "Boolean stored as a number; marks titles with an official English licence (compliance).",
    "completely_translated": "Boolean stored as a number.",
}


def page_governance() -> None:
    hero(
        "Data quality & governance",
        "What the pipeline checks before anything reaches a chart: format classes, the "
        "cleaning steps that change row counts, and the security, compliance, privacy and "
        "quality audit.",
    )

    with st.spinner("Loading audit tables…"):
        audit = load_lake("governance/audit_report")
        funnel = load_table("quality_report")
        elements = load_lake("governance/data_elements")
        formats = load_lake("governance/data_element_summary")

    audit = public_audit(audit.assign(status_en=audit["status"].map(STATUS_LABEL).fillna(audit["status"])))
    open_items = audit[audit["status_en"].isin(["Fail", "Warning"])]

    st.subheader("Needs attention")
    if open_items.empty:
        st.success(f"All {len(audit)} audit checks pass or are informational.")
    else:
        for row in open_items.itertuples(index=False):
            with st.container(border=True):
                st.markdown(
                    f'<span style="color:{STATUS_COLOR[row.status_en]};font-weight:650">'
                    f"{row.status_en}</span> · `{row.check_id}` · {html.escape(row.name_en)}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Evidence: {row.evidence_en}")
                if row.follow_en:
                    st.markdown(f"Follow-up: {row.follow_en}")

    st.write("")
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Audit result by pillar**")
        tally = audit.groupby(["pillar", "status_en"]).size().reset_index(name="checks")
        fig = px.bar(tally, x="checks", y="pillar", color="status_en", orientation="h",
                     color_discrete_map=STATUS_COLOR,
                     category_orders={"status_en": ["Pass", "Warning", "Fail", "Info"]},
                     labels={"checks": "Checks", "pillar": "", "status_en": ""})
        fig.update_yaxes(categoryorder="category descending")
        # Several statuses stack on one pillar: one tooltip lists them all.
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with right:
        st.markdown("**Every check**")
        pick = st.multiselect("Status", ["Pass", "Warning", "Fail", "Info"],
                              default=["Pass", "Warning", "Fail", "Info"])
        rows = audit[audit["status_en"].isin(pick)]
        if rows.empty:
            st.info("Select at least one status to list checks.")
        else:
            st.dataframe(
                rows[["check_id", "pillar", "status_en", "name_en", "evidence_en"]].rename(columns={
                    "check_id": "ID", "pillar": "Pillar", "status_en": "Status",
                    "name_en": "Check", "evidence_en": "Evidence"}),
                hide_index=True, use_container_width=True, height=300,
            )

    st.divider()
    st.subheader("Cleaning steps and row counts")
    main_line = ["1_raw_loaded", "2_identified", "3_sentinels_to_null", "4_deduplicated", "6_enriched"]
    steps = funnel.assign(order=funnel["step"].map(lambda x: main_line.index(x) if x in main_line else 10 + int(x[0])))
    steps = steps.sort_values("order").drop(columns="order").reset_index(drop=True)
    steps["step_label"] = steps["step"].str.replace(r"^\d+[a-z]?_", "", regex=True).str.replace("_", " ")
    steps["note"] = [FUNNEL_NOTES.get(step, lambda n: "")(note) for step, note in zip(steps["step"], steps["note"])]
    steps["order"] = range(len(steps))
    fig = px.bar(steps, x="rows", y="step_label", orientation="h", text="rows",
                 hover_data={"note": True, "order": False},
                 labels={"rows": "Rows after step", "step_label": ""})
    fig.update_traces(marker_color=BRAND, texttemplate="%{text:,}", textposition="outside",
                      cliponaxis=False)
    fig.update_yaxes(categoryorder="array", categoryarray=steps["step_label"].tolist()[::-1])
    st.plotly_chart(style_fig(fig, 40 * len(steps) + 60), use_container_width=True)
    st.caption(
        "Only duplicate series leave the catalogue; the later bars are subsets used by specific "
        "analyses, and 'reliable rating' keeps the titles with 10+ votes that enter rating "
        "averages. Hover a bar for the step note."
    )

    st.divider()
    st.subheader("Data elements by format class")
    st.caption("Profiled on the raw catalogue (novel_catalogue.csv) by data_catalog.py, before cleaning.")
    formats = formats.assign(format_class=formats["format_class"].map(ELEMENT_TERMS).fillna(formats["format_class"]))
    elements = elements.assign(
        **{col: elements[col].map(ELEMENT_TERMS).fillna(elements[col])
           for col in ("format_class", "domain", "sensitivity")},
        rationale=elements["name"].map(ELEMENT_RATIONALE).fillna(""),
    )
    fmt_cols = st.columns(len(formats))
    for col, row in zip(fmt_cols, formats.itertuples(index=False)):
        with col:
            st.markdown(f"**{row.format_class}** · {row.n_elements} elements")
            st.caption(row.elements)
    pick_fmt = st.selectbox("Show elements of", ["All"] + formats["format_class"].tolist())
    shown = elements if pick_fmt == "All" else elements[elements["format_class"] == pick_fmt]
    st.dataframe(
        shown[["name", "format_class", "domain", "physical_type", "pct_sentinel",
               "sensitivity", "rationale"]].rename(columns={
                   "name": "Element", "format_class": "Format", "domain": "Domain",
                   "physical_type": "Type", "pct_sentinel": "% placeholder",
                   "sensitivity": "Sensitivity", "rationale": "Why this class"}),
        hide_index=True, use_container_width=True, height=360,
        column_config={"% placeholder": st.column_config.NumberColumn(format="%.2f")},
    )


# PAGE — Market Trends
def page_trends() -> None:
    hero(
        "Market trends",
        "How the catalogue changes by release year: where new titles come from, which genres "
        "are gaining ground, and whether newer stories run shorter or score differently.",
    )
    viz = load_table("viz_novels")
    dated = viz.dropna(subset=["year"]).assign(year=lambda d: d["year"].astype(int))
    if dated.empty:
        st.info("No title in the catalogue has a release year yet.")
        return
    last_full = last_full_year(dated["year"])
    # A handful of classics (one from 1740) would stretch the slider across
    # centuries with nothing in them; start where releases become regular.
    per_year_all = dated["year"].value_counts()
    first = int(per_year_all[per_year_all >= 5].index.min())
    early = int((dated["year"] < first).sum())

    c1, c2 = st.columns([3, 2], gap="large")
    span = c1.slider("Release years", first, last_full, (max(first, last_full - 19), last_full))
    origins = viz["type_origin"].value_counts().index.tolist()
    picked = c2.multiselect("Country of origin", origins, placeholder="All origins")

    data = dated[dated["year"].between(*span)]
    if picked:
        data = data[data["type_origin"].isin(picked)]
    if data.empty:
        st.info("No titles match these filters. Widen the year range or clear the origin filter.")
        return
    st.caption(
        f"{len(data):,} titles in view. {viz['year'].isna().mean():.0%} of the catalogue has no "
        f"release year and is left out; {last_full + 1} onward is excluded while the year is still running"
        + (f", and {early} titles from before {first} fall outside the slider." if early else ".")
    )

    end = span[1]
    yearly = data.groupby("year").size().reindex(range(span[0], end + 1), fill_value=0)
    recent, older = data[data["year"] > end - 5], data[data["year"] <= end - 5]

    kpi = st.columns(4)
    cur, prev = int(yearly.get(end, 0)), int(yearly.get(end - 1, 0))
    card(kpi[0], f"Released in {end}", f"{cur:,}", "titles now in the catalogue",
         delta=((cur - prev) / prev, f"{(cur - prev) / prev:+.0%} vs {end - 1}") if prev else None)
    card(kpi[1], "Busiest year", str(int(yearly.idxmax())), f"{int(yearly.max()):,} titles released")
    lead = recent["type_origin"].value_counts(normalize=True)
    if len(lead):
        before = older["type_origin"].value_counts(normalize=True).get(lead.index[0], np.nan)
        card(kpi[2], f"Leading origin {end - 4} to {end}", lead.index[0], f"{lead.iloc[0]:.0%} of new titles",
             delta=(lead.iloc[0] - before, f"{(lead.iloc[0] - before) * 100:+.1f} pts vs earlier years")
             if pd.notna(before) else None)
    med_now, med_before = recent["chapter_count"].median(), older["chapter_count"].median()
    card(kpi[3], "Median length, last 5 years", f"{med_now:,.0f} ch" if pd.notna(med_now) else "n/a",
         "recent titles are partly still running",
         delta=(med_now - med_before, f"{med_now - med_before:+,.0f} vs earlier years")
         if pd.notna(med_now) and pd.notna(med_before) else None)

    st.write("")
    st.subheader("New titles per release year")
    by_origin = (
        data.assign(origin=origin_group(data["type_origin"]))
        .groupby(["year", "origin"]).size().reset_index(name="titles")
    )
    fig = px.bar(
        by_origin, x="year", y="titles", color="origin", color_discrete_map=ORIGIN_COLORS,
        category_orders={"origin": list(ORIGIN_COLORS)},
        labels={"year": "", "titles": "Titles", "origin": ""},
    )
    fig.update_layout(hovermode="x unified", bargap=0.18)
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.markdown(
        '<div class="nl-note">The catalogue lists what has been translated or picked up so far, '
        "so the newest years keep growing as translations catch up. Read the last bars as a floor.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Which genres are gaining ground")
    if end - span[0] < 9:
        st.info("Widen the year range to at least ten years to compare two five-year windows.")
    else:
        genres = genre_labels().merge(data[["series_id", "year"]], on="series_id")
        n_now = int(data["year"].between(end - 4, end).sum())
        n_before = int(data["year"].between(end - 9, end - 5).sum())
        share_now = genres[genres["year"].between(end - 4, end)].groupby("genre", observed=True).size() / n_now
        share_before = (genres[genres["year"].between(end - 9, end - 5)]
                        .groupby("genre", observed=True).size() / n_before)
        momentum = pd.DataFrame({"now": share_now, "before": share_before}).fillna(0)
        momentum = momentum[(momentum["now"] + momentum["before"]) >= 0.02]
        momentum["change"] = (momentum["now"] - momentum["before"]) * 100
        top = (momentum.reindex(momentum["change"].abs().sort_values(ascending=False).index)
               .head(14).sort_values("change").reset_index(names="genre"))
        top["direction"] = np.where(top["change"] > 0, "Gaining share", "Losing share")
        top["now_pct"], top["before_pct"] = top["now"] * 100, top["before"] * 100

        left, right = st.columns([2, 3], gap="large")
        with left:
            fig = px.bar(
                top, x="change", y="genre", orientation="h", color="direction",
                color_discrete_map={"Gaining share": ACCENT, "Losing share": "#FF6B9D"},
                text=top["change"].map("{:+.1f}".format),
                hover_data={"now_pct": ":.1f", "before_pct": ":.1f", "direction": False, "change": False},
                labels={"change": "Change in share of new titles (pts)", "genre": "",
                        "direction": "", "now_pct": f"% of {end - 4} to {end} titles",
                        "before_pct": f"% of {end - 9} to {end - 5} titles"},
            )
            # Symmetric room on both sides so the outside labels of the longest
            # bars stay inside the plot instead of running into the genre names.
            reach = top["change"].abs().max() * 1.45
            fig.update_xaxes(range=[-reach, reach])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption(f"Share of titles released {end - 4} to {end} ({n_now:,}) against "
                       f"{end - 9} to {end - 5} ({n_before:,}). Genres under 2% in both windows are hidden.")
        with right:
            years_shown = list(range(max(span[0], end - 14), end + 1))
            window = genres[genres["year"].isin(years_shown)]
            per_year = data[data["year"].isin(years_shown)].groupby("year").size()
            top_genres = window["genre"].astype(str).value_counts().head(12).index
            heat = (window[window["genre"].isin(top_genres)]
                    .groupby(["genre", "year"], observed=True).size().unstack(fill_value=0))
            heat = (heat.div(per_year, axis=1) * 100).reindex(top_genres)
            heat.columns = [str(c) for c in heat.columns]
            fig = px.imshow(heat, color_continuous_scale=SCALE, aspect="auto", text_auto=".0f",
                            labels=dict(x="", y="", color="% of titles"))
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption("Share of each year's new titles that carry the genre, for the 12 largest genres. "
                       "A title with several genres counts in each row.")

    st.write("")
    st.subheader("Length, completion and reception by release year")
    per_year = data.groupby("year").agg(
        titles=("series_id", "size"), median_chapters=("chapter_count", "median"),
        completed=("is_completed", "mean"),
    ).reset_index()
    per_year = per_year[per_year["titles"] >= 20]
    rated = (data[data["is_rated_reliable"] == True]  # noqa: E712
             .groupby("year")["weighted_rating"].agg(["mean", "size"]).reset_index())
    rated = rated[rated["size"] >= 30]

    a, b, c = st.columns(3, gap="large")
    for col, frame, y, title, fmt in (
        (a, per_year, "median_chapters", "Median chapters", ",.0f"),
        (b, per_year.assign(completed=per_year["completed"] * 100), "completed", "Completed (% of titles)", ".0f"),
        (c, rated, "mean", "Average rating (10+ votes)", ".2f"),
    ):
        with col:
            st.markdown(f"**{title}**")
            if frame.empty:
                st.info("Too few titles per year in this view. Widen the years or clear the origin filter.")
                continue
            fig = px.line(frame, x="year", y=y, markers=True, labels={"year": "", y: title})
            fig.update_traces(line=dict(color=BRAND, width=2), marker=dict(size=8, color=ACCENT),
                              hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>")
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.markdown(
        '<div class="nl-note">Years with fewer than 20 titles (30 rated titles for the rating line) '
        "are hidden. Newer titles are still being written, which pulls both their chapter count and "
        "their completion rate down; older titles that are still listed are the ones readers kept, "
        "which can lift their average rating.</div>",
        unsafe_allow_html=True,
    )


# PAGE — Genre Explorer
def page_genre() -> None:
    hero(
        "Genre explorer",
        "One genre at a time: how it performs against the whole catalogue, which story elements "
        "define it, what it is paired with, and who writes it.",
    )
    viz = load_table("viz_novels")
    genres = genre_labels()
    counts = genres["genre"].astype(str).value_counts()
    pick = st.selectbox("Genre", counts.index.tolist(), format_func=lambda g: f"{g} ({counts[g]:,} titles)")

    ids = genres.loc[genres["genre"] == pick, "series_id"]
    sub = viz[viz["series_id"].isin(ids)]
    reliable = viz["is_rated_reliable"] == True  # noqa: E712
    rel_all, rel = viz[reliable], sub[sub["is_rated_reliable"] == True]  # noqa: E712
    n, rank = len(sub), counts.index.get_loc(pick) + 1

    kpi = st.columns(4)
    card(kpi[0], "Titles", f"{n:,}", f"{n / len(viz):.1%} of the catalogue · #{rank} of {len(counts)} genres")
    if len(rel):
        diff = rel["weighted_rating"].mean() - rel_all["weighted_rating"].mean()
        card(kpi[1], "Average rating", f"{rel['weighted_rating'].mean():.2f}",
             f"{len(rel):,} titles with 10+ votes", delta=(diff, f"{diff:+.2f} vs catalogue"))
    else:
        card(kpi[1], "Average rating", "n/a", "no title with 10+ votes")
    done, done_all = sub["is_completed"].mean(), viz["is_completed"].mean()
    card(kpi[2], "Completed", f"{done:.0%}", f"catalogue {done_all:.0%}",
         delta=(done - done_all, f"{(done - done_all) * 100:+.0f} pts vs catalogue"))
    med, med_all = sub["chapter_count"].median(), viz["chapter_count"].median()
    card(kpi[3], "Median length", f"{med:,.0f} ch" if pd.notna(med) else "n/a", f"catalogue {med_all:,.0f} chapters",
         delta=(med - med_all, f"{med - med_all:+,.0f} vs catalogue") if pd.notna(med) else None)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Rating distribution** (titles with 10+ votes)")
        if rel.empty:
            st.info("No title in this genre has 10+ votes yet.")
        else:
            edges = np.round(np.arange(1.0, 5.05, 0.1), 2)
            mine = np.histogram(rel["weighted_rating"], edges)[0] / len(rel) * 100
            every = np.histogram(rel_all["weighted_rating"], edges)[0] / len(rel_all) * 100
            used = np.flatnonzero((mine + every) > 0.05)
            lo, hi = used.min(), used.max() + 1
            centers = (edges[:-1] + 0.05)[lo:hi]
            fig = go.Figure()
            fig.add_bar(x=centers, y=mine[lo:hi], name=pick, marker_color=BRAND,
                        hovertemplate="%{y:.1f}% of titles<extra>" + html.escape(pick) + "</extra>")
            fig.add_scatter(x=centers, y=every[lo:hi], name="Whole catalogue", mode="lines",
                            line=dict(color="#FFB454", width=2, shape="hvh"),
                            hovertemplate="%{y:.1f}% of titles<extra>Catalogue</extra>")
            fig.update_layout(bargap=0.08, hovermode="x unified", xaxis_title="Weighted rating",
                              yaxis_title="% of titles", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        st.markdown("**Where the titles come from**")
        mix = pd.DataFrame({
            pick: origin_group(sub["type_origin"]).value_counts(normalize=True) * 100,
            "Whole catalogue": origin_group(viz["type_origin"]).value_counts(normalize=True) * 100,
        }).reindex(list(ORIGIN_COLORS)).fillna(0)
        mix = mix[mix.max(axis=1) >= 0.5].reset_index(names="origin")
        long = mix.melt(id_vars="origin", var_name="group", value_name="share")
        fig = px.bar(long, x="share", y="origin", color="group", barmode="group", orientation="h",
                     color_discrete_map={pick: BRAND, "Whole catalogue": "#8A90A6"},
                     text=long["share"].map("{:.0f}%".format),
                     labels={"share": "% of titles", "origin": "", "group": ""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(categoryorder="array", categoryarray=list(ORIGIN_COLORS)[::-1])
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Signature story elements**")
        tags = tag_labels()
        in_genre = tags[tags["series_id"].isin(ids)].groupby("tag", observed=True).size()
        overall = tags.groupby("tag", observed=True).size()
        sig = pd.DataFrame({"titles": in_genre, "share": in_genre / n * 100,
                            "lift": (in_genre / n) / (overall.reindex(in_genre.index) / len(viz))})
        sig = sig[sig["titles"] >= max(15, 0.02 * n)].nlargest(15, "lift").sort_values("lift")
        if sig.empty:
            st.info("Too few titles to find tags that set this genre apart.")
        else:
            sig = sig.reset_index(names="tag")
            fig = px.bar(sig, x="lift", y="tag", orientation="h", text=sig["lift"].map("×{:.1f}".format),
                         hover_data={"titles": ":,", "share": ":.1f"},
                         labels={"lift": "Times more common than in the catalogue", "tag": "",
                                 "titles": "Titles", "share": "% of genre"})
            fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
            st.caption("Lift = share of this genre's titles with the tag ÷ share of all titles with it. "
                       "Tags on fewer than 2% of the genre are skipped.")
    with right:
        st.markdown("**Most often paired with**")
        pair = (genres[genres["series_id"].isin(ids) & (genres["genre"] != pick)]
                .groupby("genre", observed=True).size().nlargest(10) / n * 100)
        pair = pair.sort_values().reset_index(name="share")
        fig = px.bar(pair, x="share", y="genre", orientation="h", text=pair["share"].map("{:.0f}%".format),
                     labels={"share": f"% of {pick} titles that also carry the genre", "genre": ""})
        fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Highest-rated titles** (10+ votes)")
        best = rel.nlargest(15, "weighted_rating")[["title", "weighted_rating", "rating_votes", "type_origin", "url"]]
        st.dataframe(
            best.rename(columns={"title": "Title", "weighted_rating": "Rating", "rating_votes": "Votes",
                                 "type_origin": "Origin", "url": "Link"}),
            hide_index=True, use_container_width=True, height=420,
            column_config={
                "Title": st.column_config.TextColumn(width="large"),
                "Rating": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Votes": st.column_config.NumberColumn(format="%d", width="small"),
                "Origin": st.column_config.TextColumn(width="small"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open", width="small"),
            },
        )
    with right:
        st.markdown("**Most prolific authors in this genre**")
        authors = author_labels()
        written = authors[authors["series_id"].isin(ids)].merge(
            viz[["series_id", "weighted_rating", "is_rated_reliable"]], on="series_id")
        written["rated"] = written["weighted_rating"].where(written["is_rated_reliable"] == True)  # noqa: E712
        top_auth = (written.groupby("author", observed=True)
                    .agg(titles=("series_id", "nunique"), rating=("rated", "mean"))
                    .nlargest(12, "titles").reset_index())
        st.dataframe(
            top_auth.assign(rating=rating_text(top_auth["rating"]))
            .rename(columns={"author": "Author", "titles": "Titles", "rating": "Rating"}),
            hide_index=True, use_container_width=True, height=420,
        )


# PAGE — Story Elements
@st.cache_data(show_spinner=False, max_entries=16)
def tag_stats(genre: str, origins: tuple, recent_only: bool, version: str) -> tuple[pd.DataFrame, float, int]:
    """Per-tag title count, mean rating and median reading-list adds over titles with 10+ votes."""
    viz = _load_lake("analytics/viz_novels", version)
    base = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    if origins:
        base = base[base["type_origin"].isin(origins)]
    if recent_only:
        base = base[base["year"] >= last_full_year(viz["year"].dropna()) - 4]
    if genre:
        genres = _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", version)
        base = base[base["series_id"].isin(genres.loc[genres["genre"] == genre, "series_id"])]
    tags = _load_labels("bridge_novel_tag", "dim_tag", "tag_key", "tag", version)
    joined = tags.merge(base[["series_id", "weighted_rating", "reading_list_count"]], on="series_id")
    stats = (joined.groupby("tag", observed=True)
             .agg(titles=("series_id", "size"), rating=("weighted_rating", "mean"),
                  readers=("reading_list_count", "median"))
             .reset_index())
    stats["tag"] = stats["tag"].astype(str)
    return stats, float(base["weighted_rating"].mean()), len(base)


def spaced_labels(stats: pd.DataFrame, limit: int = 9) -> pd.Series:
    """Labels for the best underserved and proven tags, skipping points too close
    (in log10 titles and lift) to one already labelled."""
    candidates = pd.concat([
        stats[stats["quadrant"] == "Underserved"].nlargest(20, "lift"),
        stats[stats["quadrant"] == "Proven"].nlargest(10, "lift"),
    ])
    x_span = np.log10(stats["titles"].max()) - np.log10(stats["titles"].min()) or 1.0
    y_span = (stats["lift"].max() - stats["lift"].min()) or 1.0
    placed: list[tuple[float, float]] = []
    chosen = set()
    for row in candidates.itertuples(index=False):
        x, y = np.log10(row.titles) / x_span, row.lift / y_span
        if all(abs(x - px_) > 0.12 or abs(y - py_) > 0.07 for px_, py_ in placed):
            placed.append((x, y))
            chosen.add(row.tag)
        if len(chosen) == limit:
            break
    return stats["tag"].where(stats["tag"].isin(chosen), "")


QUADRANT_COLORS = {"Underserved": ACCENT, "Proven": BRAND, "Crowded": "#FFB454", "Rare and weaker": "#8A90A6"}


def page_tags() -> None:
    hero(
        "Story element opportunities",
        "Every tag placed on two axes: how many rated titles already use it, and how readers "
        "score those titles against the average. Elements that score well but are still rare "
        "are the openings for a new story.",
    )
    viz = load_table("viz_novels")
    genre_counts = genre_labels()["genre"].astype(str).value_counts()

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    genre = c1.selectbox("Within genre", ["All genres"] + genre_counts.index.tolist())
    origins = c2.multiselect("Country of origin", viz["type_origin"].value_counts().index.tolist(),
                             placeholder="All origins")
    period = c3.radio("Release years", ["All years", "Last 5 years"], horizontal=True)
    min_titles = c4.slider("Minimum titles per tag", 10, 300, 40, 10)

    stats, average, n_base = tag_stats(
        "" if genre == "All genres" else genre, tuple(sorted(origins)), period == "Last 5 years", lake_version()
    )
    stats = stats[stats["titles"] >= min_titles].copy()
    if len(stats) < 5:
        st.info(f"Only {len(stats)} tags reach {min_titles} rated titles in this view. "
                "Lower the minimum or widen the filters.")
        return

    stats["lift"] = stats["rating"] - average
    supply = stats["titles"].median()
    stats["quadrant"] = np.select(
        [(stats["lift"] > 0) & (stats["titles"] < supply), stats["lift"] > 0, stats["titles"] >= supply],
        ["Underserved", "Proven", "Crowded"], "Rare and weaker",
    )
    stats["label"] = spaced_labels(stats)

    kpi = st.columns(4)
    card(kpi[0], "Rated titles in view", f"{n_base:,}", f"average rating {average:.2f}")
    card(kpi[1], "Tags compared", f"{len(stats):,}", f"each on {min_titles}+ rated titles")
    for col, name, text in ((kpi[2], "Underserved", "rated above average, used less than the median tag"),
                            (kpi[3], "Proven", "rated above average and already common")):
        card(col, name, f"{(stats['quadrant'] == name).sum():,} tags", text)

    st.write("")
    fig = px.scatter(
        stats, x="titles", y="lift", color="quadrant", text="label", log_x=True,
        color_discrete_map=QUADRANT_COLORS, category_orders={"quadrant": list(QUADRANT_COLORS)},
        hover_name="tag", hover_data={"titles": ":,", "rating": ":.2f", "readers": ":,.0f",
                                      "lift": ":+.2f", "label": False, "quadrant": False},
        labels={"titles": "Rated titles using the tag (log scale)", "lift": "Rating vs average",
                "quadrant": "", "rating": "Average rating", "readers": "Median reading-list adds"},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1.5, color="#0D1117")),
                      textposition="top center", textfont=dict(size=11, color="#D5DAE5"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,.35)", width=1))
    fig.add_vline(x=supply, line=dict(color="rgba(255,255,255,.35)", width=1, dash="dot"))
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(style_fig(fig, 560), use_container_width=True)
    st.markdown(
        f'<div class="nl-note">Horizontal line: the average rating of the titles in view. Dotted line: '
        f"the median tag ({supply:,.0f} titles). A high score on a rare tag rests on fewer titles, so "
        "check the title count before reading much into the top-left corner.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Underserved elements, best first**")
        under = stats[stats["quadrant"] == "Underserved"].nlargest(15, "lift")
        st.dataframe(
            under[["tag", "titles", "rating", "lift", "readers"]].rename(columns={
                "tag": "Tag", "titles": "Rated titles", "rating": "Rating", "lift": "vs average",
                "readers": "Median readers"}),
            hide_index=True, use_container_width=True, height=420,
            column_config={"Rating": st.column_config.NumberColumn(format="%.2f"),
                           "vs average": st.column_config.NumberColumn(format="%+.2f"),
                           "Median readers": st.column_config.NumberColumn(format="%d")},
        )
    with right:
        st.markdown("**Check a combination**")
        combo = st.multiselect("Tags that must all be present",
                               stats.sort_values("titles", ascending=False)["tag"].tolist(),
                               max_selections=3, placeholder="Pick up to three tags")
        if combo:
            tags = tag_labels()
            have = tags[tags["tag"].isin(combo)].groupby("series_id")["tag"].nunique()
            match_ids = have[have == len(combo)].index
            hits = viz[viz["series_id"].isin(match_ids)]
            rated = hits[hits["is_rated_reliable"] == True]  # noqa: E712
            a, b = st.columns(2)
            card(a, "Titles with all", f"{len(hits):,}", f"{len(rated):,} with 10+ votes")
            if len(rated):
                diff = rated["weighted_rating"].mean() - average
                card(b, "Their rating", f"{rated['weighted_rating'].mean():.2f}", "",
                     delta=(diff, f"{diff:+.2f} vs average"))
            st.write("")
            st.dataframe(
                rated.nlargest(8, "weighted_rating")[["title", "weighted_rating", "url"]].rename(
                    columns={"title": "Title", "weighted_rating": "Rating", "url": "Link"}),
                hide_index=True, use_container_width=True,
                column_config={"Rating": st.column_config.NumberColumn(format="%.2f"),
                               "Link": st.column_config.LinkColumn("Link", display_text="Open")},
            )
        else:
            st.caption("Pick tags to see how many titles already combine them and how those score.")


# PAGE — Title Lookup
@st.cache_data(show_spinner=False, max_entries=2)
def _tag_counts(version: str) -> pd.Series:
    return _load_lake("model/bridge_novel_tag", version).groupby("series_id").size()


def tag_neighbours(series_id: int, k: int = 6) -> pd.DataFrame:
    """Titles sharing the most tags with `series_id`, by Jaccard overlap of their tag sets."""
    version = lake_version()
    bridge = _load_lake("model/bridge_novel_tag", version)
    mine = bridge.loc[bridge["series_id"] == series_id, "tag_key"]
    if mine.empty:
        return pd.DataFrame()
    shared = bridge[bridge["tag_key"].isin(mine)].groupby("series_id").size().drop(series_id, errors="ignore")
    shared = shared[shared >= 3]
    sizes = _tag_counts(version).reindex(shared.index)
    viz = load_table("viz_novels").set_index("series_id")
    out = pd.DataFrame({"similarity": shared / (len(mine) + sizes - shared)}).join(viz, how="inner")
    out = out.sort_values(["similarity", "rating_votes"], ascending=False).head(k).reset_index()
    return out.assign(genre_list=out["series_id"].map(genre_lists()))


def page_lookup() -> None:
    viz = load_table("viz_novels")
    hero(
        "Title lookup",
        f"Search any of the {len(viz):,} titles to see how it compares with its genre and which "
        "titles share the most story elements with it.",
    )
    c1, c2 = st.columns([2, 3], gap="large")
    query = c1.text_input("Search by title", placeholder="Part of a title, e.g. martial god")
    pool = viz
    if query.strip():
        pool = viz[viz["title"].str.contains(re.escape(query.strip()), case=False, na=False)]
    if pool.empty:
        st.info("No title matches that search. Try a shorter part of the name.")
        return
    pool = pool.sort_values("rating_votes", ascending=False).head(60)
    labels = {
        int(r.series_id): f"{r.title} ({int(r.year) if pd.notna(r.year) else 'year n/a'}, {r.type_origin})"
        for r in pool.itertuples(index=False)
    }
    sid = c2.selectbox(f"{len(pool)} best-known matches", list(labels), format_func=labels.get)
    row = viz.loc[viz["series_id"] == sid].iloc[0]

    tags = tag_labels()
    title_tags = tags.loc[tags["series_id"] == sid, "tag"].astype(str).tolist()
    title_genres = genre_lists().get(sid, [])
    authors = author_labels()
    names = authors.loc[authors["series_id"] == sid, "author"].astype(str).tolist()

    st.markdown(f"### {html.escape(str(row['title']))}", unsafe_allow_html=True)
    st.markdown(chips(title_genres, active=True), unsafe_allow_html=True)
    meta = [f"by {', '.join(names)}" if names else "author unknown",
            f"released {int(row['year'])}" if pd.notna(row["year"]) else "release year unknown",
            "completed" if row["is_completed"] == 1 else "ongoing",
            f"release pace {en_label(row['release_pace']).lower()}" if isinstance(row["release_pace"], str) else ""]
    st.caption(" · ".join(m for m in meta if m))
    if isinstance(row["url"], str) and row["url"].startswith("http"):
        st.markdown(f'<a class="nl-open" href="{html.escape(row["url"], quote=True)}" target="_blank" '
                    'rel="noopener noreferrer">Open on NovelUpdates</a>', unsafe_allow_html=True)

    st.write("")
    kpi = st.columns(4)
    reliable = bool(row["is_rated_reliable"])
    card(kpi[0], "Rating", f"{row['weighted_rating']:.2f}" if pd.notna(row["weighted_rating"]) else "n/a",
         f"{int(row['rating_votes']):,} votes" + ("" if reliable else ", too few for rankings"))
    card(kpi[1], "On reading lists", f"{row['reading_list_count']:,.0f}" if pd.notna(row["reading_list_count"]) else "n/a",
         "readers who saved it")
    card(kpi[2], "Chapters", f"{row['chapter_count']:,.0f}" if pd.notna(row["chapter_count"]) else "n/a",
         en_label(row["length_bucket"]) if isinstance(row["length_bucket"], str) else "length unknown")
    card(kpi[3], "Story elements", f"{len(title_tags)}", "tags on NovelUpdates")

    st.write("")
    left, right = st.columns([2, 3], gap="large")
    peers = viz[viz["primary_genre"] == row["primary_genre"]]
    with left:
        st.markdown(f"**Against other {html.escape(str(row['primary_genre']))} titles**")
        checks = [("Rating", "weighted_rating", True), ("Votes", "rating_votes", False),
                  ("Reading-list adds", "reading_list_count", False), ("Chapters", "chapter_count", False)]
        rows = []
        for name, col, need_reliable in checks:
            if pd.isna(row[col]) or (need_reliable and not reliable):
                continue
            pool_vals = peers[peers["is_rated_reliable"] == True][col] if need_reliable else peers[col]  # noqa: E712
            pool_vals = pool_vals.dropna()
            if len(pool_vals):
                # Share at or above this title: the title itself counts, so the
                # best title reads "top 0.1%", never "higher than 100%".
                top = (pool_vals >= row[col]).mean() * 100
                rows.append({"measure": name, "pct": 100 - top, "n": len(pool_vals),
                             "text": f"top {max(top, 0.1):.1f}%" if top < 10 else f"top {top:.0f}%"})
        if not rows:
            st.info("Not enough data to compare this title with its genre.")
        else:
            pct = pd.DataFrame(rows)
            fig = px.bar(pct, x="pct", y="measure", orientation="h", range_x=[0, 100],
                         text="text",
                         hover_data={"n": ":,"}, labels={"pct": "Percentile within genre", "measure": "",
                                                         "n": "Titles compared"})
            fig.update_traces(marker_color=BRAND, textposition="inside", insidetextanchor="end")
            fig.update_yaxes(categoryorder="array", categoryarray=[c[0] for c in checks][::-1])
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
            st.caption(f"Compared with {len(peers):,} titles whose primary genre is {row['primary_genre']}; "
                       "the rating only against titles with 10+ votes.")
    with right:
        st.markdown("**Story elements**")
        if title_tags:
            rarity = tags["tag"].astype(str).value_counts()
            ordered = sorted(title_tags, key=lambda t: rarity.get(t, 0))
            st.markdown(chips(ordered[:24]), unsafe_allow_html=True)
            st.caption("Rarest first: the tags near the start set this title apart most.")
        else:
            st.info("This title has no tags on NovelUpdates.")

    st.write("")
    st.subheader("Titles built from the same story elements")
    near = tag_neighbours(sid)
    if near.empty:
        st.info("No other title shares at least three tags with this one.")
    else:
        for _, other in near.iterrows():
            novel_card(other, similarity_label="tags shared", show_synopsis=False)
        st.caption("Overlap = shared tags ÷ tags on either title (Jaccard). Unlike AI Studio, this "
                   "works for every title, including those without a synopsis.")


# PAGE — Authors & Publishers
def page_creators() -> None:
    hero(
        "Authors & publishers",
        "Who writes and publishes the catalogue, measured by output, reader reception and reach.",
    )
    viz = load_table("viz_novels")
    cols = ["series_id", "weighted_rating", "is_rated_reliable", "rating_votes", "reading_list_count", "type_origin"]
    tab_a, tab_p = st.tabs(["Authors", "Original publishers"])

    with tab_a:
        written = author_labels().merge(viz[cols], on="series_id")
        written["rated"] = written["weighted_rating"].where(written["is_rated_reliable"] == True)  # noqa: E712
        per = written.groupby("author", observed=True).agg(
            titles=("series_id", "nunique"), rated_titles=("rated", "count"), rating=("rated", "mean"),
            readers=("reading_list_count", "sum"), votes=("rating_votes", "sum"))
        origin = (written.groupby(["author", "type_origin"], observed=True).size().reset_index(name="n")
                  .sort_values("n").drop_duplicates("author", keep="last").set_index("author")["type_origin"])
        per = per.join(origin.rename("origin")).reset_index()
        per["author"] = per["author"].astype(str)

        kpi = st.columns(4)
        card(kpi[0], "Authors", f"{len(per):,}", "pen names merged where aliases match")
        card(kpi[1], "One title only", f"{(per['titles'] == 1).mean():.0%}", "of authors in the catalogue")
        top = per.nlargest(1, "titles").iloc[0]
        card(kpi[2], "Most titles", top["author"], f"{int(top['titles'])} titles")
        card(kpi[3], "Median output", f"{per['titles'].median():.0f}", "titles per author")

        st.write("")
        c1, c2 = st.columns(2)
        rank_by = c1.radio("Rank by", ["Titles", "Average rating", "Reading-list adds"], horizontal=True)
        min_titles = c2.slider("Minimum titles", 1, 20, 3, key="author_min")
        pool = per[per["titles"] >= min_titles]
        metric = {"Titles": "titles", "Average rating": "rating", "Reading-list adds": "readers"}[rank_by]
        if metric == "rating":
            pool = pool[pool["rated_titles"] >= 3]
        if pool.empty:
            st.info("No author meets the minimum. Lower it.")
        else:
            board = pool.nlargest(15, metric).sort_values(metric)
            board = board.assign(origin_group=origin_group(board["origin"]))
            # Ratings differ by hundredths, so they get dots on a zoomed axis
            # (see dot_rating_chart); counts keep bars from zero.
            chart = px.scatter if metric == "rating" else px.bar
            fig = chart(board, x=metric, y="author", color="origin_group", color_discrete_map=ORIGIN_COLORS,
                        category_orders={"origin_group": list(ORIGIN_COLORS)},
                        hover_data={"titles": True, "rating": ":.2f", "readers": ":,.0f", "origin_group": False},
                        labels={metric: rank_by, "author": "", "origin_group": "", "titles": "Titles",
                                "rating": "Rating", "readers": "Reading-list adds"},
                        **({} if metric == "rating" else {"orientation": "h"}))
            fig.update_yaxes(categoryorder="array", categoryarray=board["author"].tolist())
            if metric == "rating":
                fig.update_traces(marker=dict(size=13, line=dict(width=2, color="#0D1117")))
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
            if metric == "rating":
                st.caption("Authors need 3+ titles with 10+ votes to be ranked by rating.")
            st.dataframe(
                pool.sort_values(metric, ascending=False).head(100)
                .assign(rating=lambda d: rating_text(d["rating"]))[
                    ["author", "titles", "rated_titles", "rating", "readers", "origin"]].rename(columns={
                    "author": "Author", "titles": "Titles", "rated_titles": "Rated titles",
                    "rating": "Rating", "readers": "Reading-list adds", "origin": "Origin"}),
                hide_index=True, use_container_width=True, height=360,
                column_config={"Reading-list adds": st.column_config.NumberColumn(format="%d")},
            )

    with tab_p:
        fact = load_lake("model/fact_novel")[["series_id", "publisher_key"]]
        names = load_lake("model/dim_publisher")[["publisher_key", "publisher"]]
        books = fact.merge(names, on="publisher_key").merge(viz[cols], on="series_id")
        books = books[books["publisher"].notna() & (books["publisher"].str.strip() != "")]
        books["rated"] = books["weighted_rating"].where(books["is_rated_reliable"] == True)  # noqa: E712
        per = books.groupby("publisher").agg(
            titles=("series_id", "nunique"), rated_titles=("rated", "count"), rating=("rated", "mean"),
            readers=("reading_list_count", "sum"))
        origin = (books.groupby(["publisher", "type_origin"]).size().reset_index(name="n")
                  .sort_values("n").drop_duplicates("publisher", keep="last").set_index("publisher")["type_origin"])
        per = per.join(origin.rename("origin")).reset_index()

        min_pub = st.slider("Minimum titles", 5, 200, 30, 5, key="pub_min")
        pool = per[(per["titles"] >= min_pub) & (per["rated_titles"] >= 10)].copy()
        if len(pool) < 2:
            st.info("Too few publishers at this minimum. Lower it.")
        else:
            pool["origin_group"] = origin_group(pool["origin"])
            pool["label"] = np.where(pool["publisher"].isin(pool.nlargest(8, "titles")["publisher"]),
                                     pool["publisher"], "")
            fig = px.scatter(
                pool, x="titles", y="rating", size="readers", size_max=38, log_x=True,
                color="origin_group", color_discrete_map=ORIGIN_COLORS, text="label",
                hover_name="publisher", hover_data={"label": False, "origin_group": False, "readers": ":,.0f"},
                labels={"titles": "Titles published (log scale)", "rating": "Average rating (10+ votes)",
                        "origin_group": "", "readers": "Reading-list adds"},
            )
            fig.update_traces(textposition="top center", textfont=dict(size=11, color="#D5DAE5"))
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
            st.caption(f"{len(pool):,} publishers with {min_pub}+ titles and 10+ rated titles. "
                       "Bubble size = total reading-list adds.")
            st.dataframe(
                pool.sort_values("titles", ascending=False)
                .assign(rating=lambda d: rating_text(d["rating"]))[
                    ["publisher", "titles", "rated_titles", "rating", "readers", "origin"]].rename(columns={
                    "publisher": "Publisher", "titles": "Titles", "rated_titles": "Rated titles",
                    "rating": "Rating", "readers": "Reading-list adds", "origin": "Origin"}),
                hide_index=True, use_container_width=True, height=360,
                column_config={"Reading-list adds": st.column_config.NumberColumn(format="%d")},
            )


# PAGE — Lake Operations
ZONE_COLORS = {"Raw": "#FFB454", "Clean": "#4CC9F0", "Curated": BRAND}


@st.cache_data(ttl=60, show_spinner=False)
def lake_inventory() -> pd.DataFrame:
    """Every object in the three lake zones, grouped into datasets (the first two path levels)."""
    zones = {"Raw": SETTINGS.minio.bucket_raw, "Clean": SETTINGS.minio.bucket_clean,
             "Curated": SETTINGS.minio.bucket_curated}
    rows = []
    for zone, bucket in zones.items():
        try:
            objects = list_objects(bucket, "")
        except Exception as exc:
            raise DataUnavailable(str(exc)) from exc
        for obj in objects:
            key = obj["Key"]
            if "_temporary" in key:
                continue
            parts = key.split("/")
            rows.append({"zone": zone, "bucket": bucket, "key": key,
                         "dataset": "/".join(parts[:2]) if len(parts) > 2 else parts[0],
                         "size": obj["Size"], "modified": pd.Timestamp(obj["LastModified"])})
    return pd.DataFrame(rows)


def page_lake() -> None:
    hero(
        "Lake operations",
        "Where the data physically sits: the raw, clean and curated zones of the object store, "
        "how fresh each dataset is, and what the streaming job has ingested.",
    )
    with st.spinner("Listing the lake…"):
        inventory = lake_inventory()
    if inventory.empty:
        st.info("The lake is empty. Run the pipeline first.")
        return

    now = pd.Timestamp.now(tz="UTC")
    kpi = st.columns(3)
    for col, zone in zip(kpi, ZONE_COLORS):
        part = inventory[inventory["zone"] == zone]
        if part.empty:
            card(col, f"{zone} zone", "empty", "no objects yet")
            continue
        last = part["modified"].max()
        days = (now - last).days
        card(col, f"{zone} zone · {part['bucket'].iloc[0]}", f"{part['size'].sum() / 1e6:,.1f} MB",
             f"{len(part):,} objects in {part['dataset'].nunique()} datasets · last write "
             f"{last:%d %b %H:%M} UTC ({f'{days} d ago' if days else 'today'})")

    st.write("")
    datasets = (inventory.groupby(["zone", "dataset"])
                .agg(files=("key", "size"), size=("size", "sum"), modified=("modified", "max"))
                .reset_index())
    datasets["size_mb"] = datasets["size"] / 1e6
    datasets["age_days"] = (now - datasets["modified"]).dt.total_seconds() / 86400
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Largest datasets**")
        top = datasets.nlargest(15, "size").sort_values("size")
        fig = px.bar(top, x="size_mb", y="dataset", color="zone", orientation="h",
                     color_discrete_map=ZONE_COLORS, hover_data={"files": ":,", "zone": False},
                     labels={"size_mb": "Size (MB)", "dataset": "", "zone": "", "files": "Files"})
        fig.update_yaxes(categoryorder="array", categoryarray=top["dataset"].tolist())
        st.plotly_chart(style_fig(fig, 470), use_container_width=True)
    with right:
        st.markdown("**Size per zone**")
        zones = inventory.groupby("zone")["size"].sum().reindex(list(ZONE_COLORS)).fillna(0) / 1e6
        fig = px.bar(x=zones.index, y=zones.values, color=zones.index, color_discrete_map=ZONE_COLORS,
                     text=[f"{v:,.0f} MB" for v in zones.values], labels={"x": "", "y": "MB", "color": ""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.markdown("**Freshness of every dataset**")
    st.dataframe(
        datasets.sort_values("modified", ascending=False)[["dataset", "zone", "files", "size_mb", "modified", "age_days"]]
        .rename(columns={"dataset": "Dataset", "zone": "Zone", "files": "Files", "size_mb": "MB",
                         "modified": "Last write (UTC)", "age_days": "Days since write"}),
        hide_index=True, use_container_width=True, height=360,
        column_config={"MB": st.column_config.NumberColumn(format="%.2f"),
                       "Days since write": st.column_config.NumberColumn(format="%.1f"),
                       "Last write (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")},
    )

    st.divider()
    st.subheader("Streaming ingest")
    try:
        stream = load_lake("streaming/novel_metrics_incremental")
    except DataUnavailable:
        st.info("The streaming job has not written any output yet.")
        return
    stream = stream.assign(
        scraped=pd.to_datetime(stream["scraped_at_utc"], utc=True, errors="coerce"),
        ingested=pd.to_datetime(stream["ingested_at_utc"], utc=True, errors="coerce"),
    )
    stream["latency"] = (stream["ingested"] - stream["scraped"]).dt.total_seconds()
    batches = (stream.groupby("ingested")
               .agg(rows=("series_id", "size"), latency=("latency", "median"),
                    full=("metadata_tier", lambda s: (s == "full").mean()))
               .reset_index().sort_values("ingested"))
    batches["batch"] = [f"Batch {i + 1}" for i in range(len(batches))]
    commits = inventory["key"].str.contains(r"streaming/_checkpoint/commits/\d+$", regex=True).sum()

    kpi = st.columns(4)
    card(kpi[0], "Rows ingested", f"{len(stream):,}", "into curated-data/streaming")
    card(kpi[1], "Micro-batches", f"{len(batches):,}", f"{commits} committed in the checkpoint")
    card(kpi[2], "Median latency", f"{stream['latency'].median():,.0f} s", "from scrape to ingest")
    card(kpi[3], "Last batch", f"{batches['ingested'].max():%d %b %H:%M}", "UTC")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.bar(batches, x="batch", y="rows", text="rows",
                     hover_data={"ingested": True, "latency": ":.0f"},
                     labels={"batch": "", "rows": "Rows", "ingested": "Ingested (UTC)", "latency": "Median latency (s)"})
        fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with right:
        st.dataframe(
            batches[["batch", "ingested", "rows", "latency", "full"]].rename(columns={
                "batch": "Batch", "ingested": "Ingested (UTC)", "rows": "Rows",
                "latency": "Median latency (s)", "full": "Full-detail share"}),
            hide_index=True, use_container_width=True,
            column_config={"Median latency (s)": st.column_config.NumberColumn(format="%.1f"),
                           "Full-detail share": st.column_config.NumberColumn(format="%.2f"),
                           "Ingested (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")},
        )
    st.caption("Each micro-batch is one trigger of the Structured Streaming job in "
               "src/phases/streaming_ingest.py; latency is ingest time minus the scrape timestamp "
               "carried in each record.")


# PAGE — Dataset
DATASET_PREFIX = "novels_clean"
PAGE_SIZES = [10, 25, 50, 100, 250]
#: Model-input columns derived from the text columns; search skips them so it matches what a reader sees.
DERIVED_TEXT = {"corpus_raw", "corpus_clean", "synopsis_clean", "tokens", "synopsis_tokens"}


@st.cache_resource(show_spinner=False, max_entries=1)
def _dataset(version: str) -> pd.DataFrame:
    """The whole clean table, both metadata tiers. A resource, not data: 300 MB is never copied per rerun."""
    parts = []
    for tier in ("full", "listing"):
        try:
            part = read_parquet_dir(SETTINGS.minio.bucket_clean, f"{DATASET_PREFIX}/metadata_tier={tier}")
        except FileNotFoundError:
            continue
        except Exception as exc:
            raise DataUnavailable(str(exc)) from exc
        parts.append(part.assign(metadata_tier=tier))
    if not parts:
        raise DataUnavailable(f"No parquet under {SETTINGS.minio.bucket_clean}/{DATASET_PREFIX}")
    return pd.concat(parts, ignore_index=True)


@st.cache_resource(show_spinner=False, max_entries=1)
def _dataset_haystack(version: str) -> pd.Series:
    """Lowercase text of every reader-facing text column per row, built on the first search only."""
    frame = _dataset(version)
    text_cols = [c for c in frame.columns if frame[c].dtype == object and c not in DERIVED_TEXT
                 and isinstance(frame[c].dropna().iloc[0] if frame[c].notna().any() else "", str)]
    hay = frame[text_cols[0]].fillna("").astype(str).str.lower()
    for col in text_cols[1:]:
        hay = hay + " \x1f " + frame[col].fillna("").astype(str).str.lower()
    return hay


def _sortable(frame: pd.DataFrame) -> list[str]:
    """Columns with an order: list-valued cells (genre_list, tokens) have none."""
    def scalar(col: str) -> bool:
        values = frame[col].dropna()
        return values.empty or not isinstance(values.iloc[0], (list, tuple, np.ndarray))
    return [c for c in frame.columns if scalar(c)]


def column_picker(container, columns: list[str]) -> list[str]:
    """Include/exclude columns from a popover; every column starts included."""
    for col in columns:
        st.session_state.setdefault(f"dataset_col_{col}", True)

    def set_all(value: bool) -> None:
        for col in columns:
            st.session_state[f"dataset_col_{col}"] = value

    included = [c for c in columns if st.session_state[f"dataset_col_{c}"]]
    with container.popover(f"Columns · {len(included)}/{len(columns)}", use_container_width=True):
        a, b = st.columns(2)
        a.button("Include all", on_click=set_all, args=(True,), use_container_width=True)
        b.button("Exclude all", on_click=set_all, args=(False,), use_container_width=True)
        grid = st.columns(3)
        for i, col in enumerate(columns):
            grid[i % 3].checkbox(col, key=f"dataset_col_{col}")
    return [c for c in columns if st.session_state[f"dataset_col_{c}"]]


def _set_page(value: int) -> None:
    st.session_state["dataset_page"] = value


#: Header-sortable table: sorting runs in Python over every row, so a click on a
#: column name orders the whole dataset, not just the page on screen.
_datatable = components.declare_component(
    "nl_datatable", path=str(SETTINGS.paths.project_root / "app" / "components" / "datatable")
)
#: Longest text sent per cell; the cell shows an ellipsis and the full value on hover.
CELL_CHARS = 400
DATASET_LEAD = ["series_id", "title"]


#: Identifiers and calendar years read wrong with thousands separators.
PLAIN_NUMBER = ("series_id", "year", "decade")


def _cell(value, plain: bool = False) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = ", ".join(map(str, value))
    elif isinstance(value, float):
        value = (f"{value:.0f}" if plain else f"{value:,.4f}".rstrip("0").rstrip("."))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = str(value) if plain else f"{value:,}"
    text = str(value)
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 1] + "…"


def page_dataset() -> None:
    hero(
        "Dataset",
        "Every row and every column of the cleaned catalogue, the table all other pages are built "
        "from. Click a column name to sort every row by it.",
    )
    version = lake_version()
    with st.spinner("Loading the full dataset (about 2 seconds, once)…"):
        frame = _dataset(version)
    all_columns = DATASET_LEAD + [c for c in frame.columns if c not in DATASET_LEAD]
    sortable = set(_sortable(frame))

    c1, c2, c3 = st.columns([6, 1.3, 1.7], vertical_alignment="bottom")
    query = c1.text_input("Search", placeholder="Text anywhere in the row, e.g. cultivation")
    size = c2.selectbox("Rows per page", PAGE_SIZES, index=1)
    shown_cols = column_picker(c3, all_columns)
    if not shown_cols:
        st.info("Every column is excluded. Include at least one under Columns.")
        return

    # Default order: the leftmost shown column, ascending. A header click replaces it.
    clicked = st.session_state.get("dataset_sort")
    default_col = next((c for c in shown_cols if c in sortable), None)
    sort_col, sort_dir = (clicked["col"], clicked["dir"]) if clicked and clicked["col"] in shown_cols \
        else (default_col, "asc")

    view = frame
    needle = query.strip().lower()
    if needle:
        with st.spinner("Searching every row…"):
            view = frame[_dataset_haystack(version).str.contains(needle, regex=False).to_numpy()]
    if sort_col:
        view = view.sort_values(sort_col, ascending=sort_dir == "asc", na_position="last", kind="stable")

    total = len(view)
    pages = max(1, -(-total // size))
    # A new search, sort or page size starts again at page 1.
    signature = (needle, sort_col, sort_dir, size)
    if st.session_state.get("dataset_signature") != signature:
        st.session_state["dataset_signature"] = signature
        st.session_state["dataset_page"] = 1
    page = min(max(1, st.session_state.get("dataset_page", 1)), pages)
    start = (page - 1) * size
    rows = view.iloc[start:start + size]
    if rows.empty:
        st.info(f"No row contains “{query.strip()}”. Try a shorter word.")
        return

    clicked = _datatable(
        columns=shown_cols,
        numeric=[bool(pd.api.types.is_numeric_dtype(frame[c])) and frame[c].dtype != bool for c in shown_cols],
        sortable=[c in sortable for c in shown_cols],
        rows=[[_cell(v, c in PLAIN_NUMBER) for v, c in zip(row, shown_cols)]
              for row in rows[shown_cols].itertuples(index=False, name=None)],
        sort_col=sort_col, sort_dir=sort_dir, first_row=start + 1,
        # Capped so 250 rows scroll inside the table instead of stretching the page.
        height=min(42 + 33 * len(rows), 560),
        key="dataset_table", default=None,
    )
    if clicked and clicked != st.session_state.get("dataset_sort"):
        st.session_state["dataset_sort"] = clicked
        st.rerun()

    nav = st.columns([1, 1, 1, 4], vertical_alignment="center")
    nav[0].button("Previous", disabled=page <= 1, on_click=_set_page, args=(page - 1,), use_container_width=True)
    nav[1].button("Next", disabled=page >= pages, on_click=_set_page, args=(page + 1,), use_container_width=True)
    picked = int(nav[2].number_input("Page", 1, pages, page, label_visibility="collapsed"))
    if picked != page:
        _set_page(picked)
        st.rerun()
    nav[3].caption(
        f"Rows {start + 1:,} to {start + len(rows):,} of {total:,}"
        + (f" matching “{query.strip()}”" if needle else "")
        + f" · page {page:,} of {pages:,} · {len(shown_cols)} of {len(all_columns)} columns"
        + (f" · sorted by {sort_col} {'ascending' if sort_dir == 'asc' else 'descending'}" if sort_col else "")
    )


# Router
def technical_details(text: str) -> None:
    """Raw error text for maintainers; never on a public deployment, where it would name hosts and paths."""
    if PUBLIC:
        return
    with st.expander("Technical details"):
        st.code(text)


def guarded(render):
    """Show a plain message instead of a traceback when a page's tables are missing."""
    def run() -> None:
        try:
            render()
        except DataUnavailable as exc:
            st.warning("The data for this page is not available yet. Please try again later."
                       if PUBLIC else "The data for this page is not available yet. Run the pipeline "
                       "(`python scripts/run_pipeline.py`) and reload.")
            technical_details(str(exc))
    return run


#: url_path -> page, so pages can link to each other with st.page_link.
PAGE = {
    "overview": st.Page(guarded(page_overview), title="Overview", url_path="overview", default=True),
    "lookup": st.Page(guarded(page_lookup), title="Title Lookup", url_path="lookup"),
    "studio": st.Page(guarded(page_studio), title="AI Studio", url_path="studio"),
    "analytics": st.Page(guarded(page_analytics), title="Catalogue Analytics", url_path="analytics"),
    "trends": st.Page(guarded(page_trends), title="Market Trends", url_path="trends"),
    "genre": st.Page(guarded(page_genre), title="Genre Explorer", url_path="genre"),
    "tags": st.Page(guarded(page_tags), title="Story Elements", url_path="tags"),
    "creators": st.Page(guarded(page_creators), title="Authors & Publishers", url_path="creators"),
    "warehouse": st.Page(guarded(page_warehouse), title="Data Warehouse", url_path="warehouse"),
    "graph": st.Page(guarded(page_graph), title="Relationship Graph", url_path="graph"),
    "lake": st.Page(guarded(page_lake), title="Lake Operations", url_path="lake"),
    "dataset": st.Page(guarded(page_dataset), title="Dataset", url_path="dataset"),
    "governance": st.Page(guarded(page_governance), title="Quality & Governance", url_path="governance"),
    "methodology": st.Page(guarded(page_method), title="Methodology", url_path="methodology"),
}
#: Material Symbols per page, picked for what the page holds.
PAGE_ICON = {
    "overview": ":material/space_dashboard:", "lookup": ":material/search:", "studio": ":material/model_training:",
    "analytics": ":material/bar_chart:", "trends": ":material/trending_up:", "genre": ":material/category:",
    "tags": ":material/sell:", "creators": ":material/groups:", "dataset": ":material/table_rows:",
    "lake": ":material/database:", "warehouse": ":material/schema:", "graph": ":material/account_tree:",
    "governance": ":material/verified_user:", "methodology": ":material/menu_book:",
}
PAGES = {
    "Discover": [PAGE[k] for k in ("overview", "lookup", "studio")],
    "Market insight": [PAGE[k] for k in ("analytics", "trends", "genre", "tags", "creators")],
    "Data platform": [PAGE[k] for k in (("warehouse", "graph", "governance") if PUBLIC else
                                       ("dataset", "lake", "warehouse", "graph", "governance"))],
    "About": [PAGE["methodology"]],
}


def catalogue_size() -> int | None:
    try:
        return table_profile("analytics/viz_novels")[0]
    except DataUnavailable:
        return None


def lake_status_html() -> str:
    """
    Data-lake status from the same probe main() uses, plus when the curated
    zone last changed. Static dot plus text: the state is real, refreshed on
    each rerun, and never carried by colour alone.
    """
    online, _ = backend_online()
    if not online:
        return ('<div class="nl-status" role="status"><span class="dot" style="background:#F87171"></span>'
                "Data lake unreachable</div>")
    stamp = pd.to_datetime(lake_version(), errors="coerce", utc=True)
    when = f" · updated {stamp:%d %b %Y, %H:%M} UTC" if pd.notna(stamp) else ""
    return ('<div class="nl-status" role="status"><span class="dot" style="background:#34D399"></span>'
            f"Data lake connected{when}</div>")


def sidebar_nav() -> None:
    """Brand, one collapsible group per section, and the data-lake status."""
    keys = {id(page): key for key, page in PAGE.items()}
    size = catalogue_size()
    st.markdown(
        '<div class="nl-brand"><div class="name">NovelLens</div>'
        f'<div class="sub">{f"{size:,} titles" if size else "Catalogue"} · NovelUpdates</div></div>',
        unsafe_allow_html=True,
    )
    for section, pages in PAGES.items():
        with st.expander(section, expanded=True):
            for page in pages:
                st.page_link(page, icon=PAGE_ICON[keys[id(page)]])
    st.markdown(lake_status_html(), unsafe_allow_html=True)


def main() -> None:
    # The built-in list cannot fold a section, so the sidebar draws its own.
    page = st.navigation(PAGES, position="hidden")
    with st.sidebar:
        sidebar_nav()

    online, detail = backend_online()
    if not online:
        st.error("The data service is unreachable right now. Please try again shortly.")
        technical_details(detail)
        st.stop()

    page.run()


main()
