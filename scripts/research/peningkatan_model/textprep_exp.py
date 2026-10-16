"""Pembangun ulang kolom teks dari field mentah (cermin Fase 2) + varian."""
import re
import pandas as pd
from src.common.text_prep import uncensor

_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def normalize(raw: str, fix_censor: bool) -> str:
    if not raw:
        return ""
    low = raw.lower()
    if fix_censor:
        low = uncensor(low)
    return _SPACES.sub(" ", _NON_ALPHA.sub(" ", low)).strip()


def to_corpus(raw: str, stop: set, fix_censor: bool) -> str:
    return " ".join(t for t in normalize(raw, fix_censor).split(" ") if len(t) >= 3 and t not in stop)


def rebuild(pdf: pd.DataFrame, stop, fix_censor: bool) -> pd.DataFrame:
    stop = set(stop)
    title = pdf["title"].fillna("")
    syn = pdf["synopsis"].fillna("")
    tags = pdf["tags"].fillna("")
    corpus = [to_corpus(f"{a} {b} {c}", stop, fix_censor) for a, b, c in zip(title, syn, tags)]
    synopsis = [to_corpus(s, stop, fix_censor) for s in syn]
    low_tags = tags.str.lower()
    if fix_censor:
        low_tags = low_tags.map(uncensor)
    tags_clean = low_tags.str.replace(",", " ", regex=False)
    return pd.DataFrame({"corpus_clean": corpus, "synopsis_clean": synopsis, "tags_clean": tags_clean})
