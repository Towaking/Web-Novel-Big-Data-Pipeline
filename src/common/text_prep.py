"""
Text preprocessing (jalur inferensi)
====================================
Cermin Python murni dari pipeline teks Spark ML di Fase 2
(`RegexTokenizer` + `StopWordsRemover`).

Kenapa ada dua implementasi:
  - Fase 2 memproses 4.461 dokumen sekaligus -> harus terdistribusi (Spark ML).
  - Fase 5 memproses SATU input pengguna per klik -> menyalakan Spark hanya
    untuk itu memakan detik dan sumber daya besar.

Risiko dari dua implementasi adalah *drift*: jika keduanya membersihkan teks
dengan cara berbeda, distribusi fitur saat inferensi tidak sama dengan saat
training dan akurasi turun diam-diam. Karena itu daftar stopword yang persis
dipakai Fase 2 ikut disimpan sebagai artefak model dan dibaca ulang di sini —
bukan diketik ulang.
"""

from __future__ import annotations

import re

#: Sama dengan RegexTokenizer(minTokenLength=3) di Fase 2.
MIN_TOKEN_LENGTH = 3

_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize(raw: str) -> str:
    """Lowercase, buang karakter non-huruf, rapatkan spasi (identik Fase 2)."""
    if not raw:
        return ""
    lowered = raw.lower()
    stripped = _NON_ALPHA.sub(" ", lowered)
    return _MULTI_SPACE.sub(" ", stripped).strip()


def tokenize(raw: str, stopwords: set[str]) -> list[str]:
    """Normalisasi -> pisah spasi -> buang token pendek & stopword."""
    return [
        token
        for token in normalize(raw).split(" ")
        if len(token) >= MIN_TOKEN_LENGTH and token and token not in stopwords
    ]


def to_corpus(raw: str, stopwords: set[str]) -> str:
    """Hasil akhir yang formatnya sama dengan kolom `corpus_clean` di Fase 2."""
    return " ".join(tokenize(raw, stopwords))


def build_query_text(title: str = "", tags: str = "") -> str:
    """
    Gabungkan judul dan tag persis seperti pembentukan korpus di Fase 2:
    `corpus_raw = title + " " + tags`.
    """
    return f"{title or ''} {tags or ''}".strip()
