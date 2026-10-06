"""
Generator ide judul berbasis pola katalog
=========================================
Judul web novel sangat formulaik ("Reborn as ...", "... System", "The
Villainess ..."). Modul ini mempelajari pola itu langsung dari judul-judul
katalog dengan model Markov kata orde-2, lalu mengondisikannya pada cerita
pengguna:

1. Bobot judul: tiap judul katalog diberi bobot dari kemiripan sinopsisnya
   dengan deskripsi pengguna (ruang embedding Fase 4), kecocokan genre, dan
   negara asal (gaya judul CN/KR/JP berbeda jauh).
2. Transisi kata dihitung ulang dengan bobot tersebut, jadi frasa dari novel
   yang mirip lebih sering terpakai.
3. Kata yang muncul di deskripsi dan pernah dipakai di judul katalog diberi
   dorongan tambahan saat sampling.
4. Kandidat disaring (bukan salinan judul yang sudah ada, tidak menyalin
   frasa panjang yang unik milik satu judul) lalu diberi skor relevansi +
   kelancaran dan dipilih yang beragam.

Tidak ada model bahasa: setiap pasangan tiga kata di judul hasil pernah muncul
di katalog. Itu sumber "pola dataset"-nya, sekaligus batasannya.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

BOS, EOS = "<s>", "</s>"

# Kata yang tetap huruf kecil di tengah judul (gaya judul NovelUpdates).
SMALL_WORDS = {
    "a", "an", "the", "of", "in", "to", "as", "with", "and", "but", "or", "for",
    "on", "at", "by", "from", "into", "is", "vs",
}
# Kata fungsi: tidak dihitung sebagai kata isi saat menilai relevansi.
FUNCTION_WORDS = SMALL_WORDS | {
    "i", "me", "my", "you", "your", "he", "his", "him", "she", "her", "it", "its",
    "we", "our", "they", "their", "this", "that", "who", "was", "be", "am", "are",
    "been", "being", "not", "no", "so", "all", "after", "when", "can", "will",
    "just", "up", "out", "s", "t", "m", "don't", "i'm", "what", "how", "if",
}
DANGLING = SMALL_WORDS | {"i", "my", "your", "his", "her", "their", "our", "its", "who", "that", "when", "after"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[,:!?]")
PUNCT = {",", ":", "!", "?"}
NGRAM_COPY = 5  # frasa sepanjang ini yang hanya dimiliki satu judul = menyalin


def tokenize_title(title: str) -> list[str]:
    text = str(title).replace("’", "'").replace("‘", "'").replace("–", " - ").replace("—", " - ")
    # Kata bersensor (M*rder) dibuang utuh; memotongnya menghasilkan "M Rder".
    text = re.sub(r"\S*\*\S*", " ", text)
    return TOKEN_RE.findall(text)


def _content(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in FUNCTION_WORDS and t not in PUNCT and len(t) > 1}


@dataclass
class TitleIdea:
    title: str
    relevance: float
    sources: list[str]


class TitleGenerator:
    """Model Markov judul katalog. Dibangun sekali, dikondisikan per permintaan."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        titles = self.catalog["title"].astype(str).tolist()

        surface: dict[str, Counter] = {}
        rows_state, rows_next, rows_title = [], [], []
        ngram_owner: dict[tuple, set] = {}
        self.title_tokens: list[list[str]] = []
        for idx, title in enumerate(titles):
            raw = tokenize_title(title)
            low = [t.lower() for t in raw]
            self.title_tokens.append(low)
            if len(low) < 2:
                continue
            for r, lo in zip(raw, low):
                surface.setdefault(lo, Counter())[r] += 1
            seq = [BOS, BOS] + low + [EOS]
            for i in range(2, len(seq)):
                rows_state.append((seq[i - 2], seq[i - 1]))
                rows_next.append(seq[i])
                rows_title.append(idx)
            for i in range(len(low) - NGRAM_COPY + 1):
                ngram_owner.setdefault(tuple(low[i:i + NGRAM_COPY]), set()).add(idx)

        # Bentuk tulisan tersering per kata; kata berhuruf besar khusus (CEO,
        # Mr) dipertahankan apa adanya.
        self.surface = {w: c.most_common(1)[0][0] for w, c in surface.items()}
        # Dibandingkan dalam bentuk token, jadi beda tanda kutip/sensor (Sui**de)
        # tidak meloloskan salinan.
        self.title_set = {" ".join(toks) for toks in self.title_tokens}
        # Awal judul yang ada ("The Male Lead" dari "The Male Lead's Fiancee")
        # terbaca sebagai judul terpotong, bukan judul baru.
        self.prefixes = {
            " ".join(toks[:k]) for toks in self.title_tokens for k in range(2, len(toks))
        }
        self.unique_ngrams = {g for g, owners in ngram_owner.items() if len(owners) == 1}

        frame = pd.DataFrame({
            "s1": [s[0] for s in rows_state], "s2": [s[1] for s in rows_state],
            "nxt": rows_next, "title_idx": rows_title,
        })
        self.vocab = sorted(set(frame["nxt"]) | {BOS})
        index = {w: i for i, w in enumerate(self.vocab)}
        self.word_id = index
        frame["s1"] = frame["s1"].map(index)
        frame["s2"] = frame["s2"].map(index)
        frame["nxt"] = frame["nxt"].map(index)
        self.transitions = frame.sort_values(["s1", "s2"]).reset_index(drop=True)
        self._title_idx = self.transitions["title_idx"].to_numpy()
        self._nxt = self.transitions["nxt"].to_numpy()
        # Rentang baris per state (s1, s2); transitions sudah terurut per state.
        self._starts: dict[tuple, tuple[int, int]] = {}
        for i, key in enumerate(zip(self.transitions["s1"].to_numpy(), self.transitions["s2"].to_numpy())):
            lo, _ = self._starts.get(key, (i, i))
            self._starts[key] = (lo, i + 1)

        # Model orde-1 (cadangan) dan model global tanpa bobot (skor kelancaran).
        v = len(self.vocab)
        self._v = v
        bigram = Counter(zip(frame["s2"], frame["nxt"]))
        self._bigram_global = bigram
        self._from_count = Counter(frame["s2"])
        self._trigram_global = Counter(zip(frame["s1"], frame["s2"], frame["nxt"]))
        self._state_count = Counter(zip(frame["s1"], frame["s2"]))
        # Kata yang pernah dipakai di judul: hanya kata ini yang bisa didorong.
        self.title_vocab = set(self.vocab) - {BOS, EOS}

    # Pembobotan
    def title_weights(
        self,
        similarity: np.ndarray | None,
        genres: list[str] | None = None,
        origin: str | None = None,
        focus: int = 60,
    ) -> np.ndarray:
        """
        Bobot tiap judul katalog. Semua judul tetap punya bobot dasar supaya
        tata bahasa umum (kata sambung, pola "X of the Y") selalu tersedia;
        `focus` judul termirip mendapat bobot jauh lebih besar.
        """
        n = len(self.catalog)
        weight = np.full(n, 0.05)
        if similarity is not None:
            # NaN = judul yang belum masuk ruang embedding model; hanya bobot dasar.
            similarity = np.nan_to_num(np.asarray(similarity, dtype=float), nan=-1.0)
            top = np.argsort(-similarity)[:focus]
            weight[top] += np.exp(6.0 * np.clip(similarity[top], 0, None))
        if genres:
            wanted = set(genres)
            match = self.catalog["genre_list"].map(
                lambda g: len(wanted.intersection(g if isinstance(g, (list, np.ndarray)) else [])) / len(wanted)
            ).to_numpy(dtype=float)
            weight *= 0.3 + 1.7 * match
        if origin:
            # Kuat: gaya judul JP (kalimat panjang) dan CN (frasa pendek) tidak
            # boleh tercampur hanya karena satu novel CN sangat mirip ceritanya.
            weight *= np.where(self.catalog["type_origin"].to_numpy() == origin, 1.0, 0.05)
        return weight

    # Generasi
    def _sample(self, edge_weight, starts, boost, rng, max_len=16):
        tokens, sources = [], []
        s1 = s2 = self.word_id[BOS]
        for _ in range(max_len):
            lo, hi = starts.get((s1, s2), (0, 0))
            if hi <= lo:
                return None, None
            nxt = self._nxt[lo:hi]
            w = edge_weight[lo:hi] * boost[nxt]
            if w.sum() <= 0:
                return None, None
            pick = lo + rng.choice(hi - lo, p=w / w.sum())
            word = int(self._nxt[pick])
            if word == self.word_id[EOS]:
                return tokens, sources
            tokens.append(self.vocab[word])
            sources.append(int(self._title_idx[pick]))
            s1, s2 = s2, word
        return None, None

    def _fluency(self, tokens: list[str]) -> float:
        """Rata-rata log-peluang orde-2 (dengan cadangan orde-1) di model global."""
        ids = [self.word_id[BOS]] * 2 + [self.word_id[t] for t in tokens] + [self.word_id[EOS]]
        total = 0.0
        for i in range(2, len(ids)):
            tri = self._trigram_global[(ids[i - 2], ids[i - 1], ids[i])]
            st = self._state_count[(ids[i - 2], ids[i - 1])]
            bi = self._bigram_global[(ids[i - 1], ids[i])]
            fr = self._from_count[ids[i - 1]]
            p = 0.7 * (tri / st if st else 0) + 0.3 * ((bi + 0.1) / (fr + 0.1 * self._v))
            total += np.log(p)
        return total / (len(ids) - 2)

    def _is_copy(self, tokens: list[str], sources: list[int]) -> bool:
        # Semua kata dari satu judul = potongan judul itu, bukan judul baru.
        joined = " ".join(tokens)
        if len(set(sources)) < 2 or joined in self.title_set or joined in self.prefixes:
            return True
        for i in range(len(tokens) - NGRAM_COPY + 1):
            if tuple(tokens[i:i + NGRAM_COPY]) in self.unique_ngrams:
                return True
        return False

    def render(self, tokens: list[str]) -> str:
        words = []
        for i, t in enumerate(tokens):
            form = self.surface.get(t, t)
            if t in PUNCT:
                words.append(form)
                continue
            if form == t or form == t.capitalize():
                # Huruf kecil untuk kata sambung di tengah; kapital awal untuk sisanya.
                after_colon = i > 0 and tokens[i - 1] in {":", "!", "?"}
                form = t if (t in SMALL_WORDS and i > 0 and not after_colon) else t[:1].upper() + t[1:]
            words.append(form)
        text = " ".join(words)
        return re.sub(r" ([,:!?])", r"\1", text)

    def generate(
        self,
        weights: np.ndarray,
        keywords: set[str],
        n: int = 8,
        attempts: int = 600,
        min_words: int = 2,
        max_words: int = 12,
        seed: int = 0,
        exclude: set[str] | None = None,
    ) -> list[TitleIdea]:
        """
        Sampling `attempts` kandidat, buang salinan, pilih `n` yang paling
        relevan dan lancar dengan isi yang tidak saling tumpang tindih.
        `exclude` berisi judul (huruf kecil) yang sudah pernah ditampilkan.
        """
        exclude = exclude or set()
        rng = np.random.default_rng(seed)
        edge_weight = weights[self._title_idx]
        starts = self._starts
        boost = np.ones(self._v)
        for word in keywords & self.title_vocab:
            boost[self.word_id[word]] = 4.0

        seen, pool = set(), []
        for _ in range(attempts):
            tokens, sources = self._sample(edge_weight, starts, boost, rng)
            if not tokens or not (min_words <= len([t for t in tokens if t not in PUNCT]) <= max_words):
                continue
            last = tokens[-1]
            # Judul yang berakhir di kata sambung/kepemilikan ("... with", "... My")
            # atau memuat huruf tunggal sisa singkatan ("Top O") terbaca terpotong.
            if last in PUNCT - {"!", "?"} or last in DANGLING:
                continue
            if any(len(t) == 1 and t.isalpha() and t not in {"a", "i"} for t in tokens):
                continue
            key = " ".join(tokens)
            if key in seen or self.render(tokens).lower() in exclude or self._is_copy(tokens, sources):
                continue
            seen.add(key)
            content = _content(tokens)
            # Relevansi: kata isi judul yang muncul di cerita pengguna atau di
            # judul novel termirip, ditimbang bobot judul asalnya.
            hits = len(content & keywords)
            support = float(np.mean([weights[s] for s in sources])) if sources else 0.0
            pool.append({
                "tokens": tokens, "content": content, "sources": sources,
                "hits": hits, "support": support, "fluency": self._fluency(tokens),
            })
        if not pool:
            return []

        frame = pd.DataFrame(pool)
        z = lambda s: (s - s.mean()) / (s.std() or 1.0)  # noqa: E731
        frame["score"] = (
            1.0 * frame["hits"].clip(upper=3)
            + 0.8 * z(np.log(frame["support"]))
            + 0.6 * z(frame["fluency"])
            # Sedikit acak supaya batch berikutnya tidak dibuka judul yang sama.
            + rng.normal(0, 0.35, len(frame))
        )
        frame = frame.sort_values("score", ascending=False)

        chosen: list[dict] = []
        for row in frame.to_dict("records"):
            if any(
                len(row["content"] & c["content"]) / max(1, len(row["content"] | c["content"])) > 0.4
                for c in chosen
            ):
                continue
            chosen.append(row)
            if len(chosen) == n:
                break

        top = frame["score"].max()
        ideas = []
        for row in chosen:
            # Judul asal yang paling banyak menyumbang kata, maksimal tiga.
            counts = Counter(row["sources"])
            ranked = sorted(counts, key=lambda i: (-counts[i], -weights[i]))
            ideas.append(TitleIdea(
                title=self.render(row["tokens"]),
                relevance=float(row["score"] / top) if top > 0 else 0.0,
                sources=[str(self.catalog.at[i, "title"]) for i in ranked[:3]],
            ))
        return ideas


def description_keywords(text: str) -> set[str]:
    """Kata isi dari deskripsi pengguna, huruf kecil, untuk didorong saat sampling."""
    return _content([t.lower() for t in tokenize_title(text)])
