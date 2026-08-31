"""
Embedding kalimat pralatih
==========================
Blok fitur padat yang melengkapi TF-IDF pada Head A.

**Kenapa ini ada, padahal model card sebelumnya menolak jalur pralatih.**
Penolakan itu berdiri di atas satu premis: lingkungan dianggap tanpa akses
jaringan, sehingga embedding pralatih mustahil diperoleh dan model urutan
terpaksa belajar kosakata dari nol pada 3.568 dokumen. Premis itu ternyata
keliru — registry model dapat dijangkau dari dalam container. Begitu premisnya
diperbaiki, kesimpulannya ikut berubah.

**Kenapa digabung, bukan menggantikan TF-IDF.** Diukur terpisah, embedding
kalah dari TF-IDF (makro-F1 OOF 0,514 vs 0,547). Itu masuk akal: sebagian besar
sinyal genre pada korpus ini terletak pada keberadaan frasa khas ("cultivation",
"transmigration"), dan TF-IDF menangkap persis itu. Yang tidak ditangkapnya
adalah parafrasa — dua sinopsis yang bercerita sama dengan kata berbeda. Di
situlah embedding menyumbang, sehingga keduanya saling melengkapi alih-alih
bersaing.

**Kenapa hasil encoding di-cache.** Encoder-nya beku: teks yang sama selalu
menghasilkan vektor yang sama. Tanpa memo, `cross_val_predict` akan
meng-encode ulang korpus di tiap fold dan biaya pencarian hyperparameter naik
berlipat tanpa memberi informasi baru. Karena encoder tidak pernah di-`fit`
pada data kita, memo lintas-fold ini juga tidak membocorkan apa pun dari fold
validasi.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.common.logging_utils import get_logger

logger = get_logger("model.embeddings")

#: Backbone default. Dipilih karena tiga alasan: berukuran 33 juta parameter
#: sehingga inferensi satu teks tetap di bawah 100 ms tanpa GPU, peringkatnya
#: tinggi pada MTEB untuk kelas ukurannya, dan bobotnya cukup kecil untuk
#: ikut dibundel ke dalam image Docker.
#: Dibaca dari environment supaya nilainya sama dengan yang di-bake Dockerfile;
#: mengubahnya di satu tempat saja akan membuat inferensi mencari bobot yang
#: tidak ada di dalam image.
DEFAULT_MODEL = os.environ.get("SENTENCE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_MAX_LEN = 384

#: Memo proses: {(model, max_len, sha1(teks)) -> vektor}
_CACHE: dict[tuple, np.ndarray] = {}
_MODELS: dict[tuple, object] = {}

#: Batas atas memo. Saat training isinya persis satu korpus (4.460 entri,
#: ~6,8 MB) sehingga batas ini tidak pernah tersentuh. Yang dijaga adalah
#: proses Streamlit yang hidup berminggu-minggu: tanpa batas, setiap teks baru
#: dari pengguna menambah ~1,6 KB yang tidak pernah dilepas. Saat penuh,
#: separuh entri terlama dibuang — memo ini hanya optimasi kecepatan, jadi
#: kehilangan isinya tidak pernah mengubah hasil.
_CACHE_MAX = 20_000


def _evict_if_full() -> None:
    if len(_CACHE) <= _CACHE_MAX:
        return
    for key in list(_CACHE)[: len(_CACHE) // 2]:
        del _CACHE[key]
    logger.info("Memo embedding dipangkas ke %d entri.", len(_CACHE))


def _key(model: str, max_len: int, text: str) -> tuple:
    return (model, max_len, hashlib.sha1(text.encode("utf-8")).hexdigest())


def _get_model(name: str, max_len: int):
    """Muat encoder sekali per proses (lazy — artefak joblib tidak memuat bobot)."""
    cached = _MODELS.get((name, max_len))
    if cached is not None:
        return cached

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import SentenceTransformer

    # Image ini menyetel OMP_NUM_THREADS=2 untuk mencegah deadlock antara
    # thread-pool OpenBLAS dan JVM Spark (lihat Dockerfile). PyTorch ikut
    # membaca variabel itu, sehingga encoder berjalan dua thread saja dan
    # meng-encode 4.460 dokumen makan belasan menit.
    #
    # Batas itu dinaikkan HANYA untuk encoder, dan hanya lewat API PyTorch —
    # OpenBLAS tidak tersentuh, jadi alasan asli pembatasannya tetap dihormati.
    # Aman dilakukan di sini karena Fase 4 baru berjalan setelah SparkSession
    # ditutup: tidak ada JVM yang bersaing memperebutkan core.
    threads = int(os.environ.get("ENCODER_THREADS", "4"))
    if threads > 0:
        torch.set_num_threads(threads)

    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = max_len
    _MODELS[(name, max_len)] = model
    logger.info("Encoder %s dimuat (dim=%d, max_len=%d)",
                name, model.get_sentence_embedding_dimension(), max_len)
    return model


def encode(texts: Iterable[str], model: str = DEFAULT_MODEL,
           max_len: int = DEFAULT_MAX_LEN, batch_size: int = 32) -> np.ndarray:
    """Encode daftar teks, memakai memo untuk yang sudah pernah dihitung."""
    texts = [str(t) if t is not None else "" for t in texts]
    keys = [_key(model, max_len, t) for t in texts]

    todo = [i for i, k in enumerate(keys) if k not in _CACHE]
    if todo:
        vectors = _get_model(model, max_len).encode(
            [texts[i] for i in todo], batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        for i, vec in zip(todo, vectors):
            _CACHE[keys[i]] = vec.astype(np.float32)
        # Hasil disusun SEBELUM pemangkasan, supaya entri yang baru saja
        # dihitung tidak ikut terbuang sebelum sempat dipakai.
        out = np.vstack([_CACHE[k] for k in keys])
        _evict_if_full()
        return out

    return np.vstack([_CACHE[k] for k in keys])


def prime_cache(texts: Iterable[str], vectors: np.ndarray,
                model: str = DEFAULT_MODEL, max_len: int = DEFAULT_MAX_LEN) -> None:
    """Isi memo dari embedding yang sudah dihitung sebelumnya (mis. dari berkas)."""
    for text, vec in zip(texts, vectors):
        _CACHE[_key(model, max_len, str(text))] = np.asarray(vec, dtype=np.float32)


class SentenceEncoder(BaseEstimator, TransformerMixin):
    """
    Blok fitur padat untuk dipakai di dalam `ColumnTransformer`.

    Hanya menyimpan NAMA model, bukan bobotnya, supaya artefak joblib tetap
    kecil dan encoder dimuat ulang secara lazy setelah unpickle.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_len: int = DEFAULT_MAX_LEN,
                 batch_size: int = 32):
        self.model = model
        self.max_len = max_len
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return encode(pd.Series(X).fillna("").astype(str).tolist(),
                      self.model, self.max_len, self.batch_size)

    def __sklearn_is_fitted__(self) -> bool:
        return True


def build_natural_text(frame: pd.DataFrame) -> pd.Series:
    """
    Susun teks masukan encoder.

    Berbeda dari jalur TF-IDF, teks di sini sengaja dibiarkan utuh: stopword
    tidak dibuang dan tidak ada stemming. Model pralatih dilatih pada kalimat
    lengkap, jadi merusak sintaksnya justru menjauhkan masukan dari distribusi
    yang pernah dilihatnya.
    """
    def col(*names: str) -> pd.Series:
        """Ambil kolom pertama yang tersedia; nama sinopsis berbeda antar sumber."""
        for name in names:
            if name in frame.columns:
                return frame[name].fillna("")
        return pd.Series([""] * len(frame), index=frame.index)

    title = col("title")
    # `synopsis_raw` dipakai frame riset, `synopsis` dipakai data bersih Fase 2.
    synopsis = col("synopsis_raw", "synopsis")
    tags = col("tags")
    return (title.astype(str) + ". " + synopsis.astype(str).str.strip()
            + " Tags: " + tags.astype(str).str.replace(",", ", ", regex=False)).str.strip()
