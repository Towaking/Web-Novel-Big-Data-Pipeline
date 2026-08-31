"""
Augmentasi data untuk multi-label tak seimbang
==============================================
Dua teknik dari literatur, keduanya menyasar masalah yang sama: genre langka
punya terlalu sedikit contoh positif sehingga model belajar menebak "bukan".

1. **EDA — random deletion** (Wei & Zou, EMNLP 2019).
   Membuat salinan dokumen dengan sebagian token dihapus acak. Catatan penting
   untuk kasus ini: dari empat operasi EDA, hanya *deletion* dan *insertion*
   yang berpengaruh pada model bag-of-words — *random swap* sama sekali tidak
   mengubah TF-IDF karena urutan kata tidak dipakai, dan *synonym replacement*
   butuh WordNet yang tidak tersedia di lingkungan offline ini. Jadi yang
   diterapkan adalah deletion, yang secara domain juga masuk akal: ia meniru
   novel dengan deskripsi lebih miskin.

2. **MLSMOTE** (Charte et al., Knowledge-Based Systems 2015).
   Membuat instance sintetis di ruang fitur dengan interpolasi ke tetangga
   terdekat, khusus untuk instance yang membawa label minoritas. Labelset
   sintetisnya ditentukan lewat *ranking* — label yang muncul pada mayoritas
   tetangga ikut dibawa.

Keduanya HANYA boleh diterapkan pada data latih. Menerapkannya sebelum
pembagian train/test akan membocorkan informasi dan membuat skor uji palsu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from Model.pipeline.config import AugmentationConfig, RANDOM_STATE


def imbalance_ratios(y: np.ndarray) -> np.ndarray:
    """
    IRLbl — *imbalance ratio per label* (Charte et al., 2015).

    Rasio antara frekuensi label terbanyak dan frekuensi label ini. Nilai 1
    berarti label termayoritas; makin besar makin langka.
    """
    counts = y.sum(axis=0).astype(float)
    counts[counts == 0] = np.nan
    return np.nanmax(counts) / counts


def minority_labels(y: np.ndarray, quantile: float) -> np.ndarray:
    """Indeks label yang dianggap minoritas menurut ambang kuantil IRLbl."""
    irlbl = imbalance_ratios(y)
    threshold = np.nanquantile(irlbl, quantile)
    return np.flatnonzero(irlbl > threshold)


def random_deletion(
    frame: pd.DataFrame,
    y: np.ndarray,
    config: AugmentationConfig,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Gandakan dokumen berlabel minoritas dengan sebagian token dihapus acak.

    Salinan mewarisi labelset aslinya: menghapus sebagian kata tidak mengubah
    genre sebuah cerita, sehingga label tetap sah. Ini yang membuat teknik ini
    aman dipakai, berbeda dengan augmentasi yang mengubah makna.
    """
    rng = np.random.default_rng(random_state)
    minority = minority_labels(y, config.minority_quantile)
    if len(minority) == 0:
        return frame, y

    # Dokumen yang membawa minimal satu label minoritas.
    targets = np.flatnonzero(y[:, minority].sum(axis=1) > 0)
    if len(targets) == 0:
        return frame, y

    text_columns = ["corpus_clean", "synopsis_clean", "tags_clean"]
    extra_frames, extra_labels = [], []

    for _ in range(config.deletion_copies):
        copy = frame.iloc[targets].copy()
        for column in text_columns:
            copy[column] = [
                _drop_tokens(doc, config.deletion_rate, rng) for doc in copy[column].astype(str)
            ]
        extra_frames.append(copy)
        extra_labels.append(y[targets])

    augmented = pd.concat([frame, *extra_frames], ignore_index=True)
    labels = np.vstack([y, *extra_labels])
    return augmented, labels


def _drop_tokens(doc: str, rate: float, rng: np.random.Generator) -> str:
    """Hapus tiap token dengan peluang `rate`, sisakan minimal satu token."""
    tokens = doc.split()
    if len(tokens) <= 3:
        return doc
    keep = [t for t in tokens if rng.random() > rate]
    return " ".join(keep) if keep else tokens[0]


def mlsmote(
    features: sparse.spmatrix,
    y: np.ndarray,
    config: AugmentationConfig,
    random_state: int = RANDOM_STATE,
) -> tuple[sparse.spmatrix, np.ndarray]:
    """
    MLSMOTE — sintesis instance untuk label minoritas di ruang fitur.

    Untuk setiap instance referensi yang membawa label minoritas:
      1. cari k tetangga terdekatnya (cosine, karena fitur TF-IDF),
      2. buat titik baru dengan interpolasi acak ke salah satu tetangga,
      3. tentukan labelset baru lewat *ranking*: label yang hadir pada lebih
         dari separuh tetangga ikut diaktifkan.

    Bekerja di ruang fitur, bukan ruang teks — karena itu ia dipanggil SETELAH
    vectorizer di-fit pada fold latih. Menjalankannya sebelum vectorizer akan
    menghasilkan dokumen sintetis yang tidak punya wujud tekstual.
    """
    rng = np.random.default_rng(random_state)
    minority = minority_labels(y, config.minority_quantile)
    if len(minority) == 0:
        return features, y

    targets = np.flatnonzero(y[:, minority].sum(axis=1) > 0)
    if len(targets) < config.mlsmote_neighbors + 1:
        return features, y

    neighbours = NearestNeighbors(
        n_neighbors=min(config.mlsmote_neighbors + 1, len(targets)), metric="cosine"
    ).fit(features[targets])
    _, indices = neighbours.kneighbors(features[targets])

    new_rows, new_labels = [], []
    for position, reference in enumerate(targets):
        # Kolom 0 adalah instance itu sendiri, jadi dilewati.
        pool = indices[position][1:]
        if len(pool) == 0:
            continue
        chosen = targets[pool[rng.integers(len(pool))]]
        ratio = rng.random()

        # Interpolasi linear pada baris sparse.
        row = features[reference] + ratio * (features[chosen] - features[reference])
        new_rows.append(sparse.csr_matrix(row))

        # Strategi "ranking": label yang dimiliki mayoritas tetangga ikut dibawa.
        neighbour_labels = y[targets[pool]]
        new_labels.append((neighbour_labels.mean(axis=0) >= 0.5).astype(y.dtype))

    if not new_rows:
        return features, y

    return sparse.vstack([features, *new_rows]).tocsr(), np.vstack([y, np.array(new_labels)])
