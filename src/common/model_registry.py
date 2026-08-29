"""
Model registry & jalur inferensi
================================
Memuat artefak Fase 4 satu kali, lalu menyediakan API inferensi bersih untuk
*AI Playground* di Fase 5.

Layer ini sengaja dipisah dari kode Streamlit supaya:
  * logika inferensi bisa diuji tanpa menyalakan UI,
  * urutan pra-pemrosesan teks terjamin sama dengan saat training, dan
  * Streamlit cukup memanggil fungsi, tidak perlu tahu isi artefak.

Tidak ada Spark di sini — inferensi harus selesai dalam milidetik.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.common.text_prep import build_query_text, to_corpus, tokenize

logger = get_logger("model_registry")

ARTIFACT_DIR = Path("/opt/project/Model/artifacts")

REQUIRED_FILES = (
    "vectorizer.joblib",
    "svd.joblib",
    "genre_models.joblib",
    "genre_binarizer.joblib",
    "kmeans.joblib",
    "catalog.parquet",
    "embeddings.npy",
    "metrics.json",
)


class ModelNotTrainedError(RuntimeError):
    """Dilempar bila artefak Fase 4 belum ada."""


@dataclass
class ModelRegistry:
    """Wadah seluruh artefak model beserta API inferensinya."""

    vectorizer: object
    svd: object
    #: {nama_model: {"pipeline": Pipeline, "thresholds": np.ndarray}}
    genre_models: dict
    genre_binarizer: object
    kmeans: object
    rating_regressor: object | None
    catalog: pd.DataFrame
    embeddings: np.ndarray
    metrics: dict
    stopwords: set[str] = field(default_factory=set)

    # -- Katalog model genre ----------------------------------------------
    @property
    def genre_model_names(self) -> list[str]:
        """Nama model genre yang tersedia, terbaik (macro-F1) di urutan pertama."""
        best = self.metrics.get("head_a_genre_tagger", {}).get("best_model")
        names = list(self.genre_models)
        if best in names:
            names.remove(best)
            names.insert(0, best)
        return names

    def genre_model_info(self, name: str) -> dict:
        """Metadata + metrik satu model genre (untuk ditampilkan di UI)."""
        return self.metrics.get("head_a_genre_tagger", {}).get("models", {}).get(name, {})

    # -- Representasi ------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        """
        Ubah teks bebas menjadi vektor pada ruang yang sama dengan katalog.

        Urutannya wajib identik dengan Fase 2 + Fase 4:
        normalisasi -> stopword removal -> TF-IDF -> SVD -> normalisasi L2.
        """
        corpus = to_corpus(text, self.stopwords)
        if not corpus:
            raise ValueError("Teks tidak menyisakan token bermakna setelah pembersihan.")
        tfidf = self.vectorizer.transform([corpus])
        vector = self.svd.transform(tfidf)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).astype(np.float32).ravel()

    # -- HEAD A ------------------------------------------------------------
    def predict_genres(
        self,
        text: str,
        tags: str = "",
        model: str | None = None,
        top_k: int = 8,
        type_origin: str = "CN",
        novel_type: str = "Web Novel",
        sensitivity: float = 1.0,
    ) -> pd.DataFrame:
        """
        Prediksi himpunan genre (multi-label) dengan model pilihan pengguna.

        Args:
            text: sinopsis atau deskripsi bebas.
            tags: daftar tag dipisah koma (opsional, memperkuat sinyal).
            model: nama model genre; default memakai yang terbaik.
            sensitivity: pengali ambang. <1 membuat model lebih longgar
                (lebih banyak genre muncul), >1 lebih ketat. Ambang dasarnya
                sudah dioptimasi per genre saat training, jadi ini hanya
                pengatur halus untuk pengguna.

        Raises:
            ValueError: teks tidak menyisakan token bermakna, atau model tidak ada.
        """
        name = model or self.genre_model_names[0]
        if name not in self.genre_models:
            raise ValueError(f"Model genre {name!r} tidak tersedia.")

        combined = f"{text} {tags}".strip()
        corpus = to_corpus(combined, self.stopwords)
        if not corpus:
            raise ValueError("Teks tidak menyisakan token bermakna setelah pembersihan.")

        # Bentuk masukan harus sama persis dengan saat training
        # (lihat genre_models.FEATURE_COLUMNS), termasuk urutan kolomnya.
        frame = pd.DataFrame(
            [
                {
                    "corpus_clean": corpus,
                    "tags_clean": (tags or text).lower().replace(",", " "),
                    "type_origin": type_origin,
                    "type": novel_type,
                }
            ]
        )

        bundle = self.genre_models[name]
        probabilities = np.asarray(bundle["pipeline"].predict_proba(frame))
        if probabilities.ndim == 3:  # sebagian estimator: (n_labels, n_samples, 2)
            probabilities = probabilities[:, :, 1].T
        probabilities = probabilities.ravel()

        thresholds = np.asarray(bundle["thresholds"], dtype=float) * float(sensitivity)
        result = pd.DataFrame(
            {
                "genre": self.genre_binarizer.classes_,
                "probability": probabilities,
                "threshold": thresholds,
            }
        )
        result["diprediksi"] = result["probability"] >= result["threshold"]
        # Diurutkan berdasarkan seberapa jauh melewati ambangnya sendiri, bukan
        # probabilitas mentah: genre langka dengan ambang rendah tetap terlihat.
        result["margin"] = result["probability"] - result["threshold"]
        return (
            result.sort_values(["diprediksi", "margin"], ascending=[False, False])
            .head(top_k)
            .reset_index(drop=True)
        )

    # -- HEAD B ------------------------------------------------------------
    def recommend_by_vector(self, vector: np.ndarray, k: int = 10, exclude_idx: int | None = None) -> pd.DataFrame:
        """Cari k novel termirip. Embedding sudah ternormalisasi -> dot = cosine."""
        scores = self.embeddings @ vector
        if exclude_idx is not None:
            scores[exclude_idx] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]

        result = self.catalog.iloc[top].copy()
        result["similarity"] = np.round(scores[top], 4)
        return result.reset_index(drop=True)

    def recommend_by_text(self, text: str, k: int = 10) -> pd.DataFrame:
        return self.recommend_by_vector(self.embed(text), k=k)

    def recommend_by_series(self, series_id: int, k: int = 10) -> pd.DataFrame:
        matches = np.flatnonzero(self.catalog["series_id"].to_numpy() == series_id)
        if len(matches) == 0:
            raise KeyError(f"series_id {series_id} tidak ada di katalog model.")
        idx = int(matches[0])
        return self.recommend_by_vector(self.embeddings[idx], k=k, exclude_idx=idx)

    # -- HEAD C ------------------------------------------------------------
    def assign_cluster(self, text: str) -> dict:
        """Tentukan arketipe cerita untuk teks masukan."""
        cluster_id = int(self.kmeans.predict(self.embed(text).reshape(1, -1))[0])
        profiles = self.metrics.get("head_c_clustering", {}).get("profiles", [])
        profile = next((p for p in profiles if p["cluster_id"] == cluster_id), {})
        return {"cluster_id": cluster_id, "profile": profile}

    # -- HEAD D ------------------------------------------------------------
    def estimate_rating(
        self,
        text: str,
        chapter_count: float | None = None,
        year: int | None = None,
        is_completed: bool = False,
        type_origin: str = "CN",
        novel_type: str = "Web Novel",
    ) -> float:
        """
        Estimasi rating berbasis konten.

        Vektor fitur numerik disusun ulang MENGIKUTI URUTAN NAMA KOLOM yang
        disimpan saat training. Menyusunnya berdasarkan posisi akan diam-diam
        salah begitu kategori `type_origin` di data berubah.
        """
        if self.rating_regressor is None:
            raise ModelNotTrainedError("Regressor rating tidak tersedia pada artefak ini.")

        feature_names: list[str] = self.metrics.get("numeric_features", [])
        tokens = tokenize(text, self.stopwords)
        catalog_median_chapters = float(self.catalog["chapter_count"].median())
        catalog_median_year = float(pd.to_numeric(self.catalog["year"], errors="coerce").median())

        values = {
            "log_chapters": float(np.log1p(chapter_count if chapter_count else catalog_median_chapters)),
            "n_tags": float(len([t for t in text.split(",") if t.strip()])),
            "token_count": float(len(tokens)),
            "year": float(year if year else catalog_median_year),
            "is_completed": float(bool(is_completed)),
            f"origin_{type_origin}": 1.0,
            f"type_{novel_type}": 1.0,
        }
        numeric = np.array([values.get(name, 0.0) for name in feature_names], dtype=np.float32)

        features = np.hstack([self.embed(text), numeric]).reshape(1, -1)
        prediction = float(self.rating_regressor.predict(features)[0])
        return float(np.clip(prediction, 1.0, 5.0))


def artifacts_available(artifact_dir: Path = ARTIFACT_DIR) -> bool:
    """Cek cepat apakah Fase 4 sudah pernah dijalankan."""
    return all((artifact_dir / name).is_file() for name in REQUIRED_FILES)


def load_registry(artifact_dir: Path = ARTIFACT_DIR) -> ModelRegistry:
    """
    Muat seluruh artefak model.

    Raises:
        ModelNotTrainedError: bila ada artefak wajib yang belum tersedia.
    """
    import joblib

    missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise ModelNotTrainedError(
            "Artefak model belum lengkap (hilang: " + ", ".join(missing) + "). "
            "Jalankan: docker compose exec pipeline python -m src.phases.phase4_model"
        )

    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    regressor_path = artifact_dir / "rating_regressor.joblib"

    registry = ModelRegistry(
        vectorizer=joblib.load(artifact_dir / "vectorizer.joblib"),
        svd=joblib.load(artifact_dir / "svd.joblib"),
        genre_models=joblib.load(artifact_dir / "genre_models.joblib"),
        genre_binarizer=joblib.load(artifact_dir / "genre_binarizer.joblib"),
        kmeans=joblib.load(artifact_dir / "kmeans.joblib"),
        rating_regressor=joblib.load(regressor_path) if regressor_path.is_file() else None,
        catalog=pd.read_parquet(artifact_dir / "catalog.parquet"),
        embeddings=np.load(artifact_dir / "embeddings.npy"),
        metrics=metrics,
        stopwords=set(metrics.get("stopwords", [])),
    )
    logger.info(
        "Registry dimuat: %s novel, embedding %s, %d genre, %d model genre.",
        f"{len(registry.catalog):,}",
        registry.embeddings.shape,
        len(registry.genre_binarizer.classes_),
        len(registry.genre_models),
    )
    return registry


__all__ = [
    "ModelRegistry",
    "ModelNotTrainedError",
    "load_registry",
    "artifacts_available",
    "build_query_text",
]
