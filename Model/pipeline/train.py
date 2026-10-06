"""
Orkestrasi pelatihan & protokol evaluasi
========================================
Satu tempat yang memastikan setiap arsitektur diperlakukan sama persis —
tanpa itu, perbandingan antar model tidak bermakna.

Protokolnya:

  1. Data dibagi **80% pengembangan / 20% uji**. Set uji tidak disentuh sampai
     langkah terakhir.
  2. Untuk tiap kandidat hyperparameter, probabilitas *out-of-fold* dihitung
     lewat k-fold CV pada set pengembangan. Ambang per genre disetel di atas
     probabilitas OOF itu, lalu macro-F1 OOF menjadi skor pemilihan.
  3. Konfigurasi terbaik dilatih ulang pada SELURUH set pengembangan.
  4. Angka yang dilaporkan diukur pada set uji — sekali, di akhir.

Kenapa OOF dan bukan set validasi terpisah: dengan hanya 4.460 dokumen,
menyisihkan 20% lagi berarti kehilangan ~900 sampel latih dan macro-F1 genre
langka langsung ikut turun. OOF memberi estimasi yang sama jujurnya tanpa
mengorbankan data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from Model.pipeline import evaluate as ev
from Model.pipeline import models as mdl
from Model.pipeline.config import (
    AugmentationConfig,
    CV_FOLDS,
    FeatureConfig,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.common.logging_utils import get_logger

logger = get_logger("model.pipeline")


def out_of_fold_probabilities(estimator, frame: pd.DataFrame, y: np.ndarray, folds: int = CV_FOLDS) -> np.ndarray:
    """
    Probabilitas out-of-fold untuk seluruh set pengembangan.

    Setiap baris diprediksi oleh model yang TIDAK melihatnya saat dilatih,
    sehingga probabilitas ini bebas kebocoran dan layak dipakai untuk dua hal
    sekaligus: memilih hyperparameter dan menyetel ambang.
    """
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(estimator, frame, y, cv=splitter, method="predict_proba", n_jobs=1)
    if isinstance(proba, list):
        proba = np.column_stack([p[:, 1] for p in proba])
    return np.asarray(proba)


def split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Bagi indeks menjadi set pengembangan dan set uji (deterministik)."""
    return train_test_split(np.arange(n_rows), test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_single(
    name: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_config: FeatureConfig | None = None,
    aug_config: AugmentationConfig | None = None,
    param_grid: list[dict] | None = None,
) -> tuple[object, np.ndarray, dict]:
    """Latih satu arsitektur: cari konfigurasi, setel ambang, uji sekali."""
    meta = mdl.MODEL_CATALOG[name]
    grid = param_grid if param_grid is not None else mdl.PARAM_GRID[name]
    x_dev, x_test = frame.iloc[dev_idx], frame.iloc[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]

    logger.info("  [%s] %s — mencari konfigurasi terbaik ...", name, meta["label"])
    best = {"score": -1.0, "params": grid[0], "thresholds": None}

    for params in grid:
        candidate = mdl.build_model(name, feature_config, aug_config, **params)
        prob_oof = out_of_fold_probabilities(candidate, x_dev, y_dev)
        thresholds = ev.tune_thresholds(y_dev, prob_oof, labels)
        score = ev.evaluate(y_dev, prob_oof, thresholds)["f1_macro"]
        logger.info("      %s -> makro-F1 OOF %.4f", params, score)
        if score > best["score"]:
            best = {"score": score, "params": params, "thresholds": thresholds}

    logger.info("  [%s] konfigurasi terpilih: %s", name, best["params"])
    model = mdl.build_model(name, feature_config, aug_config, **best["params"])
    model.fit(x_dev, y_dev)

    thresholds = best["thresholds"]
    prob_test = mdl.predict_proba_frame(model, x_test)
    scores = ev.evaluate(y_test, prob_test, thresholds)
    # Pembanding: model yang sama dengan ambang seragam 0,5. Selisihnya
    # memperlihatkan kontribusi penyetelan ambang per genre.
    fixed = ev.evaluate(y_test, prob_test, np.full(len(labels), 0.5, dtype=np.float32))

    scores.update(
        {
            "f1_micro_fixed_threshold": fixed["f1_micro"],
            "f1_macro_fixed_threshold": fixed["f1_macro"],
            "f1_macro_oof": round(float(best["score"]), 4),
            "params": {k: str(v) for k, v in best["params"].items()},
            "threshold_mean": round(float(thresholds.mean()), 3),
            "f1_per_label": ev.per_label_scores(y_test, prob_test, thresholds, labels),
            **meta,
        }
    )
    logger.info(
        "  [%s] UJI: F1 makro=%.4f mikro=%.4f P@3=%.4f | ambang seragam 0,5: makro=%.4f",
        name, scores["f1_macro"], scores["f1_micro"], scores["precision_at_3"], fixed["f1_macro"],
    )
    return model, thresholds, scores


def train_all(
    frame: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    names: list[str] | None = None,
    feature_config: FeatureConfig | None = None,
    aug_config: AugmentationConfig | None = None,
) -> tuple[dict, dict]:
    """Latih seluruh arsitektur pada pembagian data yang identik."""
    from Model.pipeline.embeddings import reserve

    # Seluruh korpus (fold CV + fit akhir + uji) harus muat di memo embedding;
    # ~40 ribu x 384 float32 = ~60 MB.
    reserve(2 * len(frame))
    artifacts, metrics = {}, {}
    for name in names or list(mdl.MODEL_CATALOG):
        model, thresholds, scores = train_single(
            name, frame, y, labels, dev_idx, test_idx, feature_config, aug_config
        )
        # `labels` ikut disimpan per model: penggabungan latih-parsial
        # (Model/train.py --models) harus bisa memastikan semua model berbagi
        # ruang label yang sama dengan genre_binarizer.
        artifacts[name] = {"pipeline": model, "thresholds": thresholds, "labels": list(labels)}
        metrics[name] = scores

    # Dipilih dari skor OOF set dev, bukan skor uji: memilih model berdasarkan
    # set uji membuat skor uji pemenang bias optimistis (set uji ikut seleksi).
    best = max(metrics, key=lambda k: metrics[k]["f1_macro_oof"])
    logger.info("Model terbaik menurut makro-F1 OOF: %s (OOF %.4f, uji %.4f)",
                best, metrics[best]["f1_macro_oof"], metrics[best]["f1_macro"])
    return artifacts, {"models": metrics, "best_model": best, "labels": labels}
