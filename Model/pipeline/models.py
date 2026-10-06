"""
Definisi model & katalog arsitektur
===================================
Tiga arsitektur mengikuti pipeline pembanding yang lazim pada penelitian
klasifikasi genre dari sinopsis: dua baseline berbasis fitur (TF-IDF + model
linear) dan satu model berbasis urutan.

Modul ini juga memuat dua pembungkus yang membuat teknik dari literatur bisa
dipakai di dalam pipeline scikit-learn standar:

* `AugmentedClassifier` — menerapkan augmentasi teks (EDA random deletion)
  hanya pada data latih, sehingga tetap sah dipakai di dalam cross-validation.
* `MLSmoteClassifier` — menerapkan MLSMOTE di ruang fitur sebelum klasifikator
  dilatih.

Keduanya berbentuk estimator biasa, jadi seluruh protokol evaluasi
(cross_val_predict, penyetelan ambang, perbandingan antar model) berlaku sama
untuk model dengan maupun tanpa augmentasi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from Model.pipeline import augmentation, networks
from Model.pipeline.config import AugmentationConfig, FeatureConfig, RANDOM_STATE
from Model.pipeline.features import build_feature_union, label_specific_selector

ARCHITECTURES = networks.ARCHITECTURES
TorchMultiLabelClassifier = networks.TorchMultiLabelClassifier


# Pembungkus teknik dari literatur
class MLSmoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Terapkan MLSMOTE pada matriks fitur sebelum melatih klasifikator dasar.

    Diletakkan sebagai langkah TERAKHIR pipeline (setelah vectorizer) karena
    MLSMOTE bekerja di ruang fitur. Hanya `fit` yang terpengaruh; `predict_proba`
    diteruskan apa adanya ke klasifikator dasar.
    """

    def __init__(self, base_estimator=None, aug_config: AugmentationConfig | None = None):
        self.base_estimator = base_estimator
        self.aug_config = aug_config

    def fit(self, X, y):
        config = self.aug_config or AugmentationConfig(mlsmote=True)
        features, labels = augmentation.mlsmote(X, np.asarray(y), config)
        self.estimator_ = clone(self.base_estimator).fit(features, labels)
        self.classes_ = np.arange(np.asarray(y).shape[1])
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    """
    Terapkan augmentasi teks pada data latih sebelum pipeline dalam dilatih.

    Penting: augmentasi hanya boleh menyentuh data latih. Karena pembungkus ini
    berupa estimator, `cross_val_predict` otomatis memanggilnya per fold —
    sehingga data validasi tiap fold tetap bersih dan skor OOF tetap jujur.
    """

    def __init__(self, base_estimator=None, aug_config: AugmentationConfig | None = None):
        self.base_estimator = base_estimator
        self.aug_config = aug_config

    def fit(self, X: pd.DataFrame, y):
        config = self.aug_config or AugmentationConfig(random_deletion=True)
        frame, labels = augmentation.random_deletion(X.reset_index(drop=True), np.asarray(y), config)
        self.estimator_ = clone(self.base_estimator).fit(frame, labels)
        self.classes_ = np.arange(np.asarray(y).shape[1])
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


# Konstruksi model
def _linear_head(name: str, params: dict, feature_config: FeatureConfig):
    """Klasifikator One-vs-Rest untuk arsitektur berbasis fitur."""
    if name == "logreg":
        base = LogisticRegression(
            max_iter=3000,
            C=params.get("C", 4.0),
            class_weight=params.get("class_weight", "balanced"),
        )
    else:  # svm
        # LinearSVC tidak menghasilkan probabilitas; kalibrasi sigmoid (Platt)
        # mengubah jarak ke hyperplane menjadi probabilitas yang layak
        # ditampilkan ke pengguna dan dipakai untuk penyetelan ambang.
        base = CalibratedClassifierCV(
            LinearSVC(
                C=params.get("C", 0.3),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=5000,
                dual="auto",
                # dual coordinate descent liblinear mengacak urutan sampel;
                # tanpa seed, skor svm/hybrid bergeser antar run.
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        )

    selector = label_specific_selector(feature_config.label_specific_k)
    if selector is not None:
        # Seleksi fitur ditempatkan DI DALAM One-vs-Rest, sehingga tiap label
        # memilih fiturnya sendiri (semangat LIFT, Zhang & Wu 2015).
        base = Pipeline([("select", selector), ("clf", base)])

    return OneVsRestClassifier(base, n_jobs=2)


def build_model(
    name: str,
    feature_config: FeatureConfig | None = None,
    aug_config: AugmentationConfig | None = None,
    **params,
):
    """Bangun estimator lengkap untuk satu konfigurasi arsitektur."""
    feature_config = feature_config or FeatureConfig()
    aug_config = aug_config or AugmentationConfig()

    if name in ARCHITECTURES:
        model = TorchMultiLabelClassifier(arch=name, **params)
    elif name in ("logreg", "svm", "hybrid"):
        if name == "hybrid":
            # Blok embedding pralatih dinyalakan lewat konfigurasi fitur, bukan
            # lewat kelas terpisah, supaya seluruh protokol evaluasi (OOF,
            # penyetelan ambang, perbandingan antar model) berlaku apa adanya.
            params = dict(params)
            base_kind = params.pop("base", "svm")
            weights = dict(feature_config.weights)
            weights["embedding"] = params.pop("embedding_weight", weights.get("embedding", 1.0))
            feature_config = replace(feature_config, use_embeddings=True, weights=weights)
            head = _linear_head(base_kind, params, feature_config)
        else:
            head = _linear_head(name, params, feature_config)
        if aug_config.mlsmote:
            head = MLSmoteClassifier(head, aug_config)
        model = Pipeline([("features", build_feature_union(feature_config)), ("clf", head)])
    else:
        raise ValueError(f"Model tidak dikenal: {name!r}")

    if aug_config.random_deletion:
        model = AugmentedClassifier(model, aug_config)
    return model


def predict_proba_frame(estimator, frame: pd.DataFrame) -> np.ndarray:
    """
    Ambil probabilitas per label sebagai matriks (n_samples, n_labels).

    Estimator yang berbeda mengembalikan bentuk berbeda untuk multi-label;
    fungsi ini menyeragamkannya.
    """
    proba = estimator.predict_proba(frame)
    if isinstance(proba, list):
        proba = np.column_stack([p[:, 1] for p in proba])
    proba = np.asarray(proba)
    if proba.ndim == 3:  # (n_labels, n_samples, 2)
        proba = proba[:, :, 1].T
    return proba


# Katalog & ruang pencarian
MODEL_CATALOG = {
    "logreg": {
        "label": "Logistic Regression",
        "family": "Linear · TF-IDF sparse",
        "description": (
            "Model linear pada ruang TF-IDF. Paling cepat dan probabilitasnya "
            "paling terkalibrasi; pilihan aman untuk penggunaan sehari-hari."
        ),
    },
    "svm": {
        "label": "Linear SVM",
        "family": "Margin-based · TF-IDF sparse",
        "description": (
            "Memaksimalkan margin antar kelas, secara empiris paling tajam untuk "
            "teks berdimensi tinggi. Probabilitas diperoleh lewat kalibrasi Platt."
        ),
    },
    "hybrid": {
        "label": "Hybrid TF-IDF + Embedding",
        "family": "Sparse + padat · encoder pralatih",
        "description": (
            "Menggabungkan TF-IDF dengan embedding kalimat pralatih (BGE). "
            "TF-IDF menangkap frasa khas genre, embedding menangkap parafrasa "
            "yang kata-katanya berbeda — keduanya saling melengkapi."
        ),
        "reference": "Xiao et al. (2023), BGE; Reimers & Gurevych (2019)",
    },
    "bilstm": {
        "label": "BiLSTM + Attention",
        "family": "Deep learning · rekuren dua arah",
        "description": (
            "Membaca sinopsis sebagai urutan dari dua arah, lalu attention "
            "memberi bobot lebih besar pada kata penentu genre "
            "(Ertugrul & Karagoz 2018; Zhou et al. 2016)."
        ),
        "reference": "Ertugrul & Karagoz (2018); Zhou et al. (2016)",
    },
}

#: Ruang pencarian hyperparameter per arsitektur.
#:
#: Sengaja kecil dan beralasan, bukan grid besar acak. `class_weight` diuji dua
#: arah karena berinteraksi dengan penyetelan ambang: "balanced" sudah menggeser
#: probabilitas, sehingga menyetel ambang di atasnya bisa menjadi koreksi ganda.
PARAM_GRID = {
    "logreg": [
        {"C": 4.0, "class_weight": "balanced"},
        {"C": 12.0, "class_weight": "balanced"},
        {"C": 4.0, "class_weight": None},
    ],
    "svm": [
        {"C": 0.3, "class_weight": "balanced"},
        {"C": 0.8, "class_weight": "balanced"},
    ],
    # Bobot blok embedding adalah satu-satunya hyperparameter baru di sini.
    # Sapuan OOF (0,5 / 1 / 2 / 3 / 5) memperlihatkan puncak yang landai di
    # sekitar 1-2 lalu turun tajam di 5 — embedding yang terlalu dominan
    # menenggelamkan frasa khas yang justru jadi kekuatan TF-IDF. Tiga
    # kandidat yang tersisa mengapit puncak itu; pemenangnya dipilih OOF.
    "hybrid": [
        {"base": "svm", "C": 0.3, "class_weight": "balanced", "embedding_weight": 2.0},
        {"base": "svm", "C": 0.3, "class_weight": "balanced", "embedding_weight": 1.0},
        {"base": "logreg", "C": 4.0, "class_weight": "balanced", "embedding_weight": 1.0},
    ],
    # Model deep learning memakai hiperparameter default dari makalah rujukan.
    # Gridnya tipis: satu kandidat sudah butuh 4 kali pelatihan jaringan
    # (3 fold CV + model akhir), sehingga grid lebar tidak sebanding biayanya.
    "bilstm": [
        {"max_len": 250, "embed_dim": 128, "epochs": 20, "hidden": 128, "loss": "bce"},
        {"max_len": 250, "embed_dim": 128, "epochs": 20, "hidden": 128, "loss": "asl"},
    ],
}
