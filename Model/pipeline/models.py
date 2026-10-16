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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from Model.pipeline import augmentation, networks
from Model.pipeline.config import AugmentationConfig, FeatureConfig, N_JOBS, RANDOM_STATE
from Model.pipeline.features import (
    MultiLabelChi2Selector,
    build_feature_union,
    label_specific_selector,
)

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

    return OneVsRestClassifier(base, n_jobs=N_JOBS)


def _random_forest(params: dict, feature_config: FeatureConfig) -> Pipeline:
    """
    Satu hutan multi-output untuk seluruh genre, gaya RF-PCT yang unggul pada
    perbandingan 12 metode multi-label Madjarov dkk. (2012); lebih kecil dari 35 hutan.
    """
    forest = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 60),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_leaf_nodes=params.get("max_leaf_nodes", 2000),
        # Tanpa class_weight: di hutan multi-output bobot 35 genre dikalikan per sampel
        # (balanced_subsample 0,5275 vs tanpa bobot 0,5607 makro-F1 OOF).
        class_weight=params.get("class_weight"),
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            ("features", build_feature_union(feature_config)),
            ("select", MultiLabelChi2Selector(params.get("k_features", 3000))),
            ("clf", forest),
        ]
    )


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
    elif name == "rf":
        model = _random_forest(params, feature_config)
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


def shrink_coefficients(model) -> int:
    """
    Simpan koefisien model linear sebagai float32; kembalikan jumlah estimator yang diubah.

    Separuh memori dan waktu muat dashboard; diukur pada 3.000 judul, probabilitas bergeser
    paling banyak 3e-8 dan 0 dari 105.000 keputusan genre berubah.
    """
    changed = 0
    for _, step in getattr(model, "steps", []):
        changed += shrink_coefficients(step)
    children = list(getattr(model, "estimators_", None) or [])
    children += [c.estimator for c in getattr(model, "calibrated_classifiers_", None) or []]
    for child in children:
        changed += shrink_coefficients(child)
    coef = getattr(model, "coef_", None)
    if isinstance(coef, np.ndarray) and coef.dtype == np.float64:
        model.coef_ = coef.astype(np.float32)
        changed += 1
    return changed


def share_feature_steps(artifacts: dict) -> list[str]:
    """
    Pakai satu objek untuk langkah `features` yang identik byte demi byte antar pipeline.

    Kamus kosakata TF-IDF jauh lebih besar di RAM daripada di pickle; berbagi satu salinan
    memangkas waktu muat genre_models dari 7,2 ke 3,1 detik tanpa mengubah prediksi.
    """
    import pickle

    shared, names = {}, []
    for name, bundle in artifacts.items():
        pipe = bundle.get("pipeline") if isinstance(bundle, dict) else None
        if pipe is None or "features" not in getattr(pipe, "named_steps", {}):
            continue
        idx = [i for i, (step, _) in enumerate(pipe.steps) if step == "features"][0]
        key = pickle.dumps(pipe.steps[idx][1], protocol=pickle.HIGHEST_PROTOCOL)
        if key in shared:
            pipe.steps[idx] = ("features", shared[key])
            names.append(name)
        else:
            shared[key] = pipe.steps[idx][1]
    return names


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
    "rf": {
        "label": "Random Forest",
        "family": "Ensemble pohon · TF-IDF terseleksi chi-kuadrat",
        "description": (
            "Satu hutan pohon keputusan multi-output untuk seluruh genre "
            "sekaligus, di atas fitur TF-IDF yang disaring chi-kuadrat. "
            "Menangkap interaksi antar kata yang tidak terlihat model linear, "
            "tetapi kurang tajam pada teks jarang berdimensi tinggi."
        ),
        "reference": "Breiman (2001); Madjarov dkk. (2012)",
    },
    "ensemble": {
        "label": "Ensemble SVM + Random Forest",
        "family": "Rata-rata log-odds berbobot · linear + pohon",
        "description": (
            "Menggabungkan Linear SVM dan Random Forest dengan merata-ratakan log-odds "
            "keduanya; bobotnya dipilih di prediksi out-of-fold. F1 makro dan mikronya "
            "tertinggi, dan urutan tiga genre teratasnya setara dengan SVM saja."
        ),
        "reference": "Wolpert (1992); Genest dkk. (1984)",
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

#: Arsitektur yang dilatih dan disajikan bila tidak diminta lain: ML klasik saja.
#: `hybrid` (encoder neural pralatih) dan `bilstm` (PyTorch) hanya dilatih bila namanya diminta.
DEFAULT_MODELS = ("logreg", "svm", "rf", "ensemble")

#: Anggota ensemble; dirakit dari probabilitas model yang sudah dilatih, tanpa fit baru.
ENSEMBLE_MEMBERS = ("svm", "rf")
#: Uji 5 seed, dibanding SVM saja: rata-rata probabilitas P@3 -0,0040 (p=0,02), log-odds -0,0014
#: (p=0,14) dengan makro-F1 yang sama.
ENSEMBLE_SPACE = "logit"

#: Ruang pencarian hyperparameter per arsitektur.
#:
#: Kandidat SVM diambil dari puncak pencarian TPE (Bergstra dkk. 2011) atas C,
#: class_weight, dan bobot blok fitur; LogReg disapu C di bobot blok yang sama.
#: class_weight=None menang di keduanya: ambang per genre sudah mengoreksi
#: ketimpangan, jadi "balanced" menjadi koreksi ganda.
PARAM_GRID = {
    "logreg": [
        {"C": 0.5, "class_weight": None},
        {"C": 0.25, "class_weight": None},
    ],
    "svm": [
        {"C": 0.0354, "class_weight": None},
        {"C": 0.06, "class_weight": None},
    ],
    # Tiap simpul menyimpan probabilitas 35 genre (560 byte). 60 pohon x 2.000 daun (±143 MB RAM)
    # lebih lemah sendirian dari 150 x 3.000 (±535 MB), tetapi ensemble-nya setara (OOF 0,6057 vs 0,6061).
    "rf": [
        {"n_estimators": 60, "max_leaf_nodes": 2000, "k_features": 3000},
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
