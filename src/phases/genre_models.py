"""
HEAD A — Genre Tagger: tiga arsitektur yang dapat dipilih pengguna
=================================================================

Modul ini menangani satu tugas: memprediksi HIMPUNAN genre sebuah novel dari
teksnya (judul + sinopsis + tag) beserta metadata ringan (asal & tipe).

Kenapa tiga model, bukan satu
-----------------------------
Tidak ada satu algoritma yang menang di semua situasi. Ketiganya sengaja
dipilih dengan *inductive bias* yang benar-benar berbeda, sehingga pengguna
bisa memilih sesuai kebutuhan — dan perbedaan hasilnya informatif, bukan
sekadar variasi acak:

  1. **Logistic Regression (OvR)** — linear di ruang TF-IDF sparse.
     Cepat, probabilitasnya terkalibrasi baik, dan koefisiennya bisa ditelusuri.
     Basis pembanding yang jujur.
  2. **Linear SVM terkalibrasi** — memaksimalkan margin, secara empiris paling
     kuat untuk klasifikasi teks berdimensi tinggi. LinearSVC sendiri tidak
     menghasilkan probabilitas, jadi dibungkus kalibrasi sigmoid (Platt).
  3. **Neural Network (MLP)** — dua lapis tersembunyi di atas embedding LSA
     yang padat. Satu-satunya dari ketiganya yang bisa menangkap interaksi
     non-linear antar konsep (mis. "sekolah" + "kekuatan super" -> Shounen),
     dengan biaya interpretabilitas.

Peningkatan yang berlaku untuk KETIGANYA
----------------------------------------
* **Stemming (Snowball).** Tanpa ini "cultivation", "cultivating", dan
  "cultivator" jadi tiga fitur terpisah yang masing-masing lemah.
* **Blok fitur terpisah dengan bobot.** Tag adalah label kurasi manusia
  (presisi tinggi), sinopsis adalah prosa bebas (banyak derau). Menggabungkan
  keduanya dalam satu TF-IDF membuat sinyal tag tenggelam; di sini keduanya
  divektorkan terpisah lalu tag diberi bobot lebih besar.
* **Metadata sebagai fitur.** Asal negara sangat menentukan genre — Xianxia
  praktis hanya ada pada novel Tiongkok. Informasi ini tersedia saat prediksi,
  jadi tidak memakainya berarti membuang sinyal gratis.
* **Ambang keputusan per label.** Ini peningkatan terbesar untuk multi-label:
  ambang 0,5 yang seragam menghukum genre langka, yang probabilitasnya nyaris
  tidak pernah mencapai 0,5. Setiap genre mendapat ambangnya sendiri, dioptimasi
  pada probabilitas out-of-fold — bukan pada set uji, agar angka akhir tetap jujur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.common.logging_utils import get_logger

logger = get_logger("phase4.genre")

RANDOM_STATE = 42

#: Kolom masukan yang HARUS ada saat training maupun inferensi.
FEATURE_COLUMNS = ("corpus_clean", "tags_clean", "type_origin", "type")

_STEMMER = SnowballStemmer("english")
_STEM_CACHE: dict[str, str] = {}


def _stem_word(word: str) -> str:
    """Stemming dengan memo — korpus ini banyak mengulang kata yang sama."""
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = _STEMMER.stem(word)
        _STEM_CACHE[word] = cached
    return cached


def stem_texts(texts) -> list[str]:
    """
    Stemming pada level dokumen.

    Dijadikan fungsi tingkat modul (bukan lambda atau closure) karena harus
    dapat di-pickle: fungsi ini ikut tersimpan di dalam artefak pipeline dan
    dipanggil ulang saat inferensi.
    """
    series = pd.Series(texts).fillna("").astype(str)
    return [" ".join(_stem_word(token) for token in doc.split()) for doc in series]


def _text_branch(min_df: int, max_features: int) -> Pipeline:
    """Cabang fitur teks: stemming -> TF-IDF 1-2 gram."""
    return Pipeline(
        [
            ("stem", FunctionTransformer(stem_texts, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.85,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_feature_union() -> ColumnTransformer:
    """
    Susun ruang fitur gabungan.

    Bobot ditentukan dari sifat sumbernya, bukan ditebak: `tags_clean` adalah
    label kurasi yang presisi sehingga diberi bobot lebih tinggi meski
    tokennya jauh lebih sedikit daripada sinopsis.
    """
    return ColumnTransformer(
        transformers=[
            ("corpus", _text_branch(min_df=3, max_features=40_000), "corpus_clean"),
            ("tags", _text_branch(min_df=2, max_features=10_000), "tags_clean"),
            (
                "meta",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                ["type_origin", "type"],
            ),
        ],
        transformer_weights={"corpus": 1.0, "tags": 1.6, "meta": 0.6},
        sparse_threshold=1.0,
    )


# ===========================================================================
# Definisi ketiga model
# ===========================================================================
def build_model(name: str, **params) -> Pipeline:
    """Bangun pipeline lengkap (fitur + klasifikator) untuk satu konfigurasi."""
    features = build_feature_union()

    if name == "logreg":
        classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=3000,
                C=params.get("C", 8.0),
                class_weight=params.get("class_weight", "balanced"),
            ),
            n_jobs=2,
        )
        return Pipeline([("features", features), ("clf", classifier)])

    if name == "svm":
        # LinearSVC tidak punya predict_proba; kalibrasi sigmoid (Platt scaling)
        # mengubah jarak ke hyperplane menjadi probabilitas yang layak
        # ditampilkan ke pengguna.
        classifier = OneVsRestClassifier(
            CalibratedClassifierCV(
                LinearSVC(
                    C=params.get("C", 0.6),
                    class_weight=params.get("class_weight", "balanced"),
                    max_iter=5000,
                    dual="auto",
                ),
                method="sigmoid",
                cv=3,
            ),
            n_jobs=2,
        )
        return Pipeline([("features", features), ("clf", classifier)])

    if name == "mlp":
        # MLP butuh masukan padat & terskala. SVD memampatkan TF-IDF sparse
        # menjadi ruang laten sekaligus meredam derau frasa langka.
        return Pipeline(
            [
                ("features", features),
                ("svd", TruncatedSVD(n_components=params.get("svd", 240), random_state=RANDOM_STATE)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=params.get("hidden", (384, 192)),
                        activation="relu",
                        alpha=params.get("alpha", 1e-4),
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=400,
                        early_stopping=True,
                        n_iter_no_change=15,
                        validation_fraction=0.12,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(f"Model tidak dikenal: {name!r}")


#: Ruang pencarian hyperparameter per arsitektur.
#:
#: Sengaja kecil dan beralasan, bukan grid besar acak:
#:   * `class_weight` diuji dua arah karena berinteraksi dengan penyetelan
#:     ambang — "balanced" sudah menggeser probabilitas, sehingga menyetel
#:     ambang di atasnya bisa jadi koreksi ganda yang justru merugikan.
#:   * `C` mengatur kekuatan regularisasi; korpus sparse berdimensi tinggi
#:     biasanya menyukai regularisasi lebih longgar untuk model linear dan
#:     lebih ketat untuk SVM.
#:   * Untuk MLP yang divariasikan adalah kapasitas: dimensi laten dan lebar
#:     lapisan. Menambah dimensi SVD mengembalikan informasi yang hilang saat
#:     pemampatan, tetapi menaikkan risiko overfit pada 3.568 sampel.
PARAM_GRID = {
    "logreg": [
        {"C": 4.0, "class_weight": "balanced"},
        {"C": 12.0, "class_weight": "balanced"},
        {"C": 4.0, "class_weight": None},
        {"C": 12.0, "class_weight": None},
    ],
    "svm": [
        {"C": 0.3, "class_weight": "balanced"},
        {"C": 0.8, "class_weight": "balanced"},
        {"C": 0.8, "class_weight": None},
    ],
    "mlp": [
        {"svd": 240, "hidden": (384, 192), "alpha": 1e-4},
        {"svd": 400, "hidden": (512, 256), "alpha": 3e-4},
    ],
}


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
    "mlp": {
        "label": "Neural Network (MLP)",
        "family": "Deep learning · embedding LSA 240-d",
        "description": (
            "Jaringan saraf dua lapis di atas embedding laten. Menangkap interaksi "
            "non-linear antar konsep cerita, dengan biaya interpretabilitas."
        ),
    },
}


# ===========================================================================
# Ambang keputusan per label
# ===========================================================================
def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Cari ambang optimal untuk setiap genre pada set validasi.

    Ambang tunggal 0,5 mengasumsikan setiap genre sama seringnya. Kenyataannya
    Romance muncul di 56% katalog sementara Mecha di bawah 2% — probabilitas
    Mecha nyaris tak pernah menyentuh 0,5 sehingga genre itu tidak pernah
    diprediksi sama sekali. Mengoptimalkan ambang per genre memperbaiki
    macro-F1 secara langsung.

    Rentang dibatasi 0,05-0,80: di bawah itu model memuntahkan hampir semua
    genre untuk setiap novel (presisi runtuh), yang secara metrik terlihat
    lumayan tetapi tidak berguna bagi pengguna.
    """
    grid = np.arange(0.05, 0.81, 0.025)
    thresholds = np.full(len(labels), 0.5, dtype=np.float32)

    for index in range(len(labels)):
        best_score, best_threshold = -1.0, 0.5
        for threshold in grid:
            score = f1_score(y_true[:, index], (y_prob[:, index] >= threshold).astype(int), zero_division=0)
            if score > best_score:
                best_score, best_threshold = score, float(threshold)
        thresholds[index] = best_threshold

    return thresholds


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    """Hitung metrik multi-label pada ambang tertentu."""
    y_pred = (y_prob >= thresholds).astype(int)
    return {
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_samples": round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4),
    }


def predict_proba_frame(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """
    Ambil probabilitas per label dalam bentuk matriks (n_samples, n_labels).

    OneVsRestClassifier dan MLPClassifier mengembalikan bentuk yang berbeda
    untuk kasus multi-label; fungsi ini menyeragamkannya.
    """
    proba = pipeline.predict_proba(frame)
    if isinstance(proba, list):  # sebagian estimator mengembalikan list per label
        proba = np.column_stack([p[:, 1] for p in proba])
    return np.asarray(proba)


def _oof_probabilities(pipeline: Pipeline, frame: pd.DataFrame, y: np.ndarray, folds: int = 3) -> np.ndarray:
    """
    Probabilitas *out-of-fold* pada seluruh data pengembangan.

    Setiap baris diprediksi oleh model yang TIDAK melihatnya saat dilatih,
    sehingga probabilitas ini bebas dari kebocoran dan layak dipakai untuk dua
    hal sekaligus: memilih hyperparameter dan menyetel ambang. Keuntungannya
    dibanding menyisihkan satu set validasi tetap: seluruh 80% data tetap
    terpakai untuk melatih model akhir.
    """
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipeline, frame, y, cv=splitter, method="predict_proba", n_jobs=1)
    if isinstance(proba, list):
        proba = np.column_stack([p[:, 1] for p in proba])
    return np.asarray(proba)


def train_all(
    frame: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, dict]:
    """
    Latih ketiga arsitektur dengan protokol yang identik dan adil.

    Protokol:
      1. Data dibagi **80% pengembangan / 20% uji**. Set uji tidak pernah
         disentuh sampai langkah terakhir.
      2. Untuk setiap kandidat hyperparameter, probabilitas *out-of-fold*
         dihitung lewat 3-fold CV pada set pengembangan. Ambang per genre
         disetel di atas probabilitas OOF itu, lalu macro-F1 OOF dipakai
         sebagai skor pemilihan konfigurasi.
      3. Konfigurasi terbaik dilatih ulang pada SELURUH set pengembangan,
         memakai ambang hasil langkah 2.
      4. Angka yang dilaporkan diukur pada set uji — sekali, di akhir.

    Kenapa OOF, bukan set validasi terpisah: dengan hanya 4.460 dokumen,
    menyisihkan 20% lagi untuk validasi berarti kehilangan ~900 sampel latih
    dan macro-F1 genre langka langsung ikut turun. OOF memberi estimasi yang
    sama jujurnya tanpa mengorbankan data.

    Returns:
        (artefak model, metrik) — artefak berisi pipeline terlatih + ambangnya.
    """
    artifacts: dict[str, dict] = {}
    metrics: dict[str, dict] = {}

    x_dev, x_test = frame.iloc[dev_idx], frame.iloc[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]

    for name, meta in MODEL_CATALOG.items():
        logger.info("  [%s] %s — mencari konfigurasi terbaik ...", name, meta["label"])

        best = {"score": -1.0}
        for params in PARAM_GRID[name]:
            candidate = build_model(name, **params)
            prob_oof = _oof_probabilities(candidate, x_dev, y_dev)
            thresholds = tune_thresholds(y_dev, prob_oof, labels)
            score = evaluate(y_dev, prob_oof, thresholds)["f1_macro"]
            logger.info("      %s -> macro-F1 OOF %.4f", params, score)
            if score > best["score"]:
                best = {"score": score, "params": params, "thresholds": thresholds}

        logger.info("  [%s] konfigurasi terpilih: %s", name, best["params"])
        pipeline = build_model(name, **best["params"])
        pipeline.fit(x_dev, y_dev)

        thresholds = best["thresholds"]
        prob_test = predict_proba_frame(pipeline, x_test)
        scores = evaluate(y_test, prob_test, thresholds)
        # Pembanding: performa model yang sama bila memakai ambang seragam 0,5.
        # Selisihnya memperlihatkan kontribusi penyetelan ambang per genre.
        fixed = evaluate(y_test, prob_test, np.full(len(labels), 0.5, dtype=np.float32))

        per_label = f1_score(
            y_test, (prob_test >= thresholds).astype(int), average=None, zero_division=0
        )
        scores.update(
            {
                "f1_micro_fixed_threshold": fixed["f1_micro"],
                "f1_macro_fixed_threshold": fixed["f1_macro"],
                "f1_macro_oof": round(float(best["score"]), 4),
                "params": {k: str(v) for k, v in best["params"].items()},
                "threshold_mean": round(float(thresholds.mean()), 3),
                "f1_per_label": {
                    label: round(float(score), 4)
                    for label, score in sorted(zip(labels, per_label), key=lambda kv: -kv[1])
                },
                **meta,
            }
        )

        logger.info(
            "  [%s] UJI: F1 micro=%.4f macro=%.4f | ambang seragam 0,5: micro=%.4f macro=%.4f",
            name, scores["f1_micro"], scores["f1_macro"],
            fixed["f1_micro"], fixed["f1_macro"],
        )

        artifacts[name] = {"pipeline": pipeline, "thresholds": thresholds}
        metrics[name] = scores

    best_name = max(metrics, key=lambda k: metrics[k]["f1_macro"])
    logger.info("Model terbaik menurut macro-F1: %s (%.4f)", best_name, metrics[best_name]["f1_macro"])
    return artifacts, {"models": metrics, "best_model": best_name, "labels": labels}
