"""
Model deep learning untuk pelabelan genre multi-label
=====================================================
Arsitektur **BiLSTM + self-attention**, direplikasi dari literatur klasifikasi
genre berbasis sinopsis dan dibungkus sebagai estimator scikit-learn agar
menjalani protokol evaluasi yang sama persis dengan model klasik
(OOF cross-validation + penyetelan ambang per genre).

Rujukan yang diikuti
--------------------
* **Ertugrul & Karagoz (2018), "Movie Genre Classification from Plot Summaries
  Using Bidirectional LSTM"** — BiLSTM atas sinopsis film untuk prediksi genre;
  rujukan utama arsitektur di modul ini.
* **Zhou et al. (2016)** — self-attention di atas keluaran BiLSTM sebagai
  pengganti pooling sederhana, agar kata yang menentukan genre diberi bobot
  lebih besar.
* **Liu et al. (SIGIR 2017), "Deep Learning for Extreme Multi-label Text
  Classification"** — perlakuan multi-label yang dipakai di sini: *binary
  cross-entropy* per label (bukan softmax), dengan bobot kelas positif untuk
  mengimbangi genre langka.
* **Hoang (2018), "Predicting Movie Genres Based on Plot Summaries"** —
  transformasi K-biner + **ambang probabilitas yang dipelajari**; ini yang
  mendasari penyetelan ambang per genre di `genre_models.py`.

Catatan kejujuran: literatur di atas memakai embedding pralatih (GloVe/word2vec)
dan melaporkan bahwa itu membantu. Lingkungan ini offline sehingga embedding
dilatih dari nol bersama tugasnya. Pada korpus 3,5k dokumen latih, itu
keterbatasan nyata — dan menjadi penjelasan paling masuk akal mengapa model ini
masih kalah dari baseline TF-IDF + linear (lihat `Doc/model_card.md`).

Catatan sejarah: modul ini sempat memuat TextCNN/XML-CNN (Kim 2014;
Liu et al. 2017) sebagai arsitektur kedua. Model itu dibuang karena kalah di
seluruh metrik terhadap BiLSTM di sini (makro-F1 0,4483 vs 0,4812;
P@3 0,6484 vs 0,6715), bukan karena tidak berhasil dijalankan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common.logging_utils import get_logger

logger = get_logger("phase4.dl")

# Container ini membatasi thread BLAS ke 2 (lihat Dockerfile); torch harus
# mengikuti agar tidak terjadi oversubscription yang justru memperlambat.
torch.set_num_threads(2)

PAD, UNK = 0, 1
RANDOM_STATE = 42


# ===========================================================================
# Tokenisasi & kosakata
# ===========================================================================
class Vocabulary:
    """
    Kosakata dibangun HANYA dari data latih tiap fold.

    Ini penting: membangun kosakata dari seluruh dataset berarti model sudah
    "melihat" kata-kata milik data uji, dan skor yang dilaporkan jadi optimistis.
    """

    def __init__(self, max_size: int = 30_000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi: dict[str, int] = {}

    def fit(self, texts) -> "Vocabulary":
        counts: dict[str, int] = {}
        for doc in texts:
            for token in doc.split():
                counts[token] = counts.get(token, 0) + 1

        kept = [t for t, c in counts.items() if c >= self.min_freq]
        kept.sort(key=lambda t: (-counts[t], t))
        kept = kept[: self.max_size - 2]

        self.stoi = {"<pad>": PAD, "<unk>": UNK}
        for index, token in enumerate(kept, start=2):
            self.stoi[token] = index
        return self

    def encode(self, texts, max_len: int) -> np.ndarray:
        """Ubah dokumen menjadi matriks indeks berukuran (n, max_len)."""
        matrix = np.zeros((len(texts), max_len), dtype=np.int64)
        for row, doc in enumerate(texts):
            tokens = doc.split()[:max_len]
            for col, token in enumerate(tokens):
                matrix[row, col] = self.stoi.get(token, UNK)
        return matrix

    def __len__(self) -> int:
        return len(self.stoi)


# ===========================================================================
# Arsitektur — BiLSTM dengan self-attention
# ===========================================================================
class BiLSTMAttentionNet(nn.Module):
    """
    BiLSTM + additive self-attention.

    Attention menggantikan pooling biasa: bobot tiap langkah waktu dipelajari,
    sehingga kata penentu genre ("cultivation", "duke", "system") berkontribusi
    lebih besar daripada kata pengisi. Padding di-mask agar tidak ikut dihitung.
    """

    def __init__(
        self,
        vocab_size: int,
        n_labels: int,
        embed_dim: int = 128,
        hidden: int = 128,
        attn_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * hidden, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != PAD).unsqueeze(-1)
        states, _ = self.lstm(self.embedding(x))  # (B, T, 2H)
        scores = self.attention(states)  # (B, T, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        # Dokumen yang seluruhnya padding menghasilkan NaN -> dinolkan.
        weights = torch.nan_to_num(weights)
        context = (weights * states).sum(dim=1)
        return self.output(self.dropout(context))


ARCHITECTURES = {"bilstm": BiLSTMAttentionNet}


# ===========================================================================
# Pembungkus scikit-learn
# ===========================================================================
class TorchMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptor agar model PyTorch dapat dipakai seperti estimator scikit-learn.

    Alasan dibungkus: seluruh protokol evaluasi (cross_val_predict untuk
    probabilitas out-of-fold, penyetelan ambang, perbandingan antar model)
    sudah tertulis untuk API scikit-learn. Dengan pembungkus ini, model deep
    learning menjalani prosedur yang PERSIS SAMA dengan model klasik —
    syarat agar perbandingannya bermakna.

    Masukannya DataFrame dengan kolom `genre_models.FEATURE_COLUMNS`; teks dan
    metadata digabung menjadi satu untai token (metadata diubah jadi token
    khusus seperti `__origin_cn`) sehingga jaringan tetap menerima sinyal asal
    negara yang terbukti berguna pada model klasik.
    """

    def __init__(
        self,
        arch: str = "bilstm",
        max_len: int = 300,
        embed_dim: int = 128,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 4,
        val_fraction: float = 0.12,
        stem: bool = True,
        loss: str = "bce",
        pos_weight_cap: float = 12.0,
        random_state: int = RANDOM_STATE,
        **arch_kwargs,
    ):
        self.arch = arch
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.stem = stem
        self.loss = loss
        self.pos_weight_cap = pos_weight_cap
        self.random_state = random_state
        self.arch_kwargs = arch_kwargs

    # -- Persiapan masukan ------------------------------------------------
    def _to_text(self, frame: pd.DataFrame) -> list[str]:
        """
        Gabungkan korpus, tag, dan metadata menjadi satu untai token.

        Metadata diubah menjadi token khusus (`__origin_cn`) agar jaringan tetap
        menerima sinyal asal negara — pada model klasik fitur ini terbukti
        berguna, dan tidak ada alasan menyembunyikannya dari model urutan.

        Stemming memakai fungsi yang SAMA dengan model berbasis fitur agar
        kedua jalur melihat bentuk kata yang identik. Pada korpus 3,5k dokumen,
        tanpa stemming kosakata terpecah ("cultivation/cultivating/cultivator")
        dan tiap varian jadi terlalu jarang untuk dipelajari embedding-nya.
        """
        corpus = frame["corpus_clean"].fillna("").astype(str)
        tags = frame["tags_clean"].fillna("").astype(str)
        origin = frame["type_origin"].fillna("unknown").astype(str).str.lower()
        kind = frame["type"].fillna("unknown").astype(str).str.lower().str.replace(" ", "_")
        texts = (corpus + " " + tags).tolist()

        if self.stem:
            # Impor tertunda: genre_models mengimpor modul ini di tingkat atas,
            # sehingga impor melingkar hanya dapat dihindari dengan impor lokal.
            from Model.pipeline.preprocessing import stem_texts

            texts = stem_texts(texts)

        return [
            f"{text} __origin_{o} __type_{k}"
            for text, o, k in zip(texts, origin, kind)
        ]

    def _compute_pos_weight(self, y: np.ndarray) -> torch.Tensor:
        """
        Bobot kelas positif per label untuk BCE.

        Tanpa ini, genre yang hanya muncul di 2% dokumen nyaris tidak
        memberikan gradien: menebak "bukan" sudah menghasilkan loss rendah,
        sehingga jaringan berhenti belajar genre langka dan macro-F1 anjlok.
        Bobot dibatasi (`pos_weight_cap`) karena rasio mentah pada genre
        paling langka bisa mencapai puluhan dan membuat pelatihan tidak stabil.
        """
        positives = y.sum(axis=0)
        negatives = len(y) - positives
        ratio = np.divide(
            negatives, np.maximum(positives, 1), out=np.ones_like(positives), where=positives > 0
        )
        return torch.tensor(np.clip(ratio, 1.0, self.pos_weight_cap), dtype=torch.float32)

    def _build_network(self, n_labels: int) -> nn.Module:
        return ARCHITECTURES[self.arch](
            vocab_size=len(self.vocab_),
            n_labels=n_labels,
            embed_dim=self.embed_dim,
            **self.arch_kwargs,
        )


    def _build_criterion(self, y_train: np.ndarray):
        """
        Pilih fungsi loss.

        `asl` = Asymmetric Loss (Ridnik et al., ICCV 2021), yang menurunkan
        bobot negatif mudah secara dinamis. `bce` = binary cross-entropy dengan
        bobot positif; keduanya menyasar ketimpangan positif-negatif, tetapi ASL
        melakukannya per-contoh alih-alih per-label.
        """
        if self.loss == "asl":
            from Model.pipeline.losses import AsymmetricLoss

            return AsymmetricLoss()
        return nn.BCEWithLogitsLoss(pos_weight=self._compute_pos_weight(y_train))

    # -- Pelatihan --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TorchMultiLabelClassifier":
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        texts = self._to_text(X)
        y = np.asarray(y, dtype=np.float32)
        self.n_labels_ = y.shape[1]
        self.classes_ = np.arange(self.n_labels_)

        self.vocab_ = Vocabulary().fit(texts)
        encoded = self.vocab_.encode(texts, self.max_len)

        # Sisihkan sebagian kecil data latih untuk early stopping. Tanpa ini,
        # jaringan menghafal 3.500 dokumen dalam beberapa epoch.
        indices = rng.permutation(len(encoded))
        n_val = max(1, int(len(indices) * self.val_fraction))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(encoded[train_idx]), torch.from_numpy(y[train_idx])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        x_val = torch.from_numpy(encoded[val_idx])
        y_val = torch.from_numpy(y[val_idx])

        self.network_ = self._build_network(self.n_labels_)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # BCE per label = "K-binary transformation": tiap genre diperlakukan
        # sebagai keputusan biner independen (Hoang 2018; Liu et al. 2017),
        # dengan bobot positif untuk mengimbangi ketimpangan antar genre.
        criterion = self._build_criterion(y[train_idx])
        # Validasi dipantau dengan BCE tanpa bobot: yang ingin diketahui adalah
        # kualitas prediksi apa adanya, bukan loss yang sudah dimiringkan.
        val_criterion = nn.BCEWithLogitsLoss()

        best_loss, best_state, waited = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.network_.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.network_(batch_x), batch_y)
                loss.backward()
                # Gradien BiLSTM mudah meledak pada urutan panjang.
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 5.0)
                optimizer.step()

            self.network_.eval()
            with torch.no_grad():
                val_loss = float(val_criterion(self.network_(x_val), y_val))

            if val_loss < best_loss - 1e-4:
                best_loss, waited = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.network_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    logger.debug("      early stopping pada epoch %d (val loss %.4f)", epoch + 1, best_loss)
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.eval()
        return self

    # -- Inferensi --------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.vocab_.encode(self._to_text(X), self.max_len)
        outputs = []
        self.network_.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), 256):
                batch = torch.from_numpy(encoded[start : start + 256])
                outputs.append(torch.sigmoid(self.network_(batch)).numpy())
        return np.vstack(outputs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def __sklearn_tags__(self):
        # scikit-learn >=1.6 memakai mekanisme tags baru; ditandai multi-label
        # agar cross_val_predict tidak menolak target 2-dimensi.
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags

    def _more_tags(self) -> dict:
        # Jalur kompatibilitas untuk scikit-learn < 1.6.
        return {"multilabel": True, "multioutput": True}
