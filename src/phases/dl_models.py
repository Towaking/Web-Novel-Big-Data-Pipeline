"""
Kompatibilitas — jaringan deep learning
=======================================
Implementasinya kini di `Model/pipeline/networks.py`. Modul ini dipertahankan
agar artefak model lama, yang merujuk nama modul ini di dalam pickle-nya, tetap
dapat dimuat. Kode baru sebaiknya mengimpor dari `Model.pipeline.networks`.
"""

from Model.pipeline.networks import (
    ARCHITECTURES,
    BiLSTMAttentionNet,
    TorchMultiLabelClassifier,
    Vocabulary,
)

__all__ = ["ARCHITECTURES", "BiLSTMAttentionNet", "TorchMultiLabelClassifier", "Vocabulary"]
