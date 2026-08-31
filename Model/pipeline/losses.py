"""
Fungsi loss untuk multi-label
=============================

**Asymmetric Loss (ASL)** — Ridnik et al., ICCV 2021,
"Asymmetric Loss For Multi-Label Classification".

Masalah yang diselesaikan: pada multi-label, tiap dokumen hanya membawa
sedikit label positif (di sini rata-rata 4,2 dari 24) dan sangat banyak label
negatif. Ketimpangan positif–negatif itu mendominasi gradien, sehingga model
belajar dari negatif-negatif mudah dan mengabaikan positif yang langka.

ASL menanganinya dengan memperlakukan positif dan negatif secara berbeda:

  * `gamma_neg > gamma_pos` — negatif yang sudah mudah ditebak diturunkan
    bobotnya secara dinamis, sementara positif tetap berbobot penuh;
  * `clip` — probabilitas negatif digeser turun sebesar margin, sehingga
    negatif dengan keyakinan sangat rendah dibuang sepenuhnya dari gradien.
    Ini juga meredam pengaruh label yang salah anotasi — relevan di sini,
    karena analisis konsistensi menunjukkan label genre memang berderau.

Nilai default `gamma_neg=4, gamma_pos=0, clip=0.05` diambil dari makalahnya.
"""

from __future__ import annotations

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Implementasi ASL untuk keluaran logit multi-label."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        # Pergeseran probabilitas: negatif yang keyakinannya di bawah `clip`
        # dianggap sudah selesai dan tidak lagi memberi gradien.
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Fokus asimetris: eksponen berbeda untuk positif dan negatif.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                weight = torch.pow(
                    1 - probs_pos * targets - probs_neg * (1 - targets),
                    self.gamma_pos * targets + self.gamma_neg * (1 - targets),
                )
            loss = loss * weight

        return -loss.sum(dim=1).mean()
