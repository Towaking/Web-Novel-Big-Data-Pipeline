# -*- coding: utf-8 -*-
"""
Eksperimen 3 — fine-tuning transformer pralatih untuk multi-label
=================================================================
Model card menyimpulkan "deep learning bukan alat yang tepat untuk tugas ini".
Kesimpulan itu diambil dari BiLSTM/TextCNN yang embedding-nya **dilatih dari
nol**, dengan alasan eksplisit: lingkungan dianggap offline sehingga embedding
pralatih tidak tersedia.

Premis itu tidak lagi berlaku. Uji ini menjawab pertanyaan yang sebenarnya:
apakah kekalahan DL disebabkan arsitekturnya, atau semata karena tidak ada
representasi pralatih? Backbone-nya BERT kecil (33 juta parameter) yang
di-fine-tune penuh dengan kepala sigmoid 24 label.

Protokolnya dijaga sama: pembagian dev/test identik, ambang disetel di atas
probabilitas out-of-fold, set uji disentuh sekali di akhir.
"""
from __future__ import annotations
import sys, os, time, argparse, json
sys.path.insert(0, "/opt/project")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from Model.pipeline import evaluate as ev
from Model.pipeline.config import CV_FOLDS, MIN_GENRE_SUPPORT, RANDOM_STATE
from Model.pipeline.preprocessing import build_label_matrix
from Model.pipeline.train import split_indices

ART = "/opt/project/Model/artifacts/"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--maxlen", type=int, default=256)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--folds", type=int, default=CV_FOLDS)
a = ap.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(RANDOM_STATE)

# Data
df = pd.read_parquet(ART + "_research_frame.parquet")
Y, LABELS, _ = build_label_matrix(df["genre_list"], MIN_GENRE_SUPPORT)


def text_of(r):
    """Teks natural — stopword utuh, tanpa stemming (kebalikan jalur TF-IDF)."""
    return f"{r.title}. {str(r.synopsis_raw or '').strip()} Tags: {str(r.tags or '').replace(',', ', ')}"


TEXTS = [text_of(r) for r in df.itertuples()]
dev, test = split_indices(len(df))
print("n=%d | label=%d | dev=%d test=%d | %s" % (len(df), len(LABELS), len(dev), len(test), a.model))

from transformers import AutoModel, AutoTokenizer

TOK = AutoTokenizer.from_pretrained(a.model)


def encode_batch(idx):
    enc = TOK([TEXTS[i] for i in idx], truncation=True, max_length=a.maxlen,
              padding=True, return_tensors="pt")
    return enc


class Tagger(nn.Module):
    """Backbone pralatih + kepala linear 24 label pada representasi [CLS]."""

    def __init__(self, name, n_labels):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(h, n_labels)

    def forward(self, **enc):
        # [CLS] dipakai karena BGE memang dilatih dengan pooling itu.
        cls = self.enc(**enc).last_hidden_state[:, 0]
        return self.out(self.drop(cls))


def train_fold(tr_idx, va_idx, y_tr):
    model = Tagger(a.model, len(LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(tr_idx) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    # pos_weight: tanpa ini genre langka nyaris tak memberi gradien —
    # perlakuan yang sama seperti pada BiLSTM di pipeline produksi.
    pos = y_tr.sum(0).clip(1)
    pw = torch.tensor(((len(y_tr) - pos) / pos).clip(1, 12), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    yt = torch.tensor(y_tr, dtype=torch.float32)
    model.train()
    for ep in range(a.epochs):
        perm = np.random.RandomState(RANDOM_STATE + ep).permutation(len(tr_idx))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(perm), a.batch):
            sel = perm[s:s + a.batch]
            enc = encode_batch([tr_idx[i] for i in sel])
            opt.zero_grad()
            loss = lossf(model(**enc), yt[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss); nb += 1
        print("      epoch %d/%d loss %.4f (%.0fs)" % (ep + 1, a.epochs, tot / nb, time.time() - t0), flush=True)

    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_idx), 64):
            enc = encode_batch(va_idx[s:s + 64])
            out.append(torch.sigmoid(model(**enc)).numpy())
    return model, np.vstack(out)


# OOF pada set pengembangan -> ambang; lalu model akhir -> set uji
t_all = time.time()
oof = np.zeros((len(dev), len(LABELS)), dtype=np.float32)
kf = KFold(n_splits=a.folds, shuffle=True, random_state=RANDOM_STATE)
for k, (tr, va) in enumerate(kf.split(dev), 1):
    print("  fold %d/%d" % (k, a.folds), flush=True)
    _, p = train_fold(dev[tr], dev[va], Y[dev][tr])
    oof[va] = p

th = ev.tune_thresholds(Y[dev], oof, LABELS)
oof_scores = ev.evaluate(Y[dev], oof, th)
print("\nOOF: makro %.4f mikro %.4f P@3 %.4f" %
      (oof_scores["f1_macro"], oof_scores["f1_micro"], oof_scores["precision_at_3"]))

print("\n  model akhir pada seluruh set pengembangan", flush=True)
model, prob_test = train_fold(dev, test, Y[dev])
scores = ev.evaluate(Y[test], prob_test, th)
print("\nUJI : makro %.4f mikro %.4f P@3 %.4f Hamming %.4f  (total %.0f menit)"
      % (scores["f1_macro"], scores["f1_micro"], scores["precision_at_3"],
         scores["hamming_loss"], (time.time() - t_all) / 60))

np.save(ART + "_ft_prob_test.npy", prob_test)
np.save(ART + "_ft_oof.npy", oof)
np.save(ART + "_ft_th.npy", th)
json.dump({"model": a.model, "epochs": a.epochs, "maxlen": a.maxlen, "lr": a.lr,
           "oof_f1_macro": oof_scores["f1_macro"], **scores,
           "f1_per_label": ev.per_label_scores(Y[test], prob_test, th, LABELS)},
          open(ART + "_exp3_finetune.json", "w"), indent=1)
print("tersimpan: _exp3_finetune.json")
