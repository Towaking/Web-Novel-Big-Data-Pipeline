# -*- coding: utf-8 -*-
"""Eksperimen 2 — matangkan fusi TF-IDF + embedding, lalu simpan OOF baseline."""
import sys, json
sys.path.insert(0, "/opt/project")
sys.path.insert(0, "/opt/project/scripts/research")

import numpy as np
from exp_repr import (ART, load_frame, attach_emb, run, split_indices,
                      build_union, head, out_of_fold_probabilities)
from Model.pipeline import evaluate as ev
from Model.pipeline.config import FeatureConfig
from sklearn.pipeline import Pipeline

df, frame, y, labels = load_frame()
dev, test = split_indices(len(frame))
emb = np.load(ART + "_emb_bge-small-en-v15_full.npy")
ecols = attach_emb(frame, emb, "e")
frame = frame.copy()  # rapikan fragmentasi kolom

res = []

# --- simpan OOF baseline untuk dipakai audit label (bagian 3) --------------
base = Pipeline([("features", build_union(FeatureConfig(), True, None, 1.0)),
                 ("clf", head("svm", 0.3, "balanced"))])
oof = out_of_fold_probabilities(base, frame.iloc[dev], y[dev])
th = ev.tune_thresholds(y[dev], oof, labels)
np.save(ART + "_oof_baseline.npy", oof)
np.save(ART + "_th_baseline.npy", th)
np.save(ART + "_dev_idx.npy", dev)
print("OOF baseline disimpan | makro OOF %.4f\n" % ev.evaluate(y[dev], oof, th)["f1_macro"])

# --- sapuan bobot embedding pada dua kepala klasifikasi -------------------
for kind, C in (("svm", 0.3), ("logreg", 4.0)):
    for w in (0.5, 1.0, 2.0, 3.0, 5.0):
        res.append(run(f"fusi {kind} w={w}", frame, y, labels, dev, test,
                       tfidf=True, emb_cols=ecols, emb_weight=w,
                       kind=kind, C=C, cw="balanced"))

json.dump(res, open(ART + "_exp2.json", "w"), indent=1)
print("\nRINGKASAN (urut makro-F1 OOF — kriteria pemilihan yang sah):")
for r in sorted(res, key=lambda r: -r["oof_f1_macro"]):
    print("  %-22s OOF %.4f | UJI makro %.4f mikro %.4f P@3 %.4f"
          % (r["name"], r["oof_f1_macro"], r["f1_macro"], r["f1_micro"], r["precision_at_3"]))
