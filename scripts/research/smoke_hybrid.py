# -*- coding: utf-8 -*-
"""Uji cepat arsitektur `hybrid` lewat jalur produksi (bukan harness riset).

Tujuannya bukan mengukur akurasi, melainkan memastikan rangkaiannya nyambung:
build_feature_frame -> build_feature_union (dengan blok embedding) ->
OneVsRest -> predict_proba, lalu artefaknya bisa di-pickle dan dipakai ulang
oleh jalur inferensi dengan bentuk masukan yang dipakai dashboard.
"""
import sys, time, io
sys.path.insert(0, "/opt/project")

import joblib
import numpy as np
import pandas as pd

from Model.pipeline import models as mdl
from Model.pipeline.config import MIN_GENRE_SUPPORT
from Model.pipeline.preprocessing import FEATURE_COLUMNS, build_feature_frame, build_label_matrix

df = pd.read_parquet("/opt/project/Model/artifacts/_research_frame.parquet").head(300)
frame = build_feature_frame(df)
y, labels, _ = build_label_matrix(df["genre_list"], min_support=5)
print("kolom fitur :", list(frame.columns))
print("FEATURE_COLUMNS cocok:", tuple(frame.columns) == FEATURE_COLUMNS)
print("contoh text_natural :", frame["text_natural"].iloc[0][:100], "...")

t0 = time.time()
model = mdl.build_model("hybrid", **mdl.PARAM_GRID["hybrid"][0])
model.fit(frame, y)
print("fit %d baris: %.0fs" % (len(frame), time.time() - t0))

blocks = model.named_steps["features"].transformers_
print("blok fitur aktif:", [b[0] for b in blocks])
print("dimensi gabungan:", model.named_steps["features"].transform(frame.head(2)).shape)

t0 = time.time()
p = mdl.predict_proba_frame(model, frame.head(4))
print("predict_proba 4 baris: %.2fs -> %s" % (time.time() - t0, p.shape))

# --- artefak harus bertahan lewat pickle, dan TIDAK boleh membawa bobot encoder ---
buf = io.BytesIO()
joblib.dump({"pipeline": model, "thresholds": np.full(len(labels), 0.4)}, buf)
size_mb = buf.tell() / 1e6
buf.seek(0)
loaded = joblib.load(buf)["pipeline"]
print("ukuran artefak: %.1f MB (bobot encoder TIDAK ikut ter-pickle)" % size_mb)

# --- bentuk masukan persis seperti yang dikirim dashboard ---
infer = pd.DataFrame([{
    "corpus_clean": "sword cultivation revenge tower",
    "synopsis_clean": "sword cultivation revenge tower",
    "tags_clean": "time travel cultivation revenge",
    "type_origin": "CN",
    "type": "Web Novel",
    "text_natural": "A swordsman returns ten years younger to master forbidden techniques. "
                    "Tags: Time Travel, Cultivation, Revenge",
}])
t0 = time.time()
prob = mdl.predict_proba_frame(loaded, infer)
print("inferensi 1 teks setelah unpickle: %.0f ms -> %s" % (1000 * (time.time() - t0), prob.shape))
top = np.argsort(-prob.ravel())[:5]
print("genre teratas:", [(labels[i], round(float(prob.ravel()[i]), 3)) for i in top])
print("\nSMOKE TEST LULUS")
