"""Dump frame training Fase 4 (zona clean) ke parquet lokal untuk riset."""
import sys
sys.path.insert(0, "/opt/project")
import json
import pandas as pd
from src.phases.phase4_model import load_training_frame

pdf, stopwords = load_training_frame()
pdf = pdf.copy()
for col in ("genre_list",):
    pdf[col] = pdf[col].map(lambda g: list(g) if g is not None else [])
out = "/opt/project/Model/artifacts/_research_frame39k.parquet"
pdf.to_parquet(out, index=False)
with open("/opt/project/Model/artifacts/_research_stopwords39k.json", "w", encoding="utf-8") as fh:
    json.dump(stopwords, fh)
print("OK", pdf.shape)
print(pdf.dtypes.to_string())
print(pdf["metadata_tier"].value_counts().to_string())
