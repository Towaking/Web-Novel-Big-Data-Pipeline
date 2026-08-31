"""Dump frame training Fase 4 + sinopsis MENTAH ke parquet lokal.

Transformer butuh teks natural (stopword utuh), sedangkan `corpus_clean` dari
Fase 2 sudah dibuang stopword-nya. Jadi sinopsis mentah diambil langsung dari
CSV sumber lewat `url`, dedup dengan aturan yang sama seperti Fase 2
(sinopsis terpanjang menang).
"""
import sys
sys.path.insert(0, '/opt/project')
import pandas as pd
from src.phases.phase4_model import load_training_frame

pdf, stopwords = load_training_frame()
print('frame Fase 4:', pdf.shape)

syn = pd.read_csv('/opt/project/Dataset/novel_dataset_scrapped.csv')
syn['_len'] = syn['summary'].fillna('').str.split().str.len()
syn = (syn.sort_values('_len', ascending=False)
          .drop_duplicates('url', keep='first')[['url', 'summary']]
          .rename(columns={'summary': 'synopsis_raw'}))

out = pdf.merge(syn, on='url', how='left')
out['synopsis_raw'] = out['synopsis_raw'].fillna('')
print('setelah join sinopsis mentah:', out.shape,
      '| kosong:', (out.synopsis_raw.str.strip() == '').sum())

out['genre_list'] = out['genre_list'].map(lambda g: list(g) if g is not None else [])
out.to_parquet('/opt/project/Model/artifacts/_research_frame.parquet', index=False)
pd.Series(stopwords).to_frame('w').to_parquet('/opt/project/Model/artifacts/_research_stopwords.parquet', index=False)
print('OK ->', out.shape, list(out.columns))
