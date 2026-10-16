"""Cek ruang label aktual yang dipakai model + kebocoran duplikat-dekat."""
import json, joblib, numpy as np, pandas as pd
A = '/opt/project/Model/artifacts/'
mlb = joblib.load(A + 'genre_binarizer.joblib')
print('label dipakai model (%d):' % len(mlb.classes_))
print('  ', list(mlb.classes_))

fact = pd.read_csv('/opt/project/Dataset/data_novel.csv')
from collections import Counter
c = Counter()
for s in fact.genres.fillna(''):
    c.update(g.strip() for g in s.split(',') if g.strip())
print('\nseluruh 33 genre + frekuensi:')
for g, n in c.most_common():
    used = 'DIPAKAI' if g in set(mlb.classes_) else '  dibuang'
    print('   %-16s %5d  %s' % (g, n, used))

# duplikat-dekat: judul yang sangat mirip (potensi kebocoran train/test)
print('\n--- cek duplikat-dekat judul ---')
import re
def norm(t):
    t = str(t).lower()
    t = re.sub(r'\(.*?\)', ' ', t)
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    t = re.sub(r'\b(the|a|an|of|to|i|ii|iii|novel|wn|ln|web|light)\b', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
fact['k'] = fact.title.map(norm)
d = fact[fact.k.duplicated(keep=False) & (fact.k != '')].sort_values('k')
print('judul ternormalisasi bertabrakan:', d.k.nunique(), 'kelompok /', len(d), 'baris')
for k, grp in list(d.groupby('k'))[:8]:
    print('   *', k, '->', list(grp.title)[:3])

emb = np.load(A + 'embeddings.npy')
cat = pd.read_parquet(A + 'catalog.parquet')
print('\nembeddings', emb.shape, '| catalog', cat.shape, '| kolom:', list(cat.columns)[:14])
X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
S = X @ X.T
np.fill_diagonal(S, -1)
mx = S.max(1)
for th in (0.98, 0.95, 0.90):
    print('pasangan konten sangat mirip  cos>%.2f : %d baris' % (th, (mx > th).sum()))
idx = np.argsort(-mx)[:6]
tcol = 'title' if 'title' in cat.columns else cat.columns[0]
for i in idx:
    j = int(S[i].argmax())
    print('   %.3f  %-42s || %s' % (mx[i], str(cat[tcol].iloc[i])[:42], str(cat[tcol].iloc[j])[:42]))
