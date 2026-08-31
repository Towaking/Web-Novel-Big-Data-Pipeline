"""Eksperimen 1 — apakah embedding pralatih menembus batas TF-IDF?"""
import sys, json
sys.path.insert(0, '/opt/project')
sys.path.insert(0, '/opt/project/scripts/research')
import numpy as np
from exp_repr import ART, load_frame, attach_emb, run, split_indices

df, frame, y, labels = load_frame()
dev, test = split_indices(len(frame))
print('n=%d | label=%d | dev=%d test=%d\n' % (len(frame), len(labels), len(dev), len(test)))

emb = np.load(ART + '_emb_bge-small-en-v15_full.npy')
ecols = attach_emb(frame, emb, 'e')
print('embedding bge-small:', emb.shape, '\n')

res = []
# 1. Reproduksi baseline model card (harus ~0,532 / 0,538)
res.append(run('A. TF-IDF baseline (logreg C=4 bal)', frame, y, labels, dev, test,
               tfidf=True, kind='logreg', C=4.0, cw='balanced'))
res.append(run('B. TF-IDF baseline (svm C=0.3 bal)', frame, y, labels, dev, test,
               tfidf=True, kind='svm', C=0.3, cw='balanced'))
# 2. Embedding pralatih SAJA
for C in (1.0, 4.0, 16.0):
    res.append(run(f'C. bge-small SAJA (logreg C={C} bal)', frame, y, labels, dev, test,
                   tfidf=False, emb_cols=ecols, kind='logreg', C=C, cw='balanced'))
# 3. Fusi TF-IDF + embedding
for w in (1.0, 3.0):
    res.append(run(f'D. TF-IDF + bge-small (w={w}, logreg C=4)', frame, y, labels, dev, test,
                   tfidf=True, emb_cols=ecols, emb_weight=w, kind='logreg', C=4.0, cw='balanced'))

json.dump(res, open(ART + '_exp1.json', 'w'), indent=1)
print('\nRINGKASAN (urut makro-F1 uji):')
for r in sorted(res, key=lambda r: -r['f1_macro']):
    print('  %-44s makro %.4f  mikro %.4f  P@3 %.4f' %
          (r['name'], r['f1_macro'], r['f1_micro'], r['precision_at_3']))
