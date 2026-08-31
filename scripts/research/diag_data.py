"""Diagnostik dataset: kualitas label genre lintas dua sumber."""
import pandas as pd, numpy as np, re
from collections import Counter

P = '/opt/project/Dataset/'
fact = pd.read_csv(P + 'data_novel.csv')
syn  = pd.read_csv(P + 'novel_dataset_scrapped.csv')
print('fact', fact.shape, '| syn', syn.shape)

def gset(s):
    if not isinstance(s, str) or s.strip() in ('', 'Unknown'):
        return frozenset()
    return frozenset(g.strip() for g in s.split(',') if g.strip())

# --- dedup file sinopsis: sinopsis terpanjang menang (aturan data_contract) ---
syn['_len'] = syn['summary'].fillna('').str.split().str.len()
syn_d = syn.sort_values('_len', ascending=False).drop_duplicates('url', keep='first')
print('syn unik url:', syn_d.shape[0], '| baris dibuang:', len(syn) - len(syn_d))

# apakah semua url fact ada di syn, dan sebaliknya?
uf, us = set(fact.url), set(syn_d.url)
print('url fact only :', len(uf - us))
print('url syn  only :', len(us - uf), '  <-- novel yg punya sinopsis tapi TIDAK ada di tabel fakta')

m = fact.merge(syn_d[['url','summary','genres','title']], on='url', how='left',
               suffixes=('', '_syn'))
m['G_fact'] = m['genres'].map(gset)
m['G_syn']  = m['genres_syn'].map(gset)

both = m[(m.G_fact.str.len() > 0) & (m.G_syn.str.len() > 0)]
print('\n--- perbandingan himpunan genre (n=%d) ---' % len(both))
same  = (both.G_fact == both.G_syn).sum()
print('identik            :', same, '(%.1f%%)' % (100*same/len(both)))
diff  = both[both.G_fact != both.G_syn]
print('berbeda            :', len(diff))
sub   = diff[diff.apply(lambda r: r.G_fact < r.G_syn, axis=1)]
sup   = diff[diff.apply(lambda r: r.G_fact > r.G_syn, axis=1)]
xor   = diff[diff.apply(lambda r: not (r.G_fact <= r.G_syn or r.G_syn <= r.G_fact), axis=1)]
print('  fact SUBSET syn  :', len(sub), '  <-- syn punya genre TAMBAHAN yg hilang di fact')
print('  fact SUPERSET syn:', len(sup))
print('  saling silang    :', len(xor))

added = Counter()
for _, r in sub.iterrows():
    added.update(r.G_syn - r.G_fact)
for _, r in xor.iterrows():
    added.update(r.G_syn - r.G_fact)
print('\ngenre yg PALING SERING hilang di tabel fakta (top 15):')
for g, c in added.most_common(15):
    print('   %-18s +%d' % (g, c))

drop = Counter()
for _, r in sup.iterrows():
    drop.update(r.G_fact - r.G_syn)
for _, r in xor.iterrows():
    drop.update(r.G_fact - r.G_syn)
print('\ngenre yg ada di fact tapi tidak di syn (top 10):')
for g, c in drop.most_common(10):
    print('   %-18s -%d' % (g, c))

# --- kardinalitas label ---
print('\n--- kardinalitas label ---')
print('rata2 genre / novel  fact: %.2f | syn: %.2f' %
      (both.G_fact.str.len().mean(), both.G_syn.str.len().mean()))
union = both.apply(lambda r: r.G_fact | r.G_syn, axis=1)
print('rata2 genre / novel  UNION: %.2f' % union.str.len().mean())

# --- distribusi genre keseluruhan ---
allg = Counter()
for s in m.G_fact: allg.update(s)
print('\ntotal genre unik:', len(allg), '| genre >=50 novel:',
      sum(1 for _, c in allg.items() if c >= 50))

# --- sinopsis ---
m['syn_words'] = m['summary'].fillna('').str.split().str.len()
print('\n--- sinopsis ---')
print('kosong        :', (m.syn_words == 0).sum())
print('<10 kata      :', ((m.syn_words > 0) & (m.syn_words < 10)).sum())
print('median kata   : %.0f | mean %.0f | p90 %.0f | max %d' %
      (m.syn_words.median(), m.syn_words.mean(),
       m.syn_words.quantile(.9), m.syn_words.max()))

# boilerplate / MTL noise
pat = re.compile(r'(machine translat|mtl|raw link|please support|discord|patreon|'
                 r'read more at|translator.s note|copyright|all rights reserved)', re.I)
hit = m['summary'].fillna('').str.contains(pat)
print('sinopsis mengandung boilerplate/promo :', hit.sum())

# non-latin / bahasa asing
def nonlatin_ratio(s):
    s = str(s)
    if not s: return 0.0
    return sum(1 for ch in s if ord(ch) > 0x2E80) / max(len(s), 1)
nl = m['summary'].fillna('').map(nonlatin_ratio)
print('sinopsis >30%% karakter non-latin      : %d' % (nl > .3).sum())

# --- tags ---
m['n_tags'] = m['tags'].fillna('').map(lambda s: len(gset(s)))
print('\n--- tags ---')
print('tags Unknown/kosong :', (m.n_tags == 0).sum())
print('median tag/novel    : %.0f | mean %.1f | max %d' %
      (m.n_tags.median(), m.n_tags.mean(), m.n_tags.max()))
tagc = Counter()
for s in m.tags.fillna(''): tagc.update(gset(s))
print('tag unik            :', len(tagc), '| tag >=50 novel:',
      sum(1 for _, c in tagc.items() if c >= 50))
