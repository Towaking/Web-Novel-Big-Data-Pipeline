"""Encode korpus dengan model kalimat pralatih -> cache .npy."""
import sys, os, time, argparse, hashlib
sys.path.insert(0, '/opt/project')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np, pandas as pd, torch

torch.set_num_threads(int(os.environ.get('ENC_THREADS', '4')))

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='BAAI/bge-small-en-v1.5')
ap.add_argument('--maxlen', type=int, default=384)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--view', default='full', choices=['full', 'synopsis', 'tags'])
a = ap.parse_args()

df = pd.read_parquet('/opt/project/Model/artifacts/_research_frame.parquet')

def build_text(r):
    """Teks natural untuk transformer: judul + sinopsis mentah + tag.

    Berbeda dari jalur TF-IDF, di sini stopword TIDAK dibuang dan tidak ada
    stemming — model pralatih justru mengandalkan sintaks utuh.
    """
    title = str(r.title or '')
    syn   = str(r.synopsis_raw or '').strip()
    tags  = str(r.tags or '').replace(',', ', ')
    if a.view == 'synopsis':
        return f'{title}. {syn}'
    if a.view == 'tags':
        return f'{title}. Tags: {tags}'
    return f'{title}. {syn} Tags: {tags}'

texts = [build_text(r) for r in df.itertuples()]
print('n=%d | contoh: %s...' % (len(texts), texts[0][:110]))

from sentence_transformers import SentenceTransformer
t0 = time.time()
m = SentenceTransformer(a.model, device='cpu')
m.max_seq_length = a.maxlen
print('model dimuat %.1fs | dim=%d | maxlen=%d' % (time.time()-t0, m.get_sentence_embedding_dimension(), m.max_seq_length))

t0 = time.time()
emb = m.encode(texts, batch_size=a.batch, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False)
print('encode %.1fs -> %s' % (time.time()-t0, emb.shape))

slug = a.model.split('/')[-1].replace('.', '') + '_' + a.view
out = f'/opt/project/Model/artifacts/_emb_{slug}.npy'
np.save(out, emb.astype(np.float32))
print('tersimpan:', out)
