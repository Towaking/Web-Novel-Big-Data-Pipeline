"""Unduh model HF dengan resume + retry (koneksi container sering putus)."""
import sys, time, argparse
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tries', type=int, default=12)
a = ap.parse_args()

# Hanya berkas yang dibutuhkan: safetensors saja, bukan duplikat .bin/.h5/.onnx
ALLOW = ['*.json', '*.txt', '*.safetensors', '*.model', 'README.md', '1_Pooling/*']
IGNORE = ['*.bin', '*.h5', '*.onnx', '*.ot', '*.msgpack', 'onnx/*', 'openvino/*']

for i in range(1, a.tries + 1):
    try:
        p = snapshot_download(a.model, allow_patterns=ALLOW, ignore_patterns=IGNORE,
                              max_workers=1, etag_timeout=60)
        print('OK ->', p)
        sys.exit(0)
    except Exception as e:
        print('percobaan %d/%d gagal: %s' % (i, a.tries, repr(e)[:130]), flush=True)
        time.sleep(min(3 * i, 20))
print('MENYERAH')
sys.exit(1)
