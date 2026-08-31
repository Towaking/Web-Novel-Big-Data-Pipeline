"""
Unduh encoder kalimat pralatih ke dalam image (dipanggil dari Dockerfile).

Dipisah jadi berkas sendiri, bukan `python -c` sebaris, karena butuh retry:
registry Hugging Face memutus koneksi di tengah berkas jauh lebih sering
daripada yang diduga, dan snapshot yang terpotong baru ketahuan saat inferensi
pertama — kegagalan yang mahal untuk didiagnosis di runtime.

Hanya bobot `safetensors` yang diambil; salinan `.bin`/`.onnx` di repo yang
sama isinya identik dan hanya menggandakan ukuran image.
"""

import os
import sys
import time

from huggingface_hub import snapshot_download

MODEL = os.environ.get("SENTENCE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*"]
IGNORE = ["*.bin", "*.h5", "*.onnx", "*.ot", "*.msgpack", "onnx/*", "openvino/*"]

for attempt in range(1, 9):
    try:
        path = snapshot_download(
            MODEL, allow_patterns=ALLOW, ignore_patterns=IGNORE,
            max_workers=1, etag_timeout=60,
        )
        print(f"[hf] OK: {MODEL} -> {path}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - semua kegagalan jaringan layak diulang
        print(f"[hf] percobaan {attempt}/8 gagal: {exc!r}", flush=True)
        time.sleep(min(4 * attempt, 25))

sys.exit(f"[hf] GAGAL permanen mengunduh {MODEL}")
