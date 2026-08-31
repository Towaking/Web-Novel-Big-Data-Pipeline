#!/bin/sh
set -e
cd /opt/project
for M in BAAI/bge-base-en-v1.5 intfloat/e5-base-v2; do
  echo "=== fetch $M ==="
  python scripts/research/fetch_model.py --model "$M" || exit 1
done
export HF_HUB_OFFLINE=1 ENC_THREADS=6
echo "=== encode bge-base ==="
python scripts/research/encode.py --model BAAI/bge-base-en-v1.5 --maxlen 384 --batch 16
echo "=== encode e5-base ==="
python scripts/research/encode.py --model intfloat/e5-base-v2 --maxlen 384 --batch 16
echo "=== SELESAI ==="
