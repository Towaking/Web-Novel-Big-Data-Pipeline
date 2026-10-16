# Eksperimen peningkatan model (2026-10-15)

Skrip yang menghasilkan semua angka di [Doc/peningkatan_model.md](../../../Doc/peningkatan_model.md).
Log hasil aslinya ada di `results/`.

## Menjalankan

1. Salin frame latih Fase 4 dari lake ke `Model/artifacts/_research_frame39k.parquet`
   (berkas `_*` di folder itu di-gitignore):

   ```bash
   docker compose exec -T pipeline python - < scripts/research/peningkatan_model/dump_frame.py
   ```

2. Jalankan skrip dari folder ini di env lokal dengan Python 3.11, scikit-learn 1.7, nltk,
   pyarrow, dan optuna (hanya `run_optuna.py` dan `run_headd3.py`). Cache matriks TF-IDF per fold
   ditulis ke `Model/artifacts/_research_cache/`; `N_JOBS` mengatur jumlah worker.

## Isi

| Skrip | Bagian dokumen | Isi |
|---|---|---|
| `run_baseline.py` | 1 | reproduksi SVM/LogReg lama di frame yang sama |
| `run_prep.py`, `run_serve_skew.py` | 2.1 | pemulihan sensor, dan uji input tanpa sensor |
| `run_batch1.py` (+ `run_batch1_specs.py`) | 2.2 | varian representasi |
| `run_batch2.py`, `run_rf.py`, `rf_size.py` | 2.3 | classifier, Random Forest, ukuran hutan |
| `run_thr.py` | 2.4 | strategi ambang dengan ambang ter-cross-validate |
| `run_optuna.py`, `run_final_check.py` | 2.5 | TPE untuk SVM, lalu uji representasi dan LogReg |
| `run_stack.py`, `run_blend.py` | 2.6 | stacking antar genre dan ensemble |
| `run_headd.py`, `run_headd2.py`, `run_headd3.py` | 2.7 | Head D |
| `run_paired.py`, `run_paired2.py`, `paired_stats.py` | 5 | uji berpasangan lima seed |

`exp_lib.py` memuat protokolnya (pembagian, fold OOF, skor); `blocks_lib.py`, `clf_lib.py`,
`thr_lib.py`, `stack_lib.py`, dan `headd_lib.py` adalah komponen yang dipakai skrip di atas.
