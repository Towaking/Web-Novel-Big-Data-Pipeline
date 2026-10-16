"""Statistik uji berpasangan dari paired_results.jsonl."""
import json, os, sys
import numpy as np
from scipy import stats

rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "results", "paired_results.jsonl"), encoding="utf-8")]
by = {}
for r in rows:
    by.setdefault(r["config"], {})[r["seed"]] = r

N1, N2 = 31_728, 7_932  # ukuran latih / uji tiap pembagian


def compare(new, old, metric):
    seeds = sorted(set(by[new]) & set(by[old]))
    a = np.array([by[new][s][metric] for s in seeds])
    b = np.array([by[old][s][metric] for s in seeds])
    d = a - b
    k = len(d)
    sd = d.std(ddof=1) if k > 1 else float("nan")
    t = d.mean() / (sd / np.sqrt(k)) if k > 1 else float("nan")
    t_nb = d.mean() / np.sqrt((1 / k + N2 / N1) * sd ** 2) if k > 1 else float("nan")
    p_nb = 2 * stats.t.sf(abs(t_nb), k - 1) if k > 1 else float("nan")
    return {"metric": metric, "pair": f"{new} vs {old}", "k": k,
            "old": f"{b.mean():.4f} ± {b.std(ddof=1):.4f}", "new": f"{a.mean():.4f} ± {a.std(ddof=1):.4f}",
            "diff": f"{d.mean():+.4f} ± {sd:.4f}", "wins": f"{int((d > 0).sum())}/{k}",
            "t": round(float(t), 2), "t_nadeau_bengio": round(float(t_nb), 2), "p_nb": round(float(p_nb), 4)}


for new, old in [("svm_new", "svm_old"), ("lr_new", "lr_old"), ("ens_new", "svm_new"), ("ens_new", "svm_old"),
                 ("rf_new", "svm_new"), ("svm_new75", "svm_old"), ("lr_new75", "lr_old"),
                 ("ens_new75", "svm_new75"), ("ens_new75", "svm_old"), ("svm_new75", "svm_new"),
                 ("rf_new75", "svm_new75"), ("ens_logit75", "svm_new75"), ("ens_logit75", "ens_prob75"),
                 ("ens_logit75", "svm_old")]:
    if new in by and old in by:
        for metric in ("oof_macro", "test_macro", "test_micro", "test_p3"):
            print(json.dumps(compare(new, old, metric), ensure_ascii=False))
