"""Strategi ambang multi-label + evaluasi ambang yang di-cross-validate."""
import numpy as np
from sklearn.metrics import f1_score

from Model.pipeline import evaluate as ev


def _best_threshold_exact(yj, pj, low=0.01, high=0.95, min_pos_pred=1):
    """Ambang optimal F1 per label lewat sapuan semua skor unik (Lipton dkk. 2014).

    Ambang diletakkan di tengah dua skor berurutan supaya tidak menempel tepat
    di satu contoh latih.
    """
    order = np.argsort(-pj, kind="mergesort")
    ps = pj[order]
    ys = yj[order].astype(np.int64)
    tp = np.cumsum(ys)
    k = np.arange(1, len(ps) + 1)
    pos = ys.sum()
    f1 = 2 * tp / (k + pos)
    # hanya batas di antara skor berbeda yang sah
    valid = np.r_[ps[1:] < ps[:-1], True]
    valid &= (ps >= low) & (ps <= high)
    if not valid.any():
        return 0.5, 0.0
    idx = np.flatnonzero(valid)
    best = idx[np.argmax(f1[idx])]
    nxt = ps[best + 1] if best + 1 < len(ps) else ps[best] - 1e-3
    return float((ps[best] + nxt) / 2), float(f1[best])


def tune_exact(y, p, labels=None, low=0.01, high=0.95):
    return np.array([_best_threshold_exact(y[:, j], p[:, j], low, high)[0] for j in range(y.shape[1])],
                    dtype=np.float32)


def tune_grid(y, p, labels=None, low=0.05, high=0.80, step=0.025):
    grid = np.arange(low, high + 1e-9, step)
    out = np.empty(y.shape[1], dtype=np.float32)
    for j in range(y.shape[1]):
        pred = p[:, j][:, None] >= grid[None, :]
        tp = (pred & (y[:, j][:, None] == 1)).sum(0)
        fp = (pred & (y[:, j][:, None] == 0)).sum(0)
        fn = (y[:, j] == 1).sum() - tp
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
        out[j] = grid[int(np.argmax(f1))]  # argmax = ambang terkecil bila seri (sama dgn loop asli)
    return out


def apply(p, thr, at_least_one=False):
    pred = (p >= thr).astype(int)
    if at_least_one:
        empty = pred.sum(1) == 0
        if empty.any():
            ratio = p[empty] / np.maximum(thr, 1e-6)
            pred[np.flatnonzero(empty), np.argmax(ratio, axis=1)] = 1
    return pred


def metrics(y, p, thr, at_least_one=False):
    pred = apply(p, thr, at_least_one)
    return {
        "macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "micro": round(float(f1_score(y, pred, average="micro", zero_division=0)), 4),
        "samples": round(float(f1_score(y, pred, average="samples", zero_division=0)), 4),
    }


def cv_threshold_score(y_dev, p_oof, fold_ids, tuner, at_least_one=False):
    """Skor ambang yang jujur: setel di 2 fold OOF, nilai di fold ke-3."""
    pred = np.zeros_like(y_dev)
    for k in np.unique(fold_ids):
        tr, va = fold_ids != k, fold_ids == k
        thr = tuner(y_dev[tr], p_oof[tr])
        pred[va] = apply(p_oof[va], thr, at_least_one)
    return {
        "macro": round(float(f1_score(y_dev, pred, average="macro", zero_division=0)), 4),
        "micro": round(float(f1_score(y_dev, pred, average="micro", zero_division=0)), 4),
        "samples": round(float(f1_score(y_dev, pred, average="samples", zero_division=0)), 4),
    }
