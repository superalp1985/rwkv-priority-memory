"""Source-domain and diagnostic metrics. Never maps proxy scores to project grades."""

import math

import numpy as np

from pilot_data import EMOTIONS

PAIR_MARGIN = 0.05


def admission_metrics(labels, scores):
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(y) or s.shape != y.shape or not np.isfinite(s).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("invalid binary metric inputs")
    pos, n = int(y.sum()), len(y)
    if pos in (0, n):
        raise ValueError("need both binary classes")
    order = np.argsort(-s, kind="stable")
    ordered = s[order]
    ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    ap = np.dot(np.diff(np.r_[0, tp]) / pos, tp / (ends + 1))
    auc = np.dot(np.diff(np.r_[0, fp]), tp + np.r_[0, tp[:-1]]) / (2 * pos * (n - pos))
    return dict(n=n, positives=pos, prevalence=pos / n, average_precision=float(ap), auroc=float(auc))


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / den, -1, 1)) if den > 0 else None


def emotion_metrics(labels, scores, *, discrete=None, train_medians=None):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if (s.shape != y.shape or y.ndim != 2 or not len(y) or y.shape[1] != len(EMOTIONS)
            or not np.isin(y, range(4)).all() or not np.isfinite(s).all()):
        raise ValueError("invalid emotion metric inputs")
    if discrete is not None:
        discrete = np.asarray(discrete)
        if discrete.shape != y.shape or not np.isin(discrete, range(4)).all():
            raise ValueError("invalid discrete emotion outputs")
    per = {}
    for j, name in enumerate(EMOTIONS):
        target, predicted = y[:, j], s[:, j]
        positive = target > 0
        error = np.abs(target - predicted)
        per[name] = dict(pearson=correlation(target, predicted), mae=float(error.mean()),
                         positive_mae=float(error[positive].mean()) if positive.any() else None,
                         positives=int(positive.sum()), support=np.bincount(target.astype(int), minlength=4).tolist())
        if discrete is not None:
            per[name]["discrete_pearson"] = correlation(target, discrete[:, j])
        if train_medians is not None:
            per[name]["train_median_mae"] = float(np.abs(target - train_medians[j]).mean())
    # A constant prediction must not improve the average by disappearing from it.
    macro = sum((row["pearson"] or 0.0) for row in per.values()) / len(EMOTIONS)
    result = dict(n=len(y), macro_pearson=macro, undefined_is_zero_in_macro=True, per_emotion=per)
    if discrete is not None:
        result["macro_discrete_pearson"] = sum((row["discrete_pearson"] or 0.0) for row in per.values()) / len(EMOTIONS)
    return result


def pair_result(ident, higher, lower):
    valid = (higher is not None and lower is not None and math.isfinite(higher)
             and math.isfinite(lower) and 0 <= higher <= 1 and 0 <= lower <= 1)
    margin = higher - lower if valid else None
    return {"id": ident, "higher": higher if valid else None, "lower": lower if valid else None,
            "margin": margin, "pass": bool(margin is not None and margin >= PAIR_MARGIN)}


def contrast_uses_context(higher_contextual, lower_contextual, higher_target_only, lower_target_only):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (higher_contextual, lower_contextual, higher_target_only, lower_target_only)]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("context contrast vectors must be same-shaped and finite")
    return not np.allclose(arrays[0] - arrays[2], arrays[1] - arrays[3], atol=1e-7, rtol=0)