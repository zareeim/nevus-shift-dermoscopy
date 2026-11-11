import json

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)


def expected_calibration_error(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def sens_at_spec(y_bin, score, spec):
    """Sensitivity of the positive class at a fixed specificity (clinical operating point)."""
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= (1 - spec)
    return float(tpr[ok].max()) if ok.any() else 0.0


def macro_auc(y, probs, labels):
    present = [k for k in labels if (y == k).any() and (y != k).any()]
    aucs = {int(k): float(roc_auc_score((y == k).astype(int), probs[:, k])) for k in present}
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


def core_metrics(y, probs, class_names, positive=None):
    y = np.asarray(y)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    m = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=sorted(set(y)), zero_division=0)),
        "ece": expected_calibration_error(probs, y),
    }
    m["macro_auc"], per_cls = macro_auc(y, probs, labels)
    m["auc_per_class"] = {class_names[k]: v for k, v in per_cls.items()}
    m["recall_per_class"] = {
        class_names[k]: float((pred[y == k] == k).mean()) for k in labels if (y == k).any()
    }
    # clinically relevant operating points for the malignant / melanoma class
    pos = positive or ("mel" if "mel" in class_names else ("malignant" if "malignant" in class_names else None))
    if pos is not None and pos in class_names:
        k = class_names.index(pos)
        yb = (y == k).astype(int)
        m[f"{pos}_auc"] = float(roc_auc_score(yb, probs[:, k])) if 0 < yb.sum() < len(yb) else float("nan")
        m[f"{pos}_sens@spec80"] = sens_at_spec(yb, probs[:, k], 0.80)
        m[f"{pos}_sens@spec90"] = sens_at_spec(yb, probs[:, k], 0.90)
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=labels).tolist()
    return m


def bootstrap_ci(y, probs, class_names, n_boot=1000, seed=0,
                 keys=("balanced_accuracy", "macro_f1", "macro_auc")):
    """95% percentile bootstrap CIs (resampling test images)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); probs = np.asarray(probs)
    vals = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], probs[idx]
        pred = pp.argmax(1)
        if "balanced_accuracy" in keys:
            vals["balanced_accuracy"].append(balanced_accuracy_score(yy, pred))
        if "macro_f1" in keys:
            vals["macro_f1"].append(f1_score(yy, pred, average="macro", labels=sorted(set(yy)), zero_division=0))
        if "macro_auc" in keys:
            vals["macro_auc"].append(macro_auc(yy, pp, range(len(class_names)))[0])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def full_report(y, probs, class_names, n_boot=1000):
    m = core_metrics(y, probs, class_names)
    m["ci95"] = bootstrap_ci(y, probs, class_names, n_boot=n_boot)
    return m


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
