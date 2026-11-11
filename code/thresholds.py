import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from adjust import TAUS, adjust
from metrics import full_report, save_json

GRID = np.round(np.arange(-2.0, 2.01, 0.1), 2)


def apply_bias(p, b):
    s = np.log(np.clip(p, 1e-8, 1)) + b
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, p, labels):
    return f1_score(y, p.argmax(1), average="macro", labels=labels, zero_division=0)


def fit_f1_bias(p, y, n_pass=3):
    K = p.shape[1]; labels = list(range(K)); b = np.zeros(K)
    best = macro_f1(y, apply_bias(p, b), labels)
    for _ in range(n_pass):
        for k in range(K):
            for g in GRID:
                bb = b.copy(); bb[k] = g
                s = macro_f1(y, apply_bias(p, bb), labels)
                if s > best + 1e-9:
                    best, b = s, bb
    return b, best


def process(d, objective):
    v, t = pd.read_csv(os.path.join(d, "val_predictions.csv")), pd.read_csv(os.path.join(d, "test_predictions.csv"))
    pc = [c for c in v.columns if c.startswith("p_")]; classes = [c[2:] for c in pc]
    pv, yv, pt = v[pc].values, v["y"].values, t[pc].values
    if objective == "bacc":
        prior = np.bincount(yv, minlength=len(classes)).astype(float) + 1.0; prior /= prior.sum()
        sc = [balanced_accuracy_score(yv, adjust(pv, prior, tau).argmax(1)) for tau in TAUS]
        tau = float(TAUS[int(np.argmax(sc))]); padj = adjust(pt, prior, tau); info = {"tau": tau}
    else:
        b, _ = fit_f1_bias(pv, yv); padj = apply_bias(pt, b)
        info = {"bias": {c: float(x) for c, x in zip(classes, b)}}
    out = t.copy(); out[pc] = padj
    out.to_csv(os.path.join(d, f"test_predictions_{objective}.csv"), index=False)
    m = full_report(t["y"].values, padj, classes); m.update(info)
    save_json(m, os.path.join(d, f"test_metrics_{objective}.json"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--objective", choices=["bacc", "f1"], required=True)
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if os.path.exists(os.path.join(d, "val_predictions.csv")) and os.path.exists(os.path.join(d, "test_predictions.csv")):
            m = process(d, a.objective)
            print(f"{os.path.basename(d):32s} [{a.objective}] test bacc {m['balanced_accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
