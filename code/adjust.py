import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from metrics import full_report, save_json

TAUS = np.round(np.arange(0.0, 2.01, 0.05), 2)


def adjust(p, prior, tau):
    s = np.log(np.clip(p, 1e-8, 1)) - tau * np.log(prior)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def process(run_dir):
    vf, tf = os.path.join(run_dir, "val_predictions.csv"), os.path.join(run_dir, "test_predictions.csv")
    if not (os.path.exists(vf) and os.path.exists(tf)):
        return None
    v, t = pd.read_csv(vf), pd.read_csv(tf)
    pcols = [c for c in v.columns if c.startswith("p_")]
    classes = [c[2:] for c in pcols]
    pv, yv = v[pcols].values, v["y"].values
    prior = np.bincount(yv, minlength=len(classes)).astype(float) + 1.0
    prior /= prior.sum()

    scores = [balanced_accuracy_score(yv, adjust(pv, prior, tau).argmax(1)) for tau in TAUS]
    best = float(TAUS[int(np.argmax(scores))])

    pt = adjust(t[pcols].values, prior, best)
    t_adj = t.copy()
    t_adj[pcols] = pt
    t_adj.to_csv(os.path.join(run_dir, "test_predictions_adj.csv"), index=False)
    m = full_report(t["y"].values, pt, classes)
    m.update(tau=best, val_bacc_before=float(scores[0]), val_bacc_after=float(max(scores)))
    save_json(m, os.path.join(run_dir, "test_metrics_adj.json"))
    return best, scores[0], max(scores), m["balanced_accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2")
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        r = process(d)
        if r:
            print(f"{os.path.basename(d):40s} tau={r[0]:.2f}  val bacc {r[1]:.4f} -> {r[2]:.4f}   test bacc(adj) {r[3]:.4f}")


if __name__ == "__main__":
    main()
