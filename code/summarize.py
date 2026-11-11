import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from metrics import macro_auc

KEYS = ["balanced_accuracy", "macro_f1", "macro_auc", "accuracy", "ece"]


def collect(runs_dir, fname="test_metrics.json"):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", fname))):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"(.*)_f(\d+)$", name)
        exp, fold = (m.group(1), int(m.group(2))) if m else (name, 0)
        d = json.load(open(f))
        row = {"experiment": exp, "fold": fold, **{k: d.get(k, np.nan) for k in KEYS}}
        for k in d:
            if "sens@spec" in k:
                row[k] = d[k]
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_predictions(runs_dir, exp, fname="test_predictions.csv"):
    files = sorted(glob.glob(os.path.join(runs_dir, f"{exp}_f*", fname)))
    if not files:
        files = [os.path.join(runs_dir, exp, fname)]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    key = "image_path"
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    pa = m[[c for c in m.columns if c.startswith("p_") and c.endswith("_a")]].values
    pb = m[[c for c in m.columns if c.startswith("p_") and c.endswith("_b")]].values
    y = m["y_a"].values
    assert (y == m["y_b"].values).all(), "the two experiments disagree on labels of the same images"
    assert len(m) == len(a) == len(b), "the two experiments were not tested on the same images"
    n_cls = pa.shape[1]

    def stats(idx):
        yy = y[idx]
        out = []
        for p in (pa[idx], pb[idx]):
            pr = p.argmax(1)
            out.append([balanced_accuracy_score(yy, pr),
                        f1_score(yy, pr, average="macro", zero_division=0),
                        macro_auc(yy, p, range(n_cls))[0]])
        return np.array(out[0]) - np.array(out[1])

    obs = stats(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = np.array([stats(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    names = ["balanced_accuracy", "macro_f1", "macro_auc"]
    res = {}
    for j, nme in enumerate(names):
        lo, hi = np.nanpercentile(boots[:, j], [2.5, 97.5])
        p = 2 * min((boots[:, j] <= 0).mean(), (boots[:, j] >= 0).mean())
        res[nme] = {"delta": float(obs[j]), "ci95": [float(lo), float(hi)], "p_two_sided": float(min(1.0, p))}
    return len(y), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"))
    ap.add_argument("--file", default="test_metrics.json",
                    help="use val_metrics.json to compare tuning runs on validation only")
    ap.add_argument("--pred_file", default="test_predictions.csv",
                    help="predictions used for the paired test (e.g. test_predictions_adj.csv)")
    a = ap.parse_args()

    df = collect(a.runs, a.file)
    if df.empty:
        raise SystemExit(f"no {a.file} found")
    num = [c for c in df.columns if c not in ("experiment", "fold")]
    agg = df.groupby("experiment")[num].agg(["mean", "std"])
    agg[("n_folds", "")] = df.groupby("experiment")["fold"].nunique()
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string())
    if a.out:
        agg.to_csv(a.out)

    if a.compare:
        A, B = (pooled_predictions(a.runs, e, a.pred_file) for e in a.compare)
        n, res = paired_bootstrap(A, B)
        print(f"\n{a.compare[0]} minus {a.compare[1]}  (paired bootstrap, n={n} pooled test images)")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
