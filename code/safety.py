import argparse
import os

import numpy as np
import pandas as pd

from label_shift import reweight, source_proxy


def load(a):
    P_raw, P_s = 0, 0
    for k in a.folds:
        V = pd.read_csv(os.path.join(a.runs, f"{a.exp}_f{k}", "val_predictions.csv"))
        O = pd.read_csv(os.path.join(a.ood, f"{a.exp}_f{k}", "ood_predictions.csv")).sort_values("image_path").reset_index(drop=True)
        pc = [c for c in O.columns if c.startswith("p_")]
        pi = V[pc].values.mean(0)
        P_raw = P_raw + O[pc].values; P_s = P_s + reweight(O[pc].values, 1 / pi)
    T = pd.concat([pd.read_csv(os.path.join(a.runs, f"{a.exp}_f{k}", "test_predictions.csv")) for k in a.folds])
    return O, T, pc, P_raw / len(a.folds), P_s / len(a.folds)


def metrics(y, pr, m, n, K):
    out = {"mel_sensitivity": np.mean(pr[y == m] == m), "mel_to_nv": np.mean(pr[y == m] == n),
           "nv_recall": np.mean(pr[y == n] == n)}
    rec = [np.mean(pr[y == j] == j) for j in range(K) if (y == j).any()]
    out["balanced_acc"] = float(np.mean(rec))
    return out


def boot(y, prs, groups, m, n, K, B, seed=0):
    """lesion-cluster bootstrap; prs = {'raw': hard preds, 'simple': hard preds}"""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable"); idx = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    obs = {k: metrics(y, v, m, n, K) for k, v in prs.items()}
    samples = {k: {mm: [] for mm in obs["raw"]} for k in list(prs) + ["diff"]}
    for _ in range(B):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        r = {k: metrics(y[ii], v[ii], m, n, K) for k, v in prs.items()}
        for mm in r["raw"]:
            samples["raw"][mm].append(r["raw"][mm]); samples["simple"][mm].append(r["simple"][mm])
            samples["diff"][mm].append(r["simple"][mm] - r["raw"][mm])
    rows = []
    for mm in obs["raw"]:
        ci = lambda v: f"[{np.percentile(v, 2.5)*100:.1f}, {np.percentile(v, 97.5)*100:.1f}]"
        d = np.array(samples["diff"][mm])
        rows.append({"metric": mm, "raw_%": obs["raw"][mm] * 100, "raw_CI": ci(samples["raw"][mm]),
                     "simple_%": obs["simple"][mm] * 100, "simple_CI": ci(samples["simple"][mm]),
                     "diff_pts": (obs["simple"][mm] - obs["raw"][mm]) * 100, "diff_CI": ci(d),
                     "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    return pd.DataFrame(rows)


def cond_conf(y, pr, K):
    C = np.zeros((K, K)); np.add.at(C, (pr, y), 1.0)
    return C / C.sum(0, keepdims=True).clip(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ham_baseline"); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--ood", default="runs_ood_pf"); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sources", default="runs_adapt/sources.csv"); ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    O, T, pc, Praw, Ps = load(a)
    cl = [c[2:] for c in pc]; K = len(cl); m, n = cl.index("mel"), cl.index("nv")
    y, g = O["y"].values, O["group"].astype(str).values
    hard = {"raw": Praw.argmax(1), "simple": Ps.argmax(1)}
    if os.path.exists(a.sources):
        src = pd.read_csv(a.sources).set_index("image_path").loc[O["image_path"], "source"].values
    else:
        src = np.array([source_proxy(p) for p in O["image_path"]])
    L = []; P = L.append
    pd.set_option("display.width", 220)
    P(f"Safety analysis | {a.exp} | OOD n={len(y)} | 5-fold ensemble | 95% lesion-cluster bootstrap ({a.n_boot})\n")

    P("== 1. all OOD images ==")
    P(boot(y, hard, g, m, n, K, a.n_boot).round(3).to_string(index=False))
    rr = pd.DataFrame({r: [np.mean(hard[r][y == j] == j) for j in range(K)] for r in hard}, index=cl)
    P("per-class recall:"); P(rr.T.round(3).to_string() + "\n")

    P("== 2. where do true nevi go? (% of all nevi) ==")
    P(pd.DataFrame({r: [np.mean(hard[r][y == n] == j) * 100 for j in range(K)] for r in hard}, index=[f"pred {c}" for c in cl]).T.round(1).to_string() + "\n")

    P("== 3. per source proxy ==")
    for s in pd.Series(src).value_counts().index:
        mk = src == s
        if mk.sum() < 200:
            continue
        P(f"-- {s} (n={mk.sum()}, melanomas={(y[mk] == m).sum()}, nevi={(y[mk] == n).sum()})")
        P(boot(y[mk], {k: v[mk] for k, v in hard.items()}, g[mk], m, n, K, a.n_boot).round(3).to_string(index=False))
    P("")

    P("== 4. per source: P(yhat=nv | y) and BBSE nevus-prior error decomposition (raw predictions) ==")
    yS, pS = T["y"].values, T[pc].values.argmax(1)
    CS = cond_conf(yS, pS, K); Cinv = np.linalg.pinv(CS)
    rows, dec = [{"set": "source (own test folds)", **{c: CS[n, j] for j, c in enumerate(cl)}}], []
    for s in pd.Series(src).value_counts().index:
        mk = src == s
        if mk.sum() < 200:
            continue
        CT = cond_conf(y[mk], hard["raw"][mk], K); pi = np.bincount(y[mk], minlength=K) / mk.sum()
        rows.append({"set": s, **{c: CT[n, j] for j, c in enumerate(cl)}})
        contrib = (Cinv @ (CT - CS))[n] * pi * 100
        dec.append({"source": s, **{c: contrib[j] for j, c in enumerate(cl)}, "TOTAL nv error": contrib.sum()})
    P("P(yhat=nv | true class):"); P(pd.DataFrame(rows).set_index("set").round(3).to_string())
    P("contribution of each true class to the BBSE nevus-prior error (percentage points):")
    P(pd.DataFrame(dec).set_index("source").round(2).to_string())
    txt = "\n".join(L)
    out = a.out or f"safety_results_{a.exp}.txt"
    open(out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
