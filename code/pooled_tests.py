import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from label_shift import FoldAdapter, reweight
from matched import rate, fast_ba
from extra_analysis import cluster_index

ARCH = {"ResNet-50": "ham_baseline", "EfficientNet-B3": "ham_effb3_baseline",
        "ConvNeXt-T": "ham_convnext_baseline", "Swin-T": "ham_swin_baseline"}
SEEDS = ["", "_s43", "_s44"]
TESTS = ["mel->nv increase", "other->nv increase", "simple-raw BA (T)"]


def load(runs, ood, exp, k):
    r = os.path.join(runs, f"{exp}_f{k}")
    V = pd.read_csv(os.path.join(r, "val_predictions.csv"))
    T = pd.read_csv(os.path.join(r, "test_predictions.csv")).sort_values("image_path").reset_index(drop=True)
    O = pd.read_csv(os.path.join(ood, f"{exp}_f{k}", "ood_predictions.csv")).sort_values("image_path").reset_index(drop=True)
    pc = [c for c in V.columns if c.startswith("p_")]
    ad = FoldAdapter(V[pc].values, V["y"].values, len(pc))
    return (T, O, [c[2:] for c in pc], T[pc].values.argmax(1), O[pc].values.argmax(1),
            reweight(O[pc].values, 1 / ad.pi_p).argmax(1))


def stats(models, iS, iT, yS, yT, m, n, other, K):
    v = {t: [] for t in TESTS}
    for k, hS, hT, hTs in models:
        ys, sr = yS[k][iS[k]], hS[iS[k]]
        yt, tr, ts = yT[iT], hT[iT], hTs[iT]
        v["mel->nv increase"].append(rate(yt, tr, [m], [n]) - rate(ys, sr, [m], [n]))
        v["other->nv increase"].append(rate(yt, tr, other, [n]) - rate(ys, sr, other, [n]))
        v["simple-raw BA (T)"].append(fast_ba(yt, ts, K) - fast_ba(yt, tr, K))
    return {t: np.nanmean(x) for t, x in v.items()}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r_, i_ in enumerate(o):
        run = max(run, (len(p) - r_) * p[i_]); adj[i_] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--targets", nargs="+", default=["ISIC2019=runs_ood_pf", "Derm7pt=runs_ood_d7"])
    ap.add_argument("--n_boot", type=int, default=10000); ap.add_argument("--out", default="pooled_tests.txt")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for arch, base in ARCH.items():
        for t in a.targets:
            tg, ood = t.split("=")
            models, yS, gS, yT, gT, paths = [], {}, {}, None, None, {}
            for s in SEEDS:
                for k in range(5):
                    T, O, cl, hS, hT, hTs = load(a.runs, ood, base + s, k)
                    if k in paths:      # same test fold for every seed
                        assert (T["image_path"].values == paths[k]).all(), f"test fold {k} differs between seeds"
                    else:
                        paths[k] = T["image_path"].values; yS[k] = T["y"].values; gS[k] = T["group"].astype(str).values
                    if yT is None:
                        yT, gT, tpaths = O["y"].values, O["group"].astype(str).values, O["image_path"].values
                    assert (O["image_path"].values == tpaths).all()
                    models.append((k, hS, hT, hTs))
            K = len(cl); m, n = cl.index("mel"), cl.index("nv"); other = [j for j in range(K) if j not in (m, n)]
            full = {k: np.arange(len(yS[k])) for k in yS}
            est = stats(models, full, np.arange(len(yT)), yS, yT, m, n, other, K)
            idxT = cluster_index(gT); idxS = {k: cluster_index(gS[k]) for k in gS}
            bs = {t_: [] for t_ in TESTS}
            for _ in range(a.n_boot):
                iT = np.concatenate([idxT[j] for j in rng.integers(0, len(idxT), len(idxT))])
                iS = {k: np.concatenate([ix[j] for j in rng.integers(0, len(ix), len(ix))]) for k, ix in idxS.items()}
                for t_, v_ in stats(models, iS, iT, yS, yT, m, n, other, K).items():
                    bs[t_].append(v_)
            for t_ in TESTS:
                d = np.array(bs[t_]); d = d[~np.isnan(d)]; B = len(d); e = est[t_]
                rows.append({"architecture": arch, "target": tg, "test": t_, "estimate (3 seeds x 5 folds)": e,
                             "CI_low": np.percentile(d, 2.5), "CI_high": np.percentile(d, 97.5), "B": B,
                             "p_centred": (1 + (np.abs(d - e) >= abs(e)).sum()) / (B + 1),
                             "p_pct_inversion": min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (B + 1))})
            print(f"done {arch} {tg}", flush=True)
    R = pd.DataFrame(rows)
    R["holm_centred"] = holm(R["p_centred"]); R["sig_centred"] = R["holm_centred"] < 0.05
    R["holm_pct_inversion"] = holm(R["p_pct_inversion"]); R["sig_pct_inversion"] = R["holm_pct_inversion"] < 0.05
    R.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    txt = (f"Primary tests, seed-pooled estimand (mean over 3 seeds x 5 fold models), paired cluster bootstrap "
           f"B={a.n_boot}, null-centred two-sided p (sensitivity: percentile inversion), Holm over {len(R)} tests\n"
           + R.round(5).to_string(index=False) + f"\n\nsignificant (centred, Holm): {R['sig_centred'].sum()} of {len(R)}"
           f" | (inversion, Holm): {R['sig_pct_inversion'].sum()} of {len(R)}\nDONE\n")
    open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
