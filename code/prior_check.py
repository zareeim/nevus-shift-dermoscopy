import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from label_shift import FoldAdapter, reweight
from matched import EXPS, TARGETS, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=EXPS); ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--runs", default="runs_chromo_cv"); ap.add_argument("--out", default="prior_check.txt")
    a = ap.parse_args()
    L = []; P = L.append
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    summ = []
    for exp in a.exps:
        if not os.path.exists(os.path.join(a.runs, f"{exp}_f0", "val_predictions.csv")):
            continue
        vals, rows = [], []
        for k in range(5):
            r = os.path.join(a.runs, f"{exp}_f{k}")
            V = pd.read_csv(os.path.join(r, "val_predictions.csv")); T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
            pc = [c for c in V.columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl)
            m, n = cl.index("mel"), cl.index("nv")
            ad = FoldAdapter(V[pc].values, V["y"].values, K)
            cnt = np.bincount(V["y"].values, minlength=K).astype(float) + 1
            raw = cnt / cnt.sum()
            implied = np.sqrt(cnt); implied /= implied.sum()          # w_y * pi_y with w_y ~ 1/sqrt(n_y)
            priors = {"effective (mean val prediction)": ad.pi_p, "loss-implied (sqrt counts)": implied,
                      "raw label frequency": raw, "none (raw argmax)": np.ones(K) / K}
            vals.append(pd.DataFrame({"raw label freq": raw, "loss-implied": implied, "mean val prediction": ad.pi_p}, index=cl))
            sets = {"SOURCE test fold": (T["y"].values, T[pc].values)}
            for t in a.targets:
                tg, ood = t.split("=")
                f = os.path.join(ood, f"{exp}_f{k}", "ood_predictions.csv")
                if os.path.exists(f):
                    O = pd.read_csv(f); sets[tg] = (O["y"].values, O[pc].values)
            for sname, (y, p) in sets.items():
                for pname, pr in priors.items():
                    mt = metrics(y, reweight(p, 1 / pr), m, n, K)
                    rows.append({"set": sname, "prior": pname, "fold": k,
                                 "BA": mt["BA"], "mel_sens": mt["mel_sens"], "nv_recall": mt["nv_recall"]})
        P(f"===== {exp} =====")
        P("1. per-class prior values (mean over 5 folds):")
        P((sum(vals) / len(vals)).round(3).to_string())
        R = pd.DataFrame(rows)
        G = R.groupby(["set", "prior"], sort=False)[["BA", "mel_sens", "nv_recall"]].agg(["mean", "min", "max"]).round(1)
        P("2. divided rule per prior (per-fold mean [min, max], %):"); P(G.to_string()); P("")
        M = R.groupby(["set", "prior"], sort=False)["BA"].mean().unstack()
        for s in M.index:
            summ.append({"exp": exp, "set": s, **M.loc[s].round(1).to_dict()})
    S = pd.DataFrame(summ)
    P("===== SUMMARY: balanced accuracy (%) of the divided rule, per-fold mean =====")
    P(S.to_string(index=False))
    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
