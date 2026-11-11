import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from label_shift import FoldAdapter, reweight, bbse_prior, resample_to_prior
from matched import fast_ba as _fba


def fast_ba(y, h, K):          # matched.fast_ba returns percent; this script works in fractions
    return _fba(y, h, K) / 100
from extra_analysis import cluster_index

ARCH = {"ResNet-50": "ham_baseline", "EfficientNet-B3": "ham_effb3_baseline",
        "ConvNeXt-T": "ham_convnext_baseline", "Swin-T": "ham_swin_baseline"}
SEEDS = {"": 42, "_s43": 43, "_s44": 44}
RUNS, OOD, B = "runs_chromo_cv", "runs_ood_pf", 1000
METHODS = ["em", "bcts_em", "bbse", "bcts_bbse"]


def condm(y, h, K):
    C = np.zeros((K, K)); np.add.at(C, (h, y), 1.0)
    return C / C.sum(0, keepdims=True).clip(1)


def main():
    rng = np.random.default_rng(0); rows = []
    for arch, base in ARCH.items():
        for s, seed in SEEDS.items():
            exp = base + s
            r1, r2, r3S, r3T, err, contrib, simple_T, raw_T, meth = [], [], [], [], [], [], [], [], {x: [] for x in METHODS}
            hT_raw, hT_s, yT, gT = [], [], None, None
            ok = True
            for k in range(5):
                r = os.path.join(RUNS, f"{exp}_f{k}"); f = os.path.join(OOD, f"{exp}_f{k}", "ood_predictions.csv")
                if not (os.path.exists(f) and os.path.exists(os.path.join(r, "test_predictions.csv"))):
                    ok = False; break
                V = pd.read_csv(os.path.join(r, "val_predictions.csv")); T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
                O = pd.read_csv(f).sort_values("image_path").reset_index(drop=True)
                pc = [c for c in V.columns if c.startswith("p_")]; K = len(pc); cl = [c[2:] for c in pc]
                m, n = cl.index("mel"), cl.index("nv")
                ad = FoldAdapter(V[pc].values, V["y"].values, K)
                yS, pS = T["y"].values, T[pc].values; y, pT = O["y"].values, O[pc].values
                if yT is None:
                    yT, gT = y, O["group"].astype(str).values
                piT = np.bincount(y, minlength=K) / len(y)
                # R1
                r1.append((fast_ba(yS, pS.argmax(1), K) - fast_ba(y, pT.argmax(1), K)) * 100)
                # R2
                base_ba = fast_ba(yS, pS.argmax(1), K); d = T.reset_index(drop=True)
                for _ in range(20):
                    rs = resample_to_prior(d, piT, rng)
                    r2.append((fast_ba(rs["y"].values, rs[pc].values.argmax(1), K) - base_ba) * 100)
                # R3
                r3S.append(np.mean(pS[yS == m].argmax(1) == n) * 100); r3T.append(np.mean(pT[y == m].argmax(1) == n) * 100)
                # R4
                est, _ = bbse_prior(V[pc].values, V["y"].values, pT, K); err.append((est[n] - piT[n]) * 100)
                Cv = condm(V["y"].values, V[pc].values.argmax(1), K); Ct = condm(y, pT.argmax(1), K)
                contrib.append((np.linalg.pinv(Cv) @ (Ct - Cv))[n] * piT * 100)
                # R5 / R6
                out, _, _ = ad.apply(pT)
                hr, hs = out["raw"].argmax(1), out["simple"].argmax(1)
                hT_raw.append(hr); hT_s.append(hs)
                raw_T.append(fast_ba(y, hr, K) * 100); simple_T.append(fast_ba(y, hs, K) * 100)
                for x in METHODS:
                    meth[x].append(fast_ba(y, out[x].argmax(1), K) * 100)
            if not ok:
                print(f"skip {exp}: predictions missing"); continue
            c = np.mean(contrib, 0); pos = {cl[j]: c[j] for j in range(K) if j != n}
            idx = cluster_index(gT); bs = []
            for _ in range(B):
                ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
                bs.append(np.mean([(fast_ba(yT[ii], hs[ii], K) - fast_ba(yT[ii], hr[ii], K)) * 100
                                   for hr, hs in zip(hT_raw, hT_s)]))
            lo, hi = np.percentile(bs, [2.5, 97.5]); best = max(METHODS, key=lambda x: np.mean(meth[x]))
            row = {"architecture": arch, "seed": seed,
                   "R1 drop": round(np.mean(r1), 1), "R1": np.mean(r1) > 20,
                   "R2 BA change": round(np.mean(r2), 2), "R2": abs(np.mean(r2)) < 2,
                   "R3 mel->nv S->T": f"{np.mean(r3S):.1f}->{np.mean(r3T):.1f}", "R3": np.mean(r3T) > np.mean(r3S),
                   "R4 nv err": round(np.mean(err), 1), "R4 largest": max(pos, key=pos.get),
                   "R4": np.mean(err) > 0 and max(pos, key=pos.get) == "mel",
                   "R5 simple-raw [CI]": f"{np.mean(simple_T) - np.mean(raw_T):+.1f} [{lo:+.1f}, {hi:+.1f}]", "R5": lo > 0,
                   "R6 simple vs best": f"{np.mean(simple_T):.1f} vs {np.mean(meth[best]):.1f} ({best})",
                   "R6": np.mean(simple_T) >= np.mean(meth[best])}
            row["all six"] = all(row[f"R{i}"] for i in range(1, 7))
            rows.append(row); print(f"done {exp}", flush=True)
    R = pd.DataFrame(rows); R.to_csv("s1_table.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    txt = R.to_string(index=False) + f"\n\nconfigurations meeting all six: {int(R['all six'].sum())} of {len(R)}\nDONE\n"
    open("s1_table.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
