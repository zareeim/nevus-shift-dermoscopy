import argparse
import os
import warnings
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from label_shift import FoldAdapter, bbse_prior, reweight

EXPS = ["ham_baseline", "ham_baseline_s43", "ham_baseline_s44", "ham_effb3_baseline", "ham_effb3_baseline_s43",
        "ham_effb3_baseline_s44", "ham_convnext_baseline", "ham_convnext_baseline_s43", "ham_convnext_baseline_s44",
        "ham_swin_baseline", "ham_swin_baseline_s43", "ham_swin_baseline_s44", "bcn_baseline"]
TARGETS = ["ISIC2019=runs_ood_pf", "Derm7pt=runs_ood_d7", "ISIC2020=runs_ood_i20"]


def rate(y, h, ts, ps):
    m = np.isin(y, ts)
    return np.mean(np.isin(h[m], ps)) * 100 if m.any() else np.nan


def metrics(y, p, m, n, K):
    h = p.argmax(1); other = [j for j in range(K) if j not in (m, n)]
    return {"BA": balanced_accuracy_score(y, h) * 100, "acc": (h == y).mean() * 100,
            "mel_sens": rate(y, h, [m], [m]), "mel->nv": rate(y, h, [m], [n]),
            "other->nv": rate(y, h, other, [n]), "nv->mel": rate(y, h, [n], [m]), "nv_recall": rate(y, h, [n], [n])}


def fast_ba(y, h, K):
    tot = np.bincount(y, minlength=K); ok = np.bincount(y[h == y], minlength=K)
    return (ok[tot > 0] / tot[tot > 0]).mean() * 100


def matched_boot(store, yT, gT, m, n, K, B, rng):
    """Lesion-cluster bootstrap of FOLD-AVERAGED matched differences. Each replicate resamples the target
    lesions once (shared by all five fold models, so a target image is never counted five times) and the
    lesions of every source test fold; per-fold differences are computed and then averaged over folds."""
    other = [j for j in range(K) if j not in (m, n)]
    from extra_analysis import cluster_index
    idxT = cluster_index(gT); idxS = [cluster_index(st[1]) for st in store]
    out = {"drop (S-T BA)": [], "mel->nv increase": [], "other->nv increase": [], "simple-raw BA (T)": [],
           "simple-raw mel_sens (T)": []}
    for _ in range(B):
        iT = np.concatenate([idxT[j] for j in rng.integers(0, len(idxT), len(idxT))]); yt = yT[iT]
        v = {k_: [] for k_ in out}
        for (yS, gS, hSr, hSs, hTr, hTs), ix in zip(store, idxS):
            iS = np.concatenate([ix[j] for j in rng.integers(0, len(ix), len(ix))]); ys = yS[iS]
            tr, ts, sr = hTr[iT], hTs[iT], hSr[iS]
            v["drop (S-T BA)"].append(fast_ba(ys, sr, K) - fast_ba(yt, tr, K))
            v["mel->nv increase"].append(rate(yt, tr, [m], [n]) - rate(ys, sr, [m], [n]))
            v["other->nv increase"].append(rate(yt, tr, other, [n]) - rate(ys, sr, other, [n]))
            v["simple-raw BA (T)"].append(fast_ba(yt, ts, K) - fast_ba(yt, tr, K))
            v["simple-raw mel_sens (T)"].append(rate(yt, ts, [m], [m]) - rate(yt, tr, [m], [m]))
        for k_ in out:
            out[k_].append(np.nanmean(v[k_]))
    return {k_: np.array(v_) for k_, v_ in out.items()}


def cond(y, h, K):
    C = np.zeros((K, K)); np.add.at(C, (h, y), 1.0)
    return C / C.sum(0, keepdims=True).clip(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=EXPS); ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--runs", default="runs_chromo_cv"); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="matched_results.txt"); ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--primary", action="store_true",
                    help="also compute two-sided bootstrap p-values of the 3 primary tests per exp/target and Holm-correct them")
    a = ap.parse_args()
    ptests = []
    L = []; P = L.append
    pd.set_option("display.width", 240)
    P("Matched-protocol analysis: fold model k on its own source test fold vs the same model on the target.\n"
      "Values: mean over 5 folds [min, max]; 'ens' = 5-model ensemble on the target (as reported before).\n")
    summary = []
    for exp in a.exps:
        for t in a.targets:
            tg, ood = t.split("=")
            files = [os.path.join(ood, f"{exp}_f{k}", "ood_predictions.csv") for k in a.folds]
            if not all(os.path.exists(f) for f in files) or not os.path.exists(os.path.join(a.runs, f"{exp}_f0", "test_predictions.csv")):
                continue
            rows, bb, dec, dec_t, pri, store = [], [], [], [], [], []
            ens_raw, ens_s = 0, 0
            for k, f in zip(a.folds, files):
                r = os.path.join(a.runs, f"{exp}_f{k}")
                V = pd.read_csv(os.path.join(r, "val_predictions.csv")); T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
                O = pd.read_csv(f).sort_values("image_path").reset_index(drop=True)
                pc = [c for c in V.columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl)
                m, n = cl.index("mel"), cl.index("nv")
                ad = FoldAdapter(V[pc].values, V["y"].values, K)
                yS, pS, yT, pT = T["y"].values, T[pc].values, O["y"].values, O[pc].values
                for rule, fS, fT in (("raw", pS, pT), ("simple", reweight(pS, 1 / ad.pi_p), reweight(pT, 1 / ad.pi_p))):
                    s, tt = metrics(yS, fS, m, n, K), metrics(yT, fT, m, n, K)
                    rows.append({"fold": k, "rule": rule, **{f"S_{x}": v for x, v in s.items()}, **{f"T_{x}": v for x, v in tt.items()}})
                store.append((yS, T["group"].astype(str).values, pS.argmax(1), reweight(pS, 1 / ad.pi_p).argmax(1),
                              pT.argmax(1), reweight(pT, 1 / ad.pi_p).argmax(1)))
                gT = O["group"].astype(str).values
                ens_raw = ens_raw + pT / len(files); ens_s = ens_s + reweight(pT, 1 / ad.pi_p) / len(files)
                # M2: BBSE as estimated, and its exact decomposition with the same matrices
                piT = np.bincount(yT, minlength=K) / len(yT)
                est, condn = bbse_prior(V[pc].values, V["y"].values, pT, K)
                Cv = cond(V["y"].values, V[pc].values.argmax(1), K); Ct = cond(yT, pT.argmax(1), K)
                inv = np.linalg.pinv(Cv)
                contrib = (inv @ (Ct - Cv))[n] * piT * 100
                bb.append({"fold": k, "BBSE_nv_error_as_estimated": (est[n] - piT[n]) * 100,
                           "identity_unclipped": contrib.sum(), "cond_number": condn})
                dec.append(contrib)
                Cs = cond(yS, pS.argmax(1), K)
                dec_t.append((np.linalg.pinv(Cs) @ (Ct - Cs))[n] * piT * 100)
                # M3: simple rule with different source priors
                train_prior = np.bincount(V["y"].values, minlength=K).astype(float) + 1; train_prior /= train_prior.sum()
                for name, pr, src in (("effective (mean val prediction)", ad.pi_p, pT), ("raw label frequency", train_prior, pT),
                                      ("BCTS-calibrated mean", ad.pi_q, ad.cal(pT))):
                    q = reweight(src, 1 / pr)
                    pri.append({"fold": k, "prior": name, **{x: v for x, v in metrics(yT, q, m, n, K).items() if x in ("BA", "mel_sens", "nv_recall")}})
            R = pd.DataFrame(rows)
            P(f"===== {exp} | target {tg} | n_target={len(yT)} =====")
            P("-- M1. matched per-fold metrics (%)")
            for rule in ("raw", "simple"):
                d = R[R["rule"] == rule].drop(columns=["fold", "rule"])
                ens = metrics(yT, ens_raw if rule == "raw" else ens_s, m, n, K)
                t_ = pd.DataFrame({"source (per fold)": [f"{d['S_' + x].mean():.1f} [{d['S_' + x].min():.1f}, {d['S_' + x].max():.1f}]" for x in ens],
                                   "target (per fold)": [f"{d['T_' + x].mean():.1f} [{d['T_' + x].min():.1f}, {d['T_' + x].max():.1f}]" for x in ens],
                                   "target - source (per fold)": [f"{(d['T_' + x] - d['S_' + x]).mean():+.1f} [{(d['T_' + x] - d['S_' + x]).min():+.1f}, {(d['T_' + x] - d['S_' + x]).max():+.1f}]" for x in ens],
                                   "target (ens)": [f"{v:.1f}" for v in ens.values()]}, index=list(ens))
                P(f"rule = {rule}"); P(t_.to_string())
            if True:
                rr, ss = R[R["rule"] == "raw"], R[R["rule"] == "simple"]
                summary.append({"exp": exp, "target": tg,
                                "S_BA": rr["S_BA"].mean(), "T_BA": rr["T_BA"].mean(), "drop": (rr["S_BA"] - rr["T_BA"]).mean(),
                                "mel->nv S": rr["S_mel->nv"].mean(), "mel->nv T": rr["T_mel->nv"].mean(),
                                "other->nv S": rr["S_other->nv"].mean(), "other->nv T": rr["T_other->nv"].mean(),
                                "nv->mel S": rr["S_nv->mel"].mean(), "nv->mel T": rr["T_nv->mel"].mean(),
                                "simple-raw BA (T)": (ss["T_BA"].values - rr["T_BA"].values).mean(),
                                "simple-raw mel_sens (T)": (ss["T_mel_sens"].values - rr["T_mel_sens"].values).mean(),
                                "BBSE nv err (est)": np.mean([b["BBSE_nv_error_as_estimated"] for b in bb]),
                                "BBSE nv err (identity)": np.mean([b["identity_unclipped"] for b in bb]),
                                "mel contribution": np.mean([d_[m] for d_ in dec])})
            bs = matched_boot(store, yT, gT, m, n, K, a.n_boot, np.random.default_rng(0))
            pt_ = {"drop (S-T BA)": (R[R.rule == "raw"]["S_BA"] - R[R.rule == "raw"]["T_BA"]).mean(),
                   "mel->nv increase": (R[R.rule == "raw"]["T_mel->nv"] - R[R.rule == "raw"]["S_mel->nv"]).mean(),
                   "other->nv increase": (R[R.rule == "raw"]["T_other->nv"] - R[R.rule == "raw"]["S_other->nv"]).mean(),
                   "simple-raw BA (T)": R[R.rule == "simple"]["T_BA"].mean() - R[R.rule == "raw"]["T_BA"].mean(),
                   "simple-raw mel_sens (T)": R[R.rule == "simple"]["T_mel_sens"].mean() - R[R.rule == "raw"]["T_mel_sens"].mean()}
            P(f"-- matched fold-averaged differences with 95% lesion-cluster CIs ({a.n_boot} replicates; target lesions resampled once per replicate)")
            P(pd.DataFrame({"estimate": pt_, "CI_low": {k_: np.nanpercentile(v_, 2.5) for k_, v_ in bs.items()},
                            "CI_high": {k_: np.nanpercentile(v_, 97.5) for k_, v_ in bs.items()}}).round(2).to_string())
            if a.primary:
                # two-sided bootstrap p-value with plus-one correction: H0 "difference = 0";
                # p = min(1, 2 * min((1 + #{d* <= 0}) / (B + 1), (1 + #{d* >= 0}) / (B + 1)))
                for key in ("mel->nv increase", "other->nv increase", "simple-raw BA (T)"):
                    d_ = bs[key][~np.isnan(bs[key])]; B_ = len(d_); t_ = pt_[key]
                    # main: null-centred bootstrap test (Hall & Wilson 1991): the replicates are centred on the
                    # observed estimate, d* - t, which approximates the null distribution of the statistic;
                    # p = (1 + #{|d* - t| >= |t|}) / (B + 1)
                    p_c = (1 + (np.abs(d_ - t_) >= abs(t_)).sum()) / (B_ + 1)
                    # sensitivity: inversion of the two-sided percentile interval (smallest alpha whose interval excludes 0)
                    p_ = min(1.0, 2 * min((1 + (d_ <= 0).sum()) / (B_ + 1), (1 + (d_ >= 0).sum()) / (B_ + 1)))
                    ptests.append({"exp": exp, "target": tg, "test": key, "estimate": t_,
                                   "CI_low": np.nanpercentile(d_, 2.5), "CI_high": np.nanpercentile(d_, 97.5),
                                   "B": B_, "p_centred": p_c, "p_pct_inversion": p_})
            B = pd.DataFrame(bb)
            P("-- M2. BBSE nevus-prior error per fold (points): as estimated (clipped) | exact unclipped decomposition")
            P(B.round(2).to_string(index=False))
            P("mean contribution of each true class (validation C_S, same matrices as the estimator):")
            P(pd.Series(np.mean(dec, 0), index=cl).round(2).to_string())
            P("same with the source TEST fold as C_S (larger, sensitivity check):")
            P(pd.Series(np.mean(dec_t, 0), index=cl).round(2).to_string() + f"\nTOTAL {np.sum(np.mean(dec_t, 0)):.2f}")
            Pr = pd.DataFrame(pri).groupby("prior")[["BA", "mel_sens", "nv_recall"]].agg(["mean", "min", "max"]).round(1)
            P("-- M3. simple rule on the target with different source priors (per-fold mean [min, max])")
            P(Pr.to_string()); P("")
    if a.primary and ptests:
        T_ = pd.DataFrame(ptests)

        def holm_(p):
            p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
            for r_, i_ in enumerate(o):      # Holm step-down
                run = max(run, (len(p) - r_) * p[i_]); adj[i_] = min(1.0, run)
            return adj
        T_["holm_centred"] = holm_(T_["p_centred"]); T_["sig_centred"] = T_["holm_centred"] < 0.05
        T_["holm_pct_inversion"] = holm_(T_["p_pct_inversion"]); T_["sig_pct_inversion"] = T_["holm_pct_inversion"] < 0.05
        P(f"===== PRIMARY TESTS (matched protocol, fold-averaged, paired cluster bootstrap B={a.n_boot}; two-sided; "
          f"main = null-centred bootstrap p, sensitivity = percentile-interval inversion; plus-one; Holm over {len(T_)} tests; "
          f"smallest attainable raw p = {1 / (a.n_boot + 1):.5f} (centred), {2 / (a.n_boot + 1):.5f} (inversion)) =====")
        P(T_.round(5).to_string(index=False)); P("")
        T_.to_csv(os.path.splitext(a.out)[0] + "_primary.csv", index=False)
    S = pd.DataFrame(summary)
    if len(S):
        P("===== SUMMARY (per-fold means, raw rule unless stated) =====")
        P(S.round(1).to_string(index=False))
    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
