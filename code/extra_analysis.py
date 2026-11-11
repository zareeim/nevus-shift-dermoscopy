import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve

from label_shift import FoldAdapter, METHODS, reweight, resample_to_prior

DEFAULT_EXPS = ["ham_baseline", "ham_effb3_baseline", "ham_convnext_baseline", "ham_swin_baseline",
                "bcn_baseline", "ham_baseline_s43", "ham_baseline_s44"]
EST = ["em", "bcts_em", "bbse", "bcts_bbse"]


# ------------------------------------------------------------------ helpers
def cluster_index(groups):
    codes, _ = pd.factorize(pd.Series(groups).astype(str))
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def boot(fn, groups, n_boot, rng):
    """fn(indices) -> float; returns array of bootstrap values (lesion-cluster resampling)."""
    idx = cluster_index(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = fn(np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))]))
    return out


def ci(v, scale=100):
    return f"[{np.nanpercentile(v, 2.5)*scale:.1f}, {np.nanpercentile(v, 97.5)*scale:.1f}]"


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def load(a, exp):
    folds = {}
    for k in a.folds:
        r = os.path.join(a.runs, f"{exp}_f{k}")
        V = pd.read_csv(os.path.join(r, "val_predictions.csv"))
        T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
        O = pd.read_csv(os.path.join(a.ood, f"{exp}_f{k}", "ood_predictions.csv")).sort_values("image_path").reset_index(drop=True)
        folds[k] = (V, T, O)
    return folds


def available(a, exp):
    return all(os.path.exists(os.path.join(a.ood, f"{exp}_f{k}", "ood_predictions.csv")) and
               os.path.exists(os.path.join(a.runs, f"{exp}_f{k}", "test_predictions.csv")) for k in a.folds)


# ------------------------------------------------------------------ one experiment
def analyse(a, exp):
    rng = np.random.default_rng(a.seed)
    folds = load(a, exp)
    pc = [c for c in folds[a.folds[0]][0].columns if c.startswith("p_")]
    cl = [c[2:] for c in pc]; K = len(cl); m, n = cl.index("mel"), cl.index("nv")
    O0 = folds[a.folds[0]][2]
    yO, gO = O0["y"].values, O0["group"].astype(str).values
    for k, (_, _, O) in folds.items():
        assert (O["image_path"].values == O0["image_path"].values).all(), f"fold {k}: target files differ"
    T_all = pd.concat([f[1] for f in folds.values()], ignore_index=True)
    yS, gS = T_all["y"].values, T_all["group"].astype(str).values
    pi_T = np.bincount(yO, minlength=K) / len(yO)
    pi_S = np.bincount(yS, minlength=K) / len(yS)
    ad = {k: FoldAdapter(V[pc].values, V["y"].values, K) for k, (V, _, _) in folds.items()}

    L = []; P = L.append
    pd.set_option("display.width", 220)
    P(f"Extra analyses | {exp} | source out-of-fold n={len(yS)} | target n={len(yO)} | "
      f"{len(folds)} folds | {a.n_boot} lesion-cluster bootstraps\n")
    summ = {"exp": exp}

    # ---------------- G. three controlled conditions
    P(f"== G. prior estimators under three conditions ({a.n_synth} lesion-level resamples per fold for A and B) ==")
    rows = []
    for cond in ("A pure label shift", "B pure class-conditional", "C real target"):
        rec = []
        for k, (V, T, O) in folds.items():
            draws = 1 if cond.startswith("C") else a.n_synth
            for _ in range(draws):
                if cond.startswith("A"):
                    D = resample_to_prior(T.reset_index(drop=True), pi_T, rng)
                elif cond.startswith("B"):
                    D = resample_to_prior(O.reset_index(drop=True), pi_S, rng)
                else:
                    D = O
                y = D["y"].values; truth = np.bincount(y, minlength=K) / len(y)
                out, pri, _ = ad[k].apply(D[pc].values)
                r = {f"BA_{mm}": balanced_accuracy_score(y, out[mm].argmax(1)) for mm in METHODS}
                for e in EST:
                    r[f"MAE_{e}"] = np.abs(pri[e] - truth).mean()
                    r[f"nvErr_{e}"] = (pri[e][n] - truth[n]) * 100
                    r[f"melErr_{e}"] = (pri[e][m] - truth[m]) * 100
                rec.append(r)
        s = pd.DataFrame(rec).mean()
        rows.append({"condition": cond, **s.to_dict()})
    G = pd.DataFrame(rows).set_index("condition")
    P("balanced accuracy (%) per method:")
    P((G[[f"BA_{mm}" for mm in METHODS]] * 100).rename(columns=lambda c: c[3:]).round(1).to_string())
    P("prior estimation: MAE (fraction) | error of the NEVUS prior (points) | error of the MELANOMA prior (points)")
    t = pd.concat({"MAE": G[[f"MAE_{e}" for e in EST]].rename(columns=lambda c: c[4:]),
                   "nv_error": G[[f"nvErr_{e}" for e in EST]].rename(columns=lambda c: c[6:]),
                   "mel_error": G[[f"melErr_{e}" for e in EST]].rename(columns=lambda c: c[7:])}, axis=1)
    P(t.round(3).to_string())
    P("reading: in B the true prior is the source prior, so any nevus-prior error there comes only from the\n"
      "class-conditional shift; A isolates the prior change; C is the real target.\n")
    for e in ("bbse", "bcts_em"):
        for c, tag in (("A pure label shift", "A"), ("B pure class-conditional", "B"), ("C real target", "C")):
            summ[f"{e}_nvErr_{tag}"] = G.loc[c, f"nvErr_{e}"]

    # ensemble predictions (raw and simple), source out-of-fold and target
    P_S = np.concatenate([folds[k][1][pc].values for k in folds])
    Ps_S = np.concatenate([reweight(folds[k][1][pc].values, 1 / ad[k].pi_p) for k in folds])
    P_T = np.mean([O[pc].values for (_, _, O) in folds.values()], 0)
    Ps_T = np.mean([reweight(O[pc].values, 1 / ad[k].pi_p) for k, (_, _, O) in folds.items()], 0)

    # ---------------- H. melanoma, threshold-free and fixed operating points
    P("== H. melanoma vs rest ==")
    rows = []
    for nm, y, g, p in (("source", yS, gS, P_S), ("target", yO, gO, P_T)):
        t_ = (y == m).astype(int); s_ = p[:, m]
        auc = roc_auc_score(t_, s_); ap = average_precision_score(t_, s_)
        bA = boot(lambda ii: roc_auc_score(t_[ii], s_[ii]) if 0 < t_[ii].sum() < len(ii) else np.nan, g, a.n_boot, rng)
        bP = boot(lambda ii: average_precision_score(t_[ii], s_[ii]) if t_[ii].sum() > 0 else np.nan, g, a.n_boot, rng)
        rows.append({"set": nm, "prevalence_%": t_.mean() * 100, "ROC_AUC": auc, "AUC_CI": ci(bA, 1),
                     "PR_AUC": ap, "PR_CI": ci(bP, 1)})
        summ[f"melAUC_{nm}"] = auc; summ[f"melPRAUC_{nm}"] = ap
    P(pd.DataFrame(rows).set_index("set").round(3).to_string())
    P("(PR-AUC depends on prevalence; compare it with the prevalence column)")
    # operating points fixed on source validation, per fold, applied per fold, then ensembled by vote of score
    rows = []
    for spec in (0.80, 0.90):
        thr = []
        for k, (V, _, _) in folds.items():
            neg = V.loc[V["y"] != m, f"p_mel"].values
            thr.append(np.quantile(neg, spec))
        th = float(np.mean(thr))
        for nm, y, g, p in (("source", yS, gS, P_S), ("target", yO, gO, P_T)):
            pos, neg = p[y == m, m], p[y != m, m]
            fpr, tpr, _ = roc_curve((y == m).astype(int), p[:, m])
            oracle = float(np.interp(1 - spec, fpr, tpr))
            sens_b = boot(lambda ii: np.mean(p[ii][y[ii] == m, m] >= th), g, a.n_boot, rng)
            rows.append({"spec_target_on_source_val": f"{spec:.0%}", "set": nm, "threshold": th,
                         "sensitivity_%": np.mean(pos >= th) * 100, "sens_CI": ci(sens_b),
                         "specificity_%": np.mean(neg < th) * 100,
                         "ROC_sens_at_this_spec_%(analysis)": oracle * 100})
            summ[f"melSens_spec{int(spec*100)}_{nm}"] = np.mean(pos >= th) * 100
            summ[f"melSpec_spec{int(spec*100)}_{nm}"] = np.mean(neg < th) * 100
    P("operating points chosen on SOURCE validation (threshold on p(mel)), applied unchanged:")
    P(pd.DataFrame(rows).round(3).to_string(index=False))
    P("reading: if ROC-AUC holds but sensitivity at the source-chosen threshold falls, the ranking still works\n"
      "and the operating point moved; if ROC-AUC also falls, the features themselves transfer less well.\n")

    # ---------------- I. directional errors
    P("== I. directional errors, % of the true class (95% lesion-cluster CI) ==")
    other = [j for j in range(K) if j not in (m, n)]

    def rate(y, pr, true_set, pred_set):
        mk = np.isin(y, true_set)
        return np.mean(np.isin(pr[mk], pred_set)) if mk.any() else np.nan
    paths = {"mel->nv": ([m], [n]), "mel->other": ([m], other), "nv->mel": ([n], [m]), "other->nv": (other, [n])}
    rows = []
    for rule, pS, pT in (("raw", P_S, P_T), ("simple", Ps_S, Ps_T)):
        hS, hT = pS.argmax(1), pT.argmax(1)
        for name, (ts, ps) in paths.items():
            vS, vT = rate(yS, hS, ts, ps), rate(yO, hT, ts, ps)
            bS = boot(lambda ii: rate(yS[ii], hS[ii], ts, ps), gS, a.n_boot, rng)
            bT = boot(lambda ii: rate(yO[ii], hT[ii], ts, ps), gO, a.n_boot, rng)
            d = bT - bS
            rows.append({"rule": rule, "path": name, "source_%": vS * 100, "source_CI": ci(bS),
                         "target_%": vT * 100, "target_CI": ci(bT), "target-source_pts": (vT - vS) * 100,
                         "diff_CI": ci(d), "diff_CI_excludes_0": bool(np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0)})
            summ[f"{rule}_{name}_S"] = vS * 100; summ[f"{rule}_{name}_T"] = vT * 100
    P(pd.DataFrame(rows).round(2).to_string(index=False))
    P("(source and target are resampled independently; 'other' = every class except mel and nv)\n")

    # ---------------- J. confidence of the errors
    P("== J. confidence (raw ensemble probabilities) ==")
    rows = []
    for nm, y, p in (("source", yS, P_S), ("target", yO, P_T)):
        h, conf = p.argmax(1), p.max(1)
        err = (y == m) & (h == n); ok = (y == m) & (h == m)
        rows.append({"set": nm, "n_mel->nv": int(err.sum()),
                     "mean_conf_mel->nv": conf[err].mean() if err.any() else np.nan,
                     "share_mel->nv_conf>=0.8_%": np.mean(conf[err] >= 0.8) * 100 if err.any() else np.nan,
                     "mean_conf_correct_mel": conf[ok].mean() if ok.any() else np.nan,
                     "mean_conf_all": conf.mean(), "accuracy": float((h == y).mean()), "ECE": ece(p, y)})
        summ[f"ECE_{nm}"] = ece(p, y)
        summ[f"conf_melnv_{nm}"] = rows[-1]["mean_conf_mel->nv"]
    P(pd.DataFrame(rows).set_index("set").round(3).to_string())
    P("note: ensemble averaging lowers the maximum probability; compare source with target, not with 1.0.")

    txt = "\n".join(L)
    open(f"{a.prefix}extra_results_{exp}.txt", "w").write(txt + "\nDONE\n")
    print(txt + "\n")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=DEFAULT_EXPS)
    ap.add_argument("--runs", default="runs_chromo_cv"); ap.add_argument("--ood", default="runs_ood_pf")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--n_synth", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--prefix", default=""); ap.add_argument("--summary", default="extra_summary.txt")
    a = ap.parse_args()
    S = []
    for e in a.exps:
        if not available(a, e):
            print(f"skip {e} (predictions not complete)"); continue
        S.append(analyse(a, e))
    if S:
        t = pd.DataFrame(S).set_index("exp").T
        open(a.summary, "w").write(t.round(3).to_string() + "\nDONE\n")
        print("== summary ==\n" + t.round(3).to_string())


if __name__ == "__main__":
    main()
