import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, f1_score

METHODS = ["raw", "bcts", "simple", "em", "bcts_em", "bbse", "bcts_bbse"]


# ------------------------------------------------------------------ calibration and priors
def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_bcts(p, y):
    z = np.log(np.clip(p, 1e-12, 1)); K = z.shape[1]

    def nll(t):
        T = np.exp(t[0]); b = np.r_[0.0, t[1:]]
        q = _softmax(z / T + b)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
    r = minimize(nll, np.zeros(K), method="L-BFGS-B")
    T = float(np.exp(r.x[0])); b = np.r_[0.0, r.x[1:]]
    return lambda pp: _softmax(np.log(np.clip(pp, 1e-12, 1)) / T + b)


def reweight(p, w):
    q = p * w; return q / q.sum(1, keepdims=True)


def em_prior(p, pi_s, iters=500, tol=1e-9):
    pi = pi_s.copy()
    for _ in range(iters):
        new = reweight(p, pi / pi_s).mean(0)
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def bbse_prior(p_val, y_val, p_tgt, K):
    pv, pt = p_val.argmax(1), p_tgt.argmax(1)
    C = np.zeros((K, K))
    np.add.at(C, (pv, y_val), 1.0); C /= len(y_val)        # joint P(yhat=i, y=j) on source
    mu = np.bincount(pt, minlength=K) / len(pt)             # target predicted distribution
    w = np.linalg.lstsq(C, mu, rcond=None)[0]
    w = np.clip(w, 0, None)
    pi_src_true = np.bincount(y_val, minlength=K) / len(y_val)
    pi = w * pi_src_true
    pi = pi / pi.sum() if pi.sum() > 0 else pi_src_true
    Cn = C / C.sum(0, keepdims=True).clip(1e-12)            # P(yhat | y)
    return pi, float(np.linalg.cond(Cn))


class FoldAdapter:
    """Everything that is fitted on fold k's validation predictions."""

    def __init__(self, pv, yv, K):
        self.K = K; self.yv = yv; self.pv = pv
        self.cal = fit_bcts(pv, yv)
        self.qv = self.cal(pv)
        self.pi_p = pv.mean(0); self.pi_q = self.qv.mean(0)

    def apply(self, p):
        q = self.cal(p); out, priors, info = {}, {}, {}
        out["raw"], out["bcts"] = p, q
        out["simple"] = reweight(p, 1 / self.pi_p)
        priors["em"] = em_prior(p, self.pi_p); out["em"] = reweight(p, priors["em"] / self.pi_p)
        priors["bcts_em"] = em_prior(q, self.pi_q); out["bcts_em"] = reweight(q, priors["bcts_em"] / self.pi_q)
        priors["bbse"], info["cond_raw"] = bbse_prior(self.pv, self.yv, p, self.K)
        out["bbse"] = reweight(p, priors["bbse"] / self.pi_p)
        priors["bcts_bbse"], info["cond_bcts"] = bbse_prior(self.qv, self.yv, q, self.K)
        out["bcts_bbse"] = reweight(q, priors["bcts_bbse"] / self.pi_q)
        return out, priors, info


# ------------------------------------------------------------------ metrics
def ba(y, p):
    return balanced_accuracy_score(y, p.argmax(1))


def mf1(y, p, K):
    return f1_score(y, p.argmax(1), average="macro", labels=list(range(K)), zero_division=0)


def cluster_bootstrap(y, preds, groups, ref, K, n_boot, seed=0):
    """Paired bootstrap over lesions: same resampled lesions for all methods."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx_by = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    hard = {m: preds[m].argmax(1) for m in preds}
    obs = {m: balanced_accuracy_score(y, hard[m]) - balanced_accuracy_score(y, hard[ref]) for m in preds}
    obs_f = {m: f1_score(y, hard[m], average="macro", labels=list(range(K)), zero_division=0)
             - f1_score(y, hard[ref], average="macro", labels=list(range(K)), zero_division=0) for m in preds}
    bs = {m: [] for m in preds}; bf = {m: [] for m in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by), len(idx_by))
        ii = np.concatenate([idx_by[j] for j in pick]); yy = y[ii]
        r = balanced_accuracy_score(yy, hard[ref][ii]); rf = f1_score(yy, hard[ref][ii], average="macro", labels=list(range(K)), zero_division=0)
        for m in preds:
            if m == ref:
                continue
            bs[m].append(balanced_accuracy_score(yy, hard[m][ii]) - r)
            bf[m].append(f1_score(yy, hard[m][ii], average="macro", labels=list(range(K)), zero_division=0) - rf)
    rows = []
    for m in preds:
        if m == ref:
            continue
        b, f = np.array(bs[m]), np.array(bf[m])
        rows.append({"method": m, f"dBA_vs_{ref}": obs[m] * 100,
                     "BA_CI95": f"[{np.percentile(b, 2.5)*100:+.2f}, {np.percentile(b, 97.5)*100:+.2f}]",
                     "BA_p": min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())),
                     f"dMacroF1_vs_{ref}": obs_f[m] * 100,
                     "F1_CI95": f"[{np.percentile(f, 2.5)*100:+.2f}, {np.percentile(f, 97.5)*100:+.2f}]"})
    return pd.DataFrame(rows)


def resample_to_prior(df, target_prior, rng):
    """Lesion-level resampling of one test fold to a target class prior (size = len(df))."""
    N = len(df); parts = []
    for k, pk in enumerate(target_prior):
        n_k = int(round(N * pk))
        d = df[df["y"] == k]
        if n_k == 0 or d.empty:
            continue
        g = d.groupby("group").indices; keys = list(g)
        got = []
        while sum(len(x) for x in got) < n_k:
            got.append(g[keys[rng.integers(len(keys))]])
        parts.append(d.index.values[np.concatenate(got)][:n_k])  # positions within d -> df index
    return df.loc[np.concatenate(parts)]


def source_proxy(path):
    if "_downsampled" in path:
        return "MSK-like (downsampled)"
    try:
        with Image.open(path) as im:
            return "BCN-like (1024x1024)" if im.size == (1024, 1024) else "other"
    except Exception:
        return "unreadable"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv"); ap.add_argument("--exp", default="ham_baseline")
    ap.add_argument("--ood", default="runs_ood_pf"); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--n_synth", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="label_shift_results.txt")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    folds = {}
    for k in a.folds:
        r = os.path.join(a.runs, f"{a.exp}_f{k}")
        V = pd.read_csv(os.path.join(r, "val_predictions.csv"))
        T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
        O = pd.read_csv(os.path.join(a.ood, f"{a.exp}_f{k}", "ood_predictions.csv"))
        folds[k] = (V, T, O)
    pc = [c for c in folds[a.folds[0]][0].columns if c.startswith("p_")]; classes = [c[2:] for c in pc]; K = len(classes)
    O0 = folds[a.folds[0]][2].sort_values("image_path").reset_index(drop=True)
    yO, gO = O0["y"].values, O0["group"].astype(str).values
    pi_ood = np.bincount(yO, minlength=K) / len(yO)

    adapters, ood_sum, ind_parts, prior_rows, conds = {}, {m: 0 for m in METHODS}, [], [], []
    for k, (V, T, O) in folds.items():
        ad = FoldAdapter(V[pc].values, V["y"].values, K); adapters[k] = ad
        O = O.sort_values("image_path").reset_index(drop=True)
        assert (O["image_path"].values == O0["image_path"].values).all(), "fold OOD files differ"
        out, pri, info = ad.apply(O[pc].values)
        for m in METHODS:
            ood_sum[m] = ood_sum[m] + out[m]
        for m, v in pri.items():
            prior_rows.append({"fold": k, "method": m, **{c: v[j] for j, c in enumerate(classes)},
                               "MAE": np.abs(v - pi_ood).mean()})
        conds.append(info)
        out_t, _, _ = ad.apply(T[pc].values)
        ind_parts.append((T["y"].values, T["group"].astype(str).values, out_t))
    ood = {m: ood_sum[m] / len(folds) for m in METHODS}
    yI = np.concatenate([x[0] for x in ind_parts]); gI = np.concatenate([x[1] for x in ind_parts])
    ind = {m: np.concatenate([x[2][m] for x in ind_parts]) for m in METHODS}

    L = []; P = L.append
    pd.set_option("display.width", 220)
    P(f"Label-shift study | model {a.exp} | {len(folds)} folds | OOD n={len(yO)} | HAM out-of-fold n={len(yI)}\n")

    P("== A+B. balanced accuracy / macro-F1 / accuracy per method ==")
    rows = []
    for m in METHODS:
        rows.append({"method": m, "HAM_BA": ba(yI, ind[m]), "OOD_BA": ba(yO, ood[m]), "OOD_macroF1": mf1(yO, ood[m], K),
                     "OOD_acc": float((ood[m].argmax(1) == yO).mean())})
    t = pd.DataFrame(rows).set_index("method"); t["drop_BA"] = t["HAM_BA"] - t["OOD_BA"]
    P(t.round(4).to_string())
    gap = t.loc["raw", "HAM_BA"] - t.loc["raw", "OOD_BA"]
    P(f"raw gap {gap*100:.1f} pts; fraction of the raw gap closed on OOD: " +
      ", ".join(f"{m} {(t.loc[m,'OOD_BA']-t.loc['raw','OOD_BA'])/gap*100:.0f}%" for m in METHODS if m != "raw"))
    P("OOD-specific gain = OOD gain minus in-distribution gain of the same rule: " +
      ", ".join(f"{m} {((t.loc[m,'OOD_BA']-t.loc['raw','OOD_BA'])-(t.loc[m,'HAM_BA']-t.loc['raw','HAM_BA']))*100:+.1f}" for m in METHODS if m != "raw") + "\n")

    P("== A. paired lesion-cluster bootstrap on OOD (points) ==")
    P(cluster_bootstrap(yO, ood, gO, "simple", K, a.n_boot, a.seed).round(3).to_string(index=False))
    P(cluster_bootstrap(yO, ood, gO, "raw", K, a.n_boot, a.seed).round(3).to_string(index=False) + "\n")

    P("== C. prior estimation on OOD (mean over folds), true prior in last row ==")
    pr = pd.DataFrame(prior_rows).groupby("method")[classes + ["MAE"]].mean()
    pr.loc["TRUE OOD prior"] = list(pi_ood) + [0.0]
    P((pr * np.r_[np.full(K, 100), 1]).round(3).to_string() + "   (priors in %, MAE as fraction)")
    cr = pd.DataFrame(conds).mean()
    P(f"condition number of BBSE confusion matrix P(yhat|y): raw {cr['cond_raw']:.1f} | bcts {cr['cond_bcts']:.1f}\n")

    P(f"== D. CONTROLLED pure label shift: HAM test folds resampled by lesion to the OOD prior ({a.n_synth} draws/fold) ==")
    srows = []
    for k, (V, T, O) in folds.items():
        for _ in range(a.n_synth):
            S = resample_to_prior(T.reset_index(drop=True), pi_ood, rng)
            out, pri, _ = adapters[k].apply(S[pc].values); yS = S["y"].values
            pi_S = np.bincount(yS, minlength=K) / len(yS)
            r = {"fold": k}
            for m in METHODS:
                r[f"BA_{m}"] = ba(yS, out[m])
            for m, v in pri.items():
                r[f"MAE_{m}"] = np.abs(v - pi_S).mean()
            srows.append(r)
    s = pd.DataFrame(srows).drop(columns="fold").mean()
    tab = pd.DataFrame({"synthetic_BA": [s[f"BA_{m}"] for m in METHODS],
                        "HAM_original_BA": [t.loc[m, "HAM_BA"] for m in METHODS],
                        "real_OOD_BA": [t.loc[m, "OOD_BA"] for m in METHODS]}, index=METHODS)
    P(tab.round(4).to_string())
    mae = pd.DataFrame({"synthetic_prior_MAE": [s.get(f"MAE_{m}", np.nan) for m in ["em", "bcts_em", "bbse", "bcts_bbse"]],
                        "real_OOD_prior_MAE": [pr.loc[m, "MAE"] for m in ["em", "bcts_em", "bbse", "bcts_bbse"]]},
                       index=["em", "bcts_em", "bbse", "bcts_bbse"])
    P(mae.round(4).to_string())
    P("reading: under pure label shift BA(raw) should stay close to HAM_original; if prior estimators are accurate\n"
      "on synthetic shift but poor on real OOD, the real shift is not a pure label shift.\n")

    P("== E. OOD balanced accuracy by source proxy ==")
    src = np.array([source_proxy(p) for p in O0["image_path"]])
    rows = []
    for sname in pd.Series(src).value_counts().index:
        mk = src == sname
        if mk.sum() < 100:
            continue
        rows.append({"source": sname, "n": int(mk.sum()), **{m: ba(yO[mk], ood[m][mk]) for m in METHODS}})
    P(pd.DataFrame(rows).round(4).to_string(index=False))
    P("(source proxy is approximate: ISIC 2019 metadata has no explicit source column)\n")

    # F. conditional confusion shift and its link to the prior-estimation error (labels used for ANALYSIS only)
    P("== F. class-conditional prediction shift: C[i,j] = P(yhat=i | y=j), raw predictions ==")

    def cond_conf(y, p):
        C = np.zeros((K, K)); np.add.at(C, (p.argmax(1), y), 1.0)
        return C / C.sum(0, keepdims=True).clip(1)
    CS, CT = cond_conf(yI, ind["raw"]), cond_conf(yO, ood["raw"])
    lab = [f"y={c}" for c in classes]; pl = [f"pred {c}" for c in classes]
    P("C_S (HAM10000, out-of-fold):"); P(pd.DataFrame(CS, index=pl, columns=lab).round(3).to_string())
    P("C_T (ISIC 2019 non-HAM):"); P(pd.DataFrame(CT, index=pl, columns=lab).round(3).to_string())
    P("C_T - C_S:"); P(pd.DataFrame(CT - CS, index=pl, columns=lab).round(3).to_string())
    P(f"||C_T - C_S||_F = {np.linalg.norm(CT - CS):.3f};  total change of P(yhat=nv|y) over non-nv classes: "
      f"{(CT - CS)[classes.index('nv'), [j for j in range(K) if classes[j] != 'nv']].sum():+.3f}"
      if "nv" in classes else "")
    # BBSE assumes C_T = C_S. With target predicted distribution mu_T = C_T pi_T, its estimate is
    # pi_hat = C_S^{-1} C_T pi_T, so the error decomposes exactly over TRUE classes j:
    #   pi_hat - pi_T = sum_j [C_S^{-1} (C_T - C_S)]_{:, j} * pi_T(j)
    Cs_inv = np.linalg.pinv(CS)
    contrib = (Cs_inv @ (CT - CS)) * pi_ood[None, :]
    P("\nBBSE prior-error decomposition (percentage points): row = estimated class, column = true class whose")
    P("conditional shift causes the error; last column = total error (estimate - truth) without clipping")
    tab = pd.DataFrame(contrib * 100, index=[f"est {c}" for c in classes], columns=lab)
    tab["TOTAL"] = tab.sum(1)
    P(tab.round(2).to_string())
    if "nv" in classes:
        r = tab.loc["est nv"].drop("TOTAL"); r = r.reindex(r.abs().sort_values(ascending=False).index)
        P("largest contributors to the nevus-prior error: " + ", ".join(f"{k} {v:+.1f}" for k, v in r.head(3).items()))
    P("note: this decomposition is an identity for BBSE; it uses target labels for analysis only, never for adaptation.")

    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
