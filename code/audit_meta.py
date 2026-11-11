import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
FWD = {"ResNet-50": "ham_baseline", "EfficientNet-B3": "ham_effb3_baseline",
       "ConvNeXt-T": "ham_convnext_baseline", "Swin-T": "ham_swin_baseline"}


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0].replace("_downsampled", "")


def rate(y, h, ts, ps):
    m = np.isin(y, ts)
    return np.mean(np.isin(h[m], ps)) * 100 if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="isic_metadata.csv"); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--ood", default="runs_ood_pf"); ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    L = []; P = L.append
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    M = pd.read_csv(a.meta, low_memory=False).set_index("isic_id")
    S = pd.read_csv("runs_adapt/sources.csv"); S["isic_id"] = S["image_path"].map(stem)
    S = S.join(M, on="isic_id")
    col = lambda key: next((c for c in M.columns if key in c), None)
    att, conf, dtype, pid, lid = "attribution", col("confirm"), col("dermoscopic_type"), col("patient_id"), col("lesion_id")
    P("== 1. source proxy vs ISIC Archive attribution (SCC excluded; counts) ==")
    P(pd.crosstab(S[att].fillna("no record"), S["source"], margins=True).to_string())
    bcn = S["source"].str.startswith("BCN")
    share = (S.loc[bcn, att] == "Hospital Clínic de Barcelona").mean()
    P(f"\n1024x1024 images attributed to Hospital Clinic de Barcelona: {share:.1%} -> "
      f"{'proxy VALIDATED (>= 90%)' if share >= 0.9 else 'proxy NOT validated'}")
    P("attribution names the contributing institution, which is not necessarily where the image was taken\n")
    P("coverage of further fields by attribution (% non-empty):")
    cov = {c: S.groupby(att)[c].apply(lambda v: v.notna().mean() * 100) for c in (conf, dtype, pid, lid) if c}
    P(pd.DataFrame(cov).round(1).to_string())
    for c in (conf, dtype):
        if c:
            P(f"\n{c} by attribution:"); P(pd.crosstab(S[c].fillna("missing"), S[att].fillna("no record")).to_string())
    P("")

    rng = np.random.default_rng(0)
    for arch, exp in FWD.items():
        folds = []
        for k in range(5):
            r = os.path.join(a.runs, f"{exp}_f{k}")
            T = pd.read_csv(os.path.join(r, "test_predictions.csv"))
            O = pd.read_csv(os.path.join(a.ood, f"{exp}_f{k}", "ood_predictions.csv")).sort_values("image_path").reset_index(drop=True)
            pc = [c for c in O.columns if c.startswith("p_")]
            folds.append((T["y"].values, T[pc].values.argmax(1), T["group"].astype(str).values, O[pc].values.argmax(1)))
        cl = [c[2:] for c in pc]; K = len(cl); m, n = cl.index("mel"), cl.index("nv")
        other = [j for j in range(K) if j not in (m, n)]
        info = S.set_index("image_path").loc[O["image_path"]]
        yT = O["y"].values
        P(f"===== {arch} (per-fold means over the five fold models) =====")
        strata = {"all target images": np.ones(len(yT), bool),
                  "Hospital Clinic de Barcelona": (info[att] == "Hospital Clínic de Barcelona").values,
                  "Anonymous": (info[att] == "Anonymous").values}
        if conf:
            strata["histopathology-confirmed only"] = (info[conf] == "histopathology").values
            strata["Barcelona AND histopathology"] = strata["Hospital Clinic de Barcelona"] & strata["histopathology-confirmed only"]
        if dtype:
            for v in info[dtype].dropna().unique():
                strata[f"dermoscopic_type = {v}"] = (info[dtype] == v).values
        rows = [{"subset": "SOURCE (own test folds)", "n": sum(len(f[0]) for f in folds) // 1,
                 **{c: np.mean([rate(f[0], f[1], [j], [n]) for f in folds]) for j, c in enumerate(cl)}}]
        for name, mk in strata.items():
            if mk.sum() < 100:
                continue
            rows.append({"subset": name, "n": int(mk.sum()),
                         **{c: np.mean([rate(yT[mk], f[3][mk], [j], [n]) for f in folds]) for j, c in enumerate(cl)}})
        P("P(yhat = nv | true class), %:"); P(pd.DataFrame(rows).set_index("subset").round(1).to_string())
        # patient-clustered bootstrap of the matched increases
        lesion = ("lesion_" + O["group"].astype(str)).values
        g = np.where(info[pid].notna().values, info[pid].astype(str).values, lesion) if pid else lesion
        codes, _ = pd.factorize(g); idxT = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
        idxS = []
        for f in folds:
            c2, _ = pd.factorize(f[2]); idxS.append(np.split(np.argsort(c2, kind="stable"), np.cumsum(np.bincount(c2))[:-1]))
        def est(iT, iSs):
            mel, oth = [], []
            for f, iS in zip(folds, iSs):
                mel.append(rate(yT[iT], f[3][iT], [m], [n]) - rate(f[0][iS], f[1][iS], [m], [n]))
                oth.append(rate(yT[iT], f[3][iT], other, [n]) - rate(f[0][iS], f[1][iS], other, [n]))
            return np.mean(mel), np.mean(oth)
        pt = est(np.arange(len(yT)), [np.arange(len(f[0])) for f in folds])
        bs = []
        for _ in range(a.n_boot):
            iT = np.concatenate([idxT[j] for j in rng.integers(0, len(idxT), len(idxT))])
            iSs = [np.concatenate([ix[j] for j in rng.integers(0, len(ix), len(ix))]) for ix in idxS]
            bs.append(est(iT, iSs))
        bs = np.array(bs)
        n_pat = len({x for x in g if not x.startswith("lesion_")}) if pid else 0
        P(f"target clusters: {len(idxT)} ({n_pat} archive patient IDs, the rest lesions)")
        P(f"mel->nv increase {pt[0]:+.1f} [{np.percentile(bs[:, 0], 2.5):+.1f}, {np.percentile(bs[:, 0], 97.5):+.1f}] | "
          f"other->nv increase {pt[1]:+.1f} [{np.percentile(bs[:, 1], 2.5):+.1f}, {np.percentile(bs[:, 1], 97.5):+.1f}]  "
          f"(patient-clustered where possible)\n")
    txt = "\n".join(L)
    open("audit_meta_results.txt", "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
