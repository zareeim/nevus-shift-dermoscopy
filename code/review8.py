import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

from review5 import ARCH, SEEDS, TARGETS, models, condm, stem


def mean15(vals):
    v = np.asarray(vals, float)
    return f"{np.nanmean(v):.1f} [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}]"


def clusters(a, P):
    P("== 1. clusters used by the bootstrap (column 'group' of the prediction files) ==")
    M = models(a.runs, "runs_ood_pf", "ham_baseline", seeds=[""])
    T = pd.concat([t for _, t, _ in M]); g = T["group"].astype(str)
    P(f"HAM10000 (5 test folds): images {len(T)}, clusters {g.nunique()}, images per cluster {len(T)/g.nunique():.2f}, "
      f"largest cluster {g.value_counts().max()}")
    for tg, ood in TARGETS.items():
        M = models(a.runs, ood, "ham_baseline", seeds=[""])
        if M is None:
            continue
        O = M[0][2]; g = O["group"].astype(str)
        same = (g.values == O["image_path"].map(stem).values).mean() * 100
        P(f"{tg}: images {len(O)}, clusters {g.nunique()}, images per cluster {len(O)/g.nunique():.2f}, "
          f"largest cluster {g.value_counts().max()}, clusters that are a single image id {same:.1f}%")
    M = models(a.runs, "runs_ood_pf", "bcn_baseline", seeds=[""])
    if M is not None:
        T = pd.concat([t for _, t, _ in M]); g = T["group"].astype(str)
        P(f"Barcelona subset (reverse source, 5 test folds): images {len(T)}, clusters {g.nunique()}")


def predicted_B(a, P):
    P("\n== 2. decomposition check: predicted condition-B error (A pi_S)[nv] vs observed (Table 5), seed 42 ==")
    obs = {"ResNet-50": 6.0, "EfficientNet-B3": 4.4, "ConvNeXt-T": 5.6, "Swin-T": 7.9}
    obsC = {"ResNet-50": 13.6, "EfficientNet-B3": 11.4, "ConvNeXt-T": 13.1, "Swin-T": 16.0}
    for arch, base in ARCH.items():
        M = models(a.runs, "runs_ood_pf", base, seeds=[""])
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; K = len(pc); n = [c[2:] for c in pc].index("nv")
        pb, pcc = [], []
        for V, T, O in M:
            Cv = condm(V["y"].values, V[pc].values.argmax(1), K); Ct = condm(O["y"].values, O[pc].values.argmax(1), K)
            A = np.linalg.pinv(Cv) @ (Ct - Cv)
            piS = np.bincount(T["y"].values, minlength=K) / len(T); piT = np.bincount(O["y"].values, minlength=K) / len(O)
            pb.append((A @ piS)[n] * 100); pcc.append((A @ piT)[n] * 100)
        P(f"{arch}: B predicted {np.mean(pb):+.1f} vs observed {obs[arch]:+.1f} | C predicted {np.mean(pcc):+.1f} vs observed {obsC[arch]:+.1f}")


def oracle(a, P):
    P("\n== 3. melanoma sensitivity at 80% specificity, threshold set on the WHOLE target (oracle) ==")
    rows = []
    for tg, ood in TARGETS.items():
        for arch, base in ARCH.items():
            M = models(a.runs, ood, base)
            if M is None:
                continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; m = [c[2:] for c in pc].index("mel")
            se, se42, am, amS = [], [], [], []
            for i, (V, T, O) in enumerate(M):
                y = O["y"].values == m; s = O[pc].values[:, m]
                th = np.quantile(s[~y], 0.8); v = np.mean(s[y] >= th) * 100
                se.append(v); (se42.append(v) if i < 5 else None)
                am.append(np.mean(O[pc].values[y].argmax(1) == m) * 100)
                yS = T["y"].values == m; amS.append(np.mean(T[pc].values[yS].argmax(1) == m) * 100)
            rows.append({"target": tg, "arch": arch, "oracle sens @80% spec (15 models)": mean15(se),
                         "oracle sens (seed 42)": f"{np.mean(se42):.1f}",
                         "argmax mel sens source": mean15(amS), "argmax mel sens target": mean15(am)})
    P(pd.DataFrame(rows).to_string(index=False))


def ham_subsource(a, P):
    P("\n== 4. HAM10000 by sub-source (metadata column 'dataset') ==")
    if not a.ham_meta or not os.path.exists(a.ham_meta):
        P("HAM10000 metadata not found (pass --ham_meta)"); return
    hm = pd.read_csv(a.ham_meta)
    if "dataset" not in hm.columns:
        P(f"no 'dataset' column in {a.ham_meta}; columns: {list(hm.columns)}"); return
    src = dict(zip(hm["image_id"].astype(str), hm["dataset"].astype(str)))
    P(pd.crosstab(hm["dataset"], hm["dx"]).to_string())
    rows = []
    for arch, base in list(ARCH.items())[:1] + [("ResNet-50 reverse (Barcelona model on HAM10000)", "bcn_baseline")]:
        rev = base == "bcn_baseline"
        M = models(a.runs, "runs_ood_pf", base, seeds=[""] if rev else SEEDS)
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]
        m, n = cl.index("mel"), cl.index("nv")
        D = [(O if rev else T) for _, T, O in M]
        for s in sorted(set(src.values())):
            r, rn, k = [], [], 0
            for d in D:
                mk = d["image_path"].map(stem).map(src).values == s; y = d["y"].values; h = d[pc].values.argmax(1)
                if (mk & (y == m)).any():
                    r.append(np.mean(h[mk & (y == m)] == n) * 100); k = max(k, int((mk & (y == m)).sum()))
                if (mk & (y == n)).any():
                    rn.append(np.mean(h[mk & (y == n)] == n) * 100)
            rows.append({"model": arch if not rev else "Barcelona model on HAM10000", "sub-source": s,
                         "melanomas (per eval set)": k, "mel->nv %": mean15(r) if r else "n/a",
                         "nv recall %": mean15(rn) if rn else "n/a"})
    P(pd.DataFrame(rows).to_string(index=False))
    P("(forward model: HAM10000 test folds; reverse model: all HAM10000 images)")


def corner_dark(path):
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((128, 128)), float)
    except Exception:
        return np.nan
    c = np.concatenate([g[:12, :12].ravel(), g[:12, -12:].ravel(), g[-12:, :12].ravel(), g[-12:, -12:].ravel()])
    return c.mean()


def vignette(a, P):
    P("\n== 5. dark vignette / border (mean grey level of the four 9% corners < 40 of 255) and the nevus-ward shift ==")
    from label_shift import source_proxy
    for tg, ood in (("ISIC2019", "runs_ood_pf"),):
        rows = []
        Ms = {arch: models(a.runs, ood, base) for arch, base in ARCH.items()}
        O0 = next(M for M in Ms.values() if M is not None)[0][2]
        paths = O0["image_path"].values
        dark = np.array([corner_dark(p) for p in paths]) < 40
        proxy = np.array([source_proxy(p) for p in paths])
        grp = np.where(proxy == "BCN-like (1024x1024)", "Barcelona subset", "anonymous subset")
        P(pd.crosstab(grp, np.where(dark, "dark corners", "no dark corners")).to_string())
        y = O0["y"].values
        for arch, M in Ms.items():
            if M is None:
                continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]
            m, n = cl.index("mel"), cl.index("nv"); other = [j for j in range(len(cl)) if j not in (m, n)]
            for gname in ("Barcelona subset", "anonymous subset", "all"):
                for vname, vm in (("dark corners", dark), ("no dark corners", ~dark)):
                    mk = vm & ((grp == gname) if gname != "all" else True)
                    mel = mk & (y == m); oth = mk & np.isin(y, other)
                    r1 = [np.mean(O[pc].values[mel].argmax(1) == n) * 100 for _, _, O in M] if mel.any() else [np.nan]
                    r2 = [np.mean(O[pc].values[oth].argmax(1) == n) * 100 for _, _, O in M] if oth.any() else [np.nan]
                    rows.append({"arch": arch, "subset": gname, "corners": vname, "melanomas": int(mel.sum()),
                                 "mel->nv %": mean15(r1), "other non-nv": int(oth.sum()), "other->nv %": mean15(r2)})
        P(pd.DataFrame(rows).to_string(index=False))
        hd = [corner_dark(p) for p in pd.concat([t for _, t, _ in models(a.runs, ood, "ham_baseline", seeds=[""])])["image_path"]]
        P(f"HAM10000 images with dark corners: {np.mean(np.array(hd) < 40) * 100:.1f}%")
        P("source rate for reference: mel->nv on HAM10000 test folds about 30-36% (Table 4)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv")
    import glob
    hm = sorted(glob.glob(os.path.expanduser("~/workspace/**/HAM10000_metadata*.csv"), recursive=True))
    ap.add_argument("--ham_meta", default=hm[0] if hm else "")
    ap.add_argument("--out", default="review8_results.txt")
    a = ap.parse_args()
    L = []; P = L.append
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    for name, fn in (("clusters", clusters), ("decomposition", predicted_B), ("oracle", oracle),
                     ("HAM sub-source", ham_subsource), ("vignette", vignette)):
        try:
            fn(a, P); print(f"{name} done", flush=True)
        except Exception as ex:
            P(f"!! {name} failed: {type(ex).__name__}: {ex}"); print(f"!! {name} failed: {ex}", flush=True)
        open(a.out, "w").write("\n".join(L) + "\n")
    open(a.out, "w").write("\n".join(L) + "\nDONE\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
