import argparse
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from extra_analysis import boot, ci, cluster_index
from label_shift import reweight

FWD = {"ResNet-50": "ham_baseline", "EfficientNet-B3": "ham_effb3_baseline",
       "ConvNeXt-T": "ham_convnext_baseline", "Swin-T": "ham_swin_baseline"}
TARGETS = {"ISIC2019": "runs_ood_pf", "Derm7pt": "runs_ood_d7"}


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0].replace("_downsampled", "")


def load(a, exp, ood):
    S, O, pis = [], [], []
    for k in a.folds:
        r = os.path.join(a.runs, f"{exp}_f{k}")
        V = pd.read_csv(os.path.join(r, "val_predictions.csv"))
        S.append(pd.read_csv(os.path.join(r, "test_predictions.csv")))
        O.append(pd.read_csv(os.path.join(ood, f"{exp}_f{k}", "ood_predictions.csv")).sort_values("image_path").reset_index(drop=True))
        pc = [c for c in V.columns if c.startswith("p_")]; pis.append(V[pc].values.mean(0))
    S = pd.concat(S, ignore_index=True)
    T = O[0][["image_path", "label", "group", "y"]].copy()
    PT = np.mean([o[pc].values for o in O], 0)
    PTs = np.mean([reweight(o[pc].values, 1 / pi) for o, pi in zip(O, pis)], 0)
    PSs = np.concatenate([reweight(pd.read_csv(os.path.join(a.runs, f"{exp}_f{k}", "test_predictions.csv"))[pc].values, 1 / pi)
                          for k, pi in zip(a.folds, pis)])
    return S, S[pc].values, PSs, T, PT, PTs, [c[2:] for c in pc]


def rate(y, h, ts, ps):
    m = np.isin(y, ts)
    return np.mean(np.isin(h[m], ps)) if m.any() else np.nan


def pval(d):
    d = d[~np.isnan(d)]
    return max(1 / len(d), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16); bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(A, B):
    """pairwise Hamming distances between two uint64 arrays, yields (i, j, d) for d <= thr."""
    Ab = A.astype(">u8").view(np.uint8).reshape(-1, 8); Bb = B.astype(">u8").view(np.uint8).reshape(-1, 8)
    for s in range(0, len(Ab), 256):
        x = np.bitwise_xor(Ab[s:s + 256, None, :], Bb[None, :, :])
        yield s, POP[x].sum(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham_meta", required=True); ap.add_argument("--d7_root", required=True)
    ap.add_argument("--isic_meta", default=None); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup_thr", type=int, default=4); ap.add_argument("--skip_dups", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    L = []; P = L.append
    pd.set_option("display.width", 220)
    P(f"Robustness analyses | {a.n_boot} lesion-cluster bootstraps\n")
    hm = pd.read_csv(a.ham_meta); hm["stem"] = hm["image_id"].astype(str)
    D = {(arch, tg): load(a, exp, od) for arch, exp in FWD.items() for tg, od in TARGETS.items()
         if os.path.exists(os.path.join(od, f"{exp}_f0", "ood_predictions.csv"))}

    # ---------------- P + H
    P("== P. P(yhat=nv | y) per class, raw rule, source -> target (95% CI of the target - source difference) ==")
    tests = []
    for (arch, tg), (S, PS, PSs, T, PT, PTs, cl) in D.items():
        n, m = cl.index("nv"), cl.index("mel"); other = [j for j in range(len(cl)) if j not in (m, n)]
        yS, gS, hS = S["y"].values, S["group"].astype(str).values, PS.argmax(1)
        yT, gT, hT = T["y"].values, T["group"].astype(str).values, PT.argmax(1)
        row = {"model": arch, "target": tg}
        for j, c in enumerate(cl):
            if c == "nv" or not (yT == j).any():
                continue
            bS = boot(lambda ii: rate(yS[ii], hS[ii], [j], [n]), gS, a.n_boot, rng)
            bT = boot(lambda ii: rate(yT[ii], hT[ii], [j], [n]), gT, a.n_boot, rng)
            row[c] = f"{rate(yS, hS, [j], [n]):.2f}->{rate(yT, hT, [j], [n]):.2f} {ci(bT - bS, 1)}"
        P(pd.Series(row).to_string()); P("")
        bS = boot(lambda ii: rate(yS[ii], hS[ii], [m], [n]), gS, a.n_boot, rng)
        bT = boot(lambda ii: rate(yT[ii], hT[ii], [m], [n]), gT, a.n_boot, rng)
        tests.append((arch, tg, "mel->nv increase", (rate(yT, hT, [m], [n]) - rate(yS, hS, [m], [n])) * 100, pval(bT - bS)))
        bS = boot(lambda ii: rate(yS[ii], hS[ii], other, [n]), gS, a.n_boot, rng)
        bT = boot(lambda ii: rate(yT[ii], hT[ii], other, [n]), gT, a.n_boot, rng)
        tests.append((arch, tg, "other->nv increase", (rate(yT, hT, other, [n]) - rate(yS, hS, other, [n])) * 100, pval(bT - bS)))
        hTs = PTs.argmax(1)
        d = boot(lambda ii: balanced_accuracy_score(yT[ii], hTs[ii]) - balanced_accuracy_score(yT[ii], hT[ii]), gT, a.n_boot, rng)
        tests.append((arch, tg, "simple - raw BA", (balanced_accuracy_score(yT, hTs) - balanced_accuracy_score(yT, hT)) * 100, pval(d)))
    H = pd.DataFrame(tests, columns=["model", "target", "test", "effect_pts", "p_boot"])
    H["p_holm"] = holm(H["p_boot"].values); H["significant_holm_0.05"] = H["p_holm"] < 0.05
    P(f"== H. primary hypotheses, Holm-corrected over {len(H)} tests (p floor = 1/{a.n_boot}) ==")
    P(H.round(4).to_string(index=False) + "\n")

    # ---------------- Q. case mix on the source side
    P("== Q. case mix: source behaviour on histopathology-confirmed HAM10000 lesions only (raw rule) ==")
    for arch in FWD:
        if (arch, "ISIC2019") not in D:
            continue
        S, PS, _, T, PT, _, cl = D[(arch, "ISIC2019")]
        n = cl.index("nv")
        dx = S["image_path"].map(stem).map(hm.set_index("stem")["dx_type"])
        rows = []
        for j, c in enumerate(cl):
            for nm, mk in (("HAM all", np.ones(len(S), bool)), ("HAM histo only", (dx == "histo").values)):
                sel = mk & (S["y"].values == j)
                rows.append({"class": c, "set": nm, "n": int(sel.sum()),
                             "P(nv|y)": np.mean(PS[sel].argmax(1) == n) if sel.any() else np.nan})
            sel = T["y"].values == j
            rows.append({"class": c, "set": "ISIC 2019 target", "n": int(sel.sum()),
                         "P(nv|y)": np.mean(PT[sel].argmax(1) == n) if sel.any() else np.nan})
        t = pd.DataFrame(rows).pivot(index="class", columns="set", values=["P(nv|y)", "n"])
        P(f"-- {arch}"); P(t.round(3).to_string()); P("")
    P("dx_type counts in HAM10000:\n" + hm.groupby(["dx", "dx_type"]).size().unstack(fill_value=0).to_string() + "\n")

    # ---------------- R. Derm7pt melanoma subtypes
    meta = pd.read_csv(os.path.join(a.d7_root, "meta", "meta.csv"))
    meta["image_path"] = [os.path.abspath(os.path.join(a.d7_root, "images", p)) for p in meta["derm"]]
    sub = meta.set_index("image_path")["diagnosis"]
    P("== R. Derm7pt melanomas by subtype: % predicted as nevus (raw rule) ==")
    rows = []
    for arch in FWD:
        if (arch, "Derm7pt") not in D:
            continue
        _, _, _, T, PT, _, cl = D[(arch, "Derm7pt")]
        mk = T["y"].values == cl.index("mel"); s = T.loc[mk, "image_path"].map(sub).values
        h = PT[mk].argmax(1) == cl.index("nv")
        for st in sorted(set(s)):
            rows.append({"model": arch, "subtype": st, "n": int((s == st).sum()), "mel->nv_%": h[s == st].mean() * 100})
    P(pd.DataFrame(rows).pivot(index="subtype", columns="model", values="mel->nv_%").round(1).to_string())
    P(pd.DataFrame(rows).groupby("subtype")["n"].first().to_string() + "\n")

    # ---------------- S. true ISIC attribution
    if a.isic_meta and os.path.exists(a.isic_meta):
        im = pd.read_csv(a.isic_meta, low_memory=False)
        idc = "isic_id" if "isic_id" in im else im.columns[0]
        att = next((c for c in im.columns if "attribution" in c.lower()), None)
        conf = next((c for c in im.columns if "confirm" in c.lower()), None)
        im = im.set_index(im[idc].astype(str))
        src = pd.read_csv("runs_adapt/sources.csv"); src["stem"] = src["image_path"].map(stem)
        P(f"== S. ISIC metadata: attribution column = {att}, confirmation column = {conf} ==")
        if att:
            src["attribution"] = src["stem"].map(im[att])
            P("source proxy vs attribution (counts):")
            P(pd.crosstab(src["attribution"].fillna("missing").str[:60], src["source"]).to_string() + "\n")
        for arch in FWD:
            if (arch, "ISIC2019") not in D:
                continue
            _, _, _, T, PT, _, cl = D[(arch, "ISIC2019")]
            st = T["image_path"].map(stem); m, n = cl.index("mel"), cl.index("nv"); h = PT.argmax(1); y = T["y"].values
            if att:
                A = st.map(im[att]).fillna("missing").str[:60].values
                rows = [{"attribution": g, "n_mel": int(((A == g) & (y == m)).sum()),
                         "P(nv|mel)": rate(y[A == g], h[A == g], [m], [n])} for g in pd.unique(A) if ((A == g) & (y == m)).sum() >= 50]
                P(f"-- {arch}: P(nv|mel) by institution (>= 50 melanomas)"); P(pd.DataFrame(rows).round(3).to_string(index=False))
            if conf:
                C = st.map(im[conf]).fillna("missing").astype(str).values; hist = np.char.find(np.char.lower(C.astype(str)), "histo") >= 0
                rows = [{"class": c, "P(nv|y) all": rate(y, h, [j], [n]), "P(nv|y) histo-confirmed": rate(y[hist], h[hist], [j], [n]),
                         "n_histo": int((hist & (y == j)).sum())} for j, c in enumerate(cl) if c != "nv"]
                P(f"-- {arch}: target restricted to histopathology-confirmed images"); P(pd.DataFrame(rows).round(3).to_string(index=False))
            P("")
    else:
        P("== S. skipped (no --isic_meta) ==\n")

    # ---------------- T. near duplicates
    if not a.skip_dups:
        sets = {"HAM10000": pd.read_csv("ham.csv")["image_path"].tolist(),
                "ISIC2019 non-HAM": pd.read_csv("isic19_noham.csv")["image_path"].tolist(),
                "Derm7pt": pd.read_csv("derm7pt.csv")["image_path"].tolist()}
        Hs = {k: np.array([dhash(p) for p in v], dtype=np.uint64) for k, v in sets.items()}
        pairs = []
        for x, y_ in (("HAM10000", "ISIC2019 non-HAM"), ("HAM10000", "Derm7pt"), ("ISIC2019 non-HAM", "Derm7pt")):
            for s, dist in hamming(Hs[x], Hs[y_]):
                ii, jj = np.nonzero(dist <= a.dup_thr)
                pairs += [(x, sets[x][s + i], y_, sets[y_][j], int(dist[i, j])) for i, j in zip(ii, jj)]
        Dp = pd.DataFrame(pairs, columns=["set_a", "image_a", "set_b", "image_b", "hamming"])
        Dp.to_csv("dup_pairs.csv", index=False)
        P(f"== T. near-duplicate pairs (difference hash, Hamming <= {a.dup_thr} of 64) ==")
        P((Dp.groupby(["set_a", "set_b"]).size().to_string() if len(Dp) else "none found") + "\n")
        if len(Dp):
            Dp = Dp.sort_values("hamming").head(12); grid = Image.new("RGB", (400, 200 * len(Dp)))
            for i, r in enumerate(Dp.itertuples()):
                for j, p in enumerate((r.image_a, r.image_b)):
                    grid.paste(Image.open(p).convert("RGB").resize((200, 200)), (200 * j, 200 * i))
            grid.save("dup_examples.jpg", quality=85)
            P("closest pairs shown in dup_examples.jpg (left: first set, right: second set) - inspect by eye\n")

    # ---------------- U. scale control without failed masks
    if os.path.exists("scale_fractions.csv") and os.path.exists("runs_scale_ood/ham_baseline_scale_f0/ood_predictions.csv"):
        F = pd.read_csv("scale_fractions.csv").set_index("image_path")["frac"]
        P("== U. lesion-scale control excluding suspicious masks (standard-view fraction < 0.01 or > 0.95) ==")
        for arch, exp in (("ResNet-50", "ham_baseline"), ("Swin-T", "ham_swin_baseline")):
            out = {}
            for tag, runs, ood, e in (("standard", a.runs, "runs_ood_pf", exp), ("scale", "runs_scale", "runs_scale_ood", exp + "_scale")):
                a2 = argparse.Namespace(**{**vars(a), "runs": runs})
                S, PS, _, T, PT, _, cl = load(a2, e, ood)
                m, n = cl.index("mel"), cl.index("nv")
                fS, fT = S["image_path"].map(F).values, T["image_path"].map(F).values
                okS, okT = (fS >= 0.01) & (fS <= 0.95), (fT >= 0.01) & (fT <= 0.95)
                out[tag] = (rate(T["y"].values[okT], PT[okT].argmax(1), [m], [n]) - rate(S["y"].values[okS], PS[okS].argmax(1), [m], [n])) * 100
                out[f"excluded_{tag}"] = f"source {int((~okS).sum())}, target {int((~okT).sum())}"
            P(f"-- {arch}: mel->nv increase standard {out['standard']:+.1f} | scale-normalised {out['scale']:+.1f} "
              f"(reduction {100 * (1 - out['scale'] / out['standard']):.0f}%) | excluded {out['excluded_scale']}")
        P("")

    txt = "\n".join(L)
    open("robustness_results.txt", "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
