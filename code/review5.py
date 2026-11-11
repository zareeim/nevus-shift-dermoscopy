import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ARCH = {"ResNet-50": "ham_baseline", "EfficientNet-B3": "ham_effb3_baseline",
        "ConvNeXt-T": "ham_convnext_baseline", "Swin-T": "ham_swin_baseline"}
SEEDS = ["", "_s43", "_s44"]
TARGETS = {"ISIC2019": "runs_ood_pf", "Derm7pt": "runs_ood_d7", "ISIC2020": "runs_ood_i20"}


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0].replace("_downsampled", "")


def load(runs, ood, exp, k):
    r = os.path.join(runs, f"{exp}_f{k}")
    V = pd.read_csv(os.path.join(r, "val_predictions.csv"))
    T = pd.read_csv(os.path.join(r, "test_predictions.csv")).sort_values("image_path").reset_index(drop=True)
    f = os.path.join(ood, f"{exp}_f{k}", "ood_predictions.csv")
    O = pd.read_csv(f).sort_values("image_path").reset_index(drop=True) if os.path.exists(f) else None
    return V, T, O


def models(runs, ood, base, seeds=SEEDS):
    out = []
    for s in seeds:
        for k in range(5):
            V, T, O = load(runs, ood, base + s, k)
            if O is None:
                return None
            out.append((V, T, O))
    return out


def recall_ba(y, h, classes):
    return np.mean([np.mean(h[y == c] == c) for c in classes if (y == c).any()]) * 100


def ece(conf, correct, bins=15):
    e = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.any():
            tot += mk.mean() * abs(conf[mk].mean() - correct[mk].mean())
    return tot


def summ(vals):
    v = np.asarray(vals, float)
    return f"{np.nanmean(v):.1f} [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}]"


def condm(y, h, K):
    C = np.zeros((K, K)); np.add.at(C, (h, y), 1.0)
    return C / C.sum(0, keepdims=True).clip(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--ham_meta", required=True); ap.add_argument("--gt19", required=True)
    ap.add_argument("--isic19_meta", default="isic_metadata.csv")
    ap.add_argument("--i20_csv", default="isic2020_target.csv"); ap.add_argument("--i20_meta", default="isic2020_metadata.csv")
    ap.add_argument("--out", default="review5_results.txt")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    L = []; P = L.append
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    rel_rows, dC_rows = [], []

    for tg, ood in TARGETS.items():
        P(f"\n######## TARGET {tg} ########")
        A1, A2, A6o, A7 = [], [], [], []
        for arch, base in ARCH.items():
            M = models(a.runs, ood, base)
            if M is None:
                P(f"{arch}: predictions missing, skipped"); continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl)
            m, n = cl.index("mel"), cl.index("nv")
            r1, r2, r6, r7 = {}, {}, {}, {}
            CS, CT = [], []
            for V, T, O in M:
                yS, pS, yT, pT, yV, pV = T["y"].values, T[pc].values, O["y"].values, O[pc].values, V["y"].values, V[pc].values
                hS, hT = pS.argmax(1), pT.argmax(1)
                present = sorted(set(yT))
                # A2
                r2.setdefault("source BA, 7 classes", []).append(recall_ba(yS, hS, range(K)))
                r2.setdefault("source BA, target classes", []).append(recall_ba(yS, hS, present))
                r2.setdefault("target BA", []).append(recall_ba(yT, hT, present))
                r2.setdefault("drop (same classes)", []).append(recall_ba(yS, hS, present) - recall_ba(yT, hT, present))
                # A1
                for nm, y, p in (("source", yS, pS), ("target", yT, pT)):
                    r1.setdefault(f"ROC-AUC {nm}", []).append(roc_auc_score(y == m, p[:, m]) * 100)
                for spec in (0.8, 0.9):
                    th = np.quantile(pV[yV != m, m], spec)
                    for nm, y, p in (("source", yS, pS), ("target", yT, pT)):
                        r1.setdefault(f"{int(spec*100)}%-val threshold: sens {nm}", []).append(np.mean(p[y == m, m] >= th) * 100)
                        r1.setdefault(f"{int(spec*100)}%-val threshold: spec {nm}", []).append(np.mean(p[y != m, m] < th) * 100)
                # A6
                CS.append(condm(yS, hS, K)); CT.append(condm(yT, hT, K))
                other = [j for j in present if j not in (m, n)]
                oth_noak = [j for j in other if cl[j] != "akiec"]
                for nm, y, h in (("source", yS, hS), ("target", yT, hT)):
                    mk = np.isin(y, other); r6.setdefault(f"other->nv pooled {nm}", []).append(np.mean(h[mk] == n) * 100)
                    mk = np.isin(y, oth_noak); r6.setdefault(f"other->nv pooled, no akiec {nm}", []).append(np.mean(h[mk] == n) * 100)
                    r6.setdefault(f"other->nv macro {nm}", []).append(np.mean([np.mean(h[y == j] == n) for j in other if (y == j).any()]) * 100)
                    r6.setdefault(f"nv->mel {nm}", []).append(np.mean(h[y == n] == m) * 100)
                    r6.setdefault(f"nv->any other class {nm}", []).append(np.mean(h[y == n] != n) * 100)
                # A7
                for nm, y, p in (("source", yS, pS), ("target", yT, pT)):
                    conf, h = p.max(1), p.argmax(1); corr = (h == y).astype(float)
                    r7.setdefault(f"ECE top-label {nm}", []).append(ece(conf, corr) * 100)
                    r7.setdefault(f"signed (conf - acc) {nm}", []).append((conf.mean() - corr.mean()) * 100)
                    r7.setdefault(f"ECE melanoma one-vs-rest {nm}", []).append(ece(p[:, m], (y == m).astype(float)) * 100)
                    pn = h == n
                    if pn.any() and 0 < (y[pn] != n).sum() < pn.sum():
                        r7.setdefault(f"AUROC error detection | pred nv (1-p_nv) {nm}", []).append(roc_auc_score(y[pn] != n, 1 - p[pn, n]) * 100)
                    sel = pn & np.isin(y, [n, m])
                    if (y[sel] == m).any() and (y[sel] == n).any():
                        r7.setdefault(f"AUROC mel vs true nv | pred nv {nm}", []).append(roc_auc_score(y[sel] == m, 1 - p[sel, n]) * 100)
                        r7.setdefault(f"median p_nv: mel predicted nv {nm}", []).append(np.median(p[sel & (y == m), n]) * 100)
                        r7.setdefault(f"median p_nv: true nv predicted nv {nm}", []).append(np.median(p[sel & (y == n), n]) * 100)
                    if tg == "ISIC2019":
                        e = np.linspace(0, 1, 16)
                        for lo, hi in zip(e[:-1], e[1:]):
                            mk = (conf > lo) & (conf <= hi)
                            if mk.any():
                                rel_rows.append({"arch": arch, "set": nm, "bin_lo": lo, "conf": conf[mk].mean(),
                                                 "acc": corr[mk].mean(), "n": int(mk.sum())})
            A1.append({"arch": arch, **{k_: summ(v) for k_, v in r1.items()}})
            A2.append({"arch": arch, **{k_: summ(v) for k_, v in r2.items()}})
            A6o.append({"arch": arch, **{k_: summ(v) for k_, v in r6.items()}})
            A7.append({"arch": arch, **{k_: summ(v) for k_, v in r7.items()}})
            dC = np.mean(CT, 0) - np.mean(CS, 0)
            P(f"\n-- A6 {arch} | {tg}: P(yhat=row | y=col), mean over 15 models; SOURCE, TARGET, TARGET-SOURCE (points)")
            for nmx, Mx in (("source", np.mean(CS, 0)), ("target", np.mean(CT, 0)), ("target - source", dC)):
                P(f"{nmx}:"); P(pd.DataFrame(Mx * 100, index=[f"pred {c}" for c in cl], columns=cl).round(1).to_string())
            for i in range(K):
                for j in range(K):
                    dC_rows.append({"target": tg, "arch": arch, "pred": cl[i], "true": cl[j],
                                    "source": np.mean(CS, 0)[i, j], "target_rate": np.mean(CT, 0)[i, j], "delta": dC[i, j]})
        for name, R in (("A1 melanoma detection, per fold model (no ensemble)", A1),
                        ("A2 balanced accuracy over the target's classes", A2),
                        ("A6 nevus-ward rates, akiec sensitivity, macro average, nevus errors", A6o),
                        ("A7 calibration and error detection per fold model (points / %)", A7)):
            if R:
                P(f"\n== {name} | {tg} =="); P(pd.DataFrame(R).set_index("arch").T.to_string())

    # reverse model on HAM10000
    P("\n######## REVERSE: bcn_baseline -> HAM10000 ########")
    M = models(a.runs, "runs_ood_pf", "bcn_baseline", seeds=[""])
    if M is not None:
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl)
        CS = np.mean([condm(T["y"].values, T[pc].values.argmax(1), K) for _, T, _ in M], 0)
        CT = np.mean([condm(O["y"].values, O[pc].values.argmax(1), K) for _, _, O in M], 0)
        P("per-class recall (%), source test folds vs HAM10000, mean over 5 fold models:")
        P(pd.DataFrame({"source": np.diag(CS) * 100, "HAM10000": np.diag(CT) * 100,
                        "change": (np.diag(CT) - np.diag(CS)) * 100}, index=cl).round(1).to_string())
        P("TARGET - SOURCE, P(yhat=row | y=col), points:")
        P(pd.DataFrame((CT - CS) * 100, index=[f"pred {c}" for c in cl], columns=cl).round(1).to_string())
        for i in range(K):
            for j in range(K):
                dC_rows.append({"target": "HAM10000 (reverse)", "arch": "ResNet-50 (Barcelona)", "pred": cl[i], "true": cl[j],
                                "source": CS[i, j], "target_rate": CT[i, j], "delta": CT[i, j] - CS[i, j]})
    else:
        P("reverse predictions not found in runs_ood_pf/bcn_baseline_f*")

    # A3 ISIC 2020 by contributor
    P("\n######## A3 ISIC 2020 by ISIC Archive attribution ########")
    if os.path.exists(a.i20_csv) and os.path.exists(a.i20_meta):
        tg_csv = pd.read_csv(a.i20_csv); meta = pd.read_csv(a.i20_meta)
        att = dict(zip(meta["isic_id"].astype(str), meta["attribution"].fillna("no record")))
        rows = []
        for arch, base in ARCH.items():
            M = models(a.runs, "runs_ood_i20", base)
            if M is None:
                continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]
            m, n = cl.index("mel"), cl.index("nv")
            grp = np.array([att.get(stem(p), "no record") for p in M[0][2]["image_path"]])
            for g in sorted(set(grp)):
                mk = grp == g; yT = M[0][2]["y"].values
                nmel = int((yT[mk] == m).sum())
                vals = [np.mean(O[pc].values[mk & (yT == m)].argmax(1) == n) * 100 for _, _, O in M] if nmel else [np.nan]
                src = [np.mean(T[pc].values[T["y"].values == m].argmax(1) == n) * 100 for _, T, _ in M]
                rows.append({"arch": arch, "contributor": g, "images": int(mk.sum()), "melanomas": nmel,
                             "mel->nv target %": summ(vals), "mel->nv source %": summ(src),
                             "increase": f"{np.nanmean(vals) - np.mean(src):+.1f}" if nmel else "n/a"})
        P(pd.DataFrame(rows).to_string(index=False))
        P("contributors with < 20 melanomas are imprecise")
    else:
        P(f"missing {a.i20_csv} or {a.i20_meta}")

    # A4 taxonomy of HAM 'akiec' in ISIC 2019
    P("\n######## A4 HAM10000 'akiec' images in the ISIC 2019 ground truth ########")
    hm = pd.read_csv(a.ham_meta); gt = pd.read_csv(a.gt19)
    gt["image"] = gt["image"].astype(str).str.replace("_downsampled", "")
    lab = gt.set_index("image").drop(columns=[c for c in ("UNK",) if c in gt.columns]).idxmax(axis=1)
    hm["isic19_label"] = hm["image_id"].map(lab)
    P(pd.crosstab(hm["dx"], hm["isic19_label"].fillna("not in ISIC 2019")).to_string())
    ak = hm[hm["dx"] == "akiec"]
    P(f"\nHAM akiec images: {len(ak)}; labelled AK in ISIC 2019: {(ak['isic19_label'] == 'AK').sum()}, "
      f"SCC: {(ak['isic19_label'] == 'SCC').sum()}")
    P("non-HAM ISIC 2019 'akiec' in this study = AK only (SCC excluded)")

    # T5 counts
    P("\n######## T5 image counts behind Table 5 ########")
    P("HAM10000, all vs histopathology-confirmed:")
    P(pd.DataFrame({"all": hm["dx"].value_counts(), "histo": hm.loc[hm["dx_type"] == "histo", "dx"].value_counts()}).fillna(0).astype(int).to_string())
    if os.path.exists(a.isic19_meta):
        im = pd.read_csv(a.isic19_meta, low_memory=False)
        conf = next((c for c in im.columns if "confirm" in c), None)
        t19 = pd.read_csv("isic19_noham.csv"); t19["isic_id"] = t19["image_path"].map(stem)
        t19 = t19.merge(im[["isic_id", conf]], on="isic_id", how="left")
        P("ISIC 2019 non-HAM, all vs histopathology-confirmed:")
        P(pd.DataFrame({"all": t19["label"].value_counts(), "histo": t19.loc[t19[conf] == "histopathology", "label"].value_counts()}).fillna(0).astype(int).to_string())

    # LC local recalibration
    P("\n######## LC local recalibration of the melanoma threshold (ISIC 2019, seed-42 fold models) ########")
    rows = []
    for arch, base in ARCH.items():
        M = models(a.runs, "runs_ood_pf", base, seeds=[""])
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; m = cl.index("mel")
        yT = M[0][2]["y"].values; gT = M[0][2]["group"].astype(str).values
        codes, _ = pd.factorize(gT); order = np.argsort(codes, kind="stable")
        idx = np.split(order, np.cumsum(np.bincount(codes))[:-1])
        for N in (100, 200, 500, 1000):
            se, sp, nm_ = [], [], []
            for _ in range(200):
                perm = rng.permutation(len(idx)); sizes = np.cumsum([len(idx[j]) for j in perm])
                last = int(np.searchsorted(sizes, N)) + 1
                cal = np.zeros(len(yT), bool); cal[np.concatenate([idx[j] for j in perm[:last]])] = True
                for _, _, O in M:
                    s = O[pc].values[:, m]
                    neg = s[cal & (yT != m)]
                    if len(neg) == 0 or not (cal & (yT == m)).any():
                        continue
                    th = np.quantile(neg, 0.8); ev = ~cal
                    se.append(np.mean(s[ev & (yT == m)] >= th) * 100); sp.append(np.mean(s[ev & (yT != m)] < th) * 100)
                nm_.append((cal & (yT == m)).sum())
            rows.append({"arch": arch, "local labelled images": N, "of which melanomas (median)": int(np.median(nm_)),
                         "target sensitivity %": f"{np.mean(se):.1f} (SD {np.std(se):.1f})",
                         "target specificity %": f"{np.mean(sp):.1f} (SD {np.std(sp):.1f})"})
    P(pd.DataFrame(rows).to_string(index=False))
    P("reference: compare with the source-validation threshold in section A1 (80%-val threshold rows, ISIC2019)")

    pd.DataFrame(rel_rows).to_csv("review5_reliability.csv", index=False)
    pd.DataFrame(dC_rows).to_csv("review5_deltaC.csv", index=False)
    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
