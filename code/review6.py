import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from review5 import ARCH, SEEDS, TARGETS, models, condm, stem
from extra_analysis import cluster_index

TNAME = {"ISIC2019": "ISIC 2019", "Derm7pt": "Derm7pt", "ISIC2020": "ISIC 2020"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_SRC, C_19, C_D7, C_20 = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9c2a29"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def auc(pos, s):
    n1 = pos.sum(); n0 = len(s) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def save(fig, name):
    fig.savefig(name + ".png", bbox_inches="tight"); fig.savefig(name + ".pdf", bbox_inches="tight"); plt.close(fig)


def roc_ci(a, P, B=1000):
    rng = np.random.default_rng(1)
    rows = []
    for tg, ood in TARGETS.items():
        for arch, base in ARCH.items():
            M = models(a.runs, ood, base)
            if M is None:
                P(f"{arch} {tg}: predictions missing"); continue
            pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; m = cl.index("mel")
            mods, gS, yT = [], {}, None
            for i, (V, T, O) in enumerate(M):
                k = i % 5
                th = np.quantile(V[pc].values[V["y"].values != m, m], 0.8)
                mods.append((k, T["y"].values == m, T[pc].values[:, m], O[pc].values[:, m], th))
                gS.setdefault(k, T["group"].astype(str).values)
                if yT is None:
                    yT = O["y"].values == m; gT = O["group"].astype(str).values

            def stat(iS, iT):
                aS, aT, se, sp = [], [], [], []
                for k, yS, sS, sT, th in mods:
                    ys, ss = yS[iS[k]], sS[iS[k]]; yt, st = yT[iT], sT[iT]
                    aS.append(auc(ys, ss)); aT.append(auc(yt, st))
                    se.append(np.mean(st[yt] >= th)); sp.append(np.mean(st[~yt] < th))
                aS, aT = np.nanmean(aS), np.nanmean(aT)
                return np.array([aS, aT, aS - aT, np.mean(se), np.mean(sp)])

            full = {k: np.arange(len(g)) for k, g in gS.items()}
            est = stat(full, np.arange(len(yT)))
            idxT = cluster_index(gT); idxS = {k: cluster_index(g) for k, g in gS.items()}
            bs = []
            for _ in range(B):
                iT = np.concatenate([idxT[j] for j in rng.integers(0, len(idxT), len(idxT))])
                iS = {k: np.concatenate([ix[j] for j in rng.integers(0, len(ix), len(ix))]) for k, ix in idxS.items()}
                bs.append(stat(iS, iT))
            bs = np.array(bs); lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
            names = ["ROC-AUC source", "ROC-AUC target", "ROC-AUC drop", "sens target (80%-val thr)", "spec target (80%-val thr)"]
            for j, nm in enumerate(names):
                rows.append({"target": tg, "arch": arch, "measure": nm,
                             "estimate": round(est[j], 3), "CI_low": round(lo[j], 3), "CI_high": round(hi[j], 3)})
            print(f"CI done {arch} {tg}", flush=True)
    R = pd.DataFrame(rows)
    P("\n######## CI melanoma ROC-AUC and operating point, seed-pooled (15 models), paired cluster bootstrap "
      f"B={B} ########")
    P(R.to_string(index=False))
    return R


def contributors(a, P, B=2000):
    P("\n######## VI ISIC 2020 by contributor, with patient-cluster CIs (mean over 15 models) ########")
    if not os.path.exists(a.i20_meta):
        P(f"missing {a.i20_meta}"); return
    meta = pd.read_csv(a.i20_meta)
    att = dict(zip(meta["isic_id"].astype(str), meta["attribution"].fillna("no record")))
    rng = np.random.default_rng(2); rows = []
    for arch, base in ARCH.items():
        M = models(a.runs, "runs_ood_i20", base)
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]
        m, n, b = cl.index("mel"), cl.index("nv"), cl.index("bkl")
        O0 = M[0][2]; y = O0["y"].values; g = O0["group"].astype(str).values
        grp = np.array([att.get(stem(p), "no record") for p in O0["image_path"]])
        H = np.array([O[pc].values.argmax(1) for _, _, O in M])          # 15 x N predictions
        for c in sorted(set(grp)):
            mk = np.where(grp == c)[0]
            idx = cluster_index(g[mk])
            def rates(ii):
                yy, hh = y[mk][ii], H[:, mk][:, ii]
                out = []
                for t, p in ((m, n), (b, n), (n, m)):
                    s = yy == t
                    out.append(np.mean(hh[:, s] == p) * 100 if s.any() else np.nan)
                return np.array(out)
            est = rates(np.arange(len(mk)))
            bs = np.array([rates(np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])) for _ in range(B)])
            lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
            f = lambda j: "n/a" if np.isnan(est[j]) else f"{est[j]:.1f} [{lo[j]:.1f}, {hi[j]:.1f}]"
            rows.append({"arch": arch, "contributor": c[:45], "images": len(mk), "patients": len(idx),
                         "mel": int((y[mk] == m).sum()), "bkl": int((y[mk] == b).sum()), "nv": int((y[mk] == n).sum()),
                         "mel->nv %": f(0), "bkl->nv %": f(1), "nv->mel %": f(2)})
    pd.set_option("display.width", 260)
    P(pd.DataFrame(rows).to_string(index=False))


def fig1(dc_csv, P):
    D = pd.read_csv(dc_csv); D = D[D["arch"] == "ResNet-50"]
    order = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"width_ratios": [7, 6, 3]})
    for ax, tg in zip(axs, ["ISIC2019", "Derm7pt", "ISIC2020"]):
        d = D[D["target"] == tg]
        present = [c for c in order if (d[d["true"] == c]["target_rate"].sum() > 0)]
        Mx = d.pivot(index="pred", columns="true", values="delta").reindex(index=order, columns=present) * 100
        im = ax.imshow(Mx.values, cmap=DIV, vmin=-60, vmax=60, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                v = Mx.values[i, j]
                if abs(v) >= 5:
                    ax.text(j, i, f"{v:+.0f}".replace("-", "\u2212"), ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) >= 35 else INK)
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order if ax is axs[0] else [])
        ax.set_xlabel("true class"); ax.set_title(TNAME[tg], color=INK, fontsize=10)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        ax.add_patch(plt.Rectangle((-0.5, order.index("nv") - 0.5), len(present), 1, fill=False, ec=INK, lw=1.2))
    axs[0].set_ylabel("predicted class")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("target − source (percentage points)")
    cb.outline.set_visible(False)
    save(fig, "fig1_deltaC")
    P("fig1_deltaC: ResNet-50, P(yhat|y) target - source, cells with |delta| >= 5 labelled; nv row outlined")


def fig2(a, P, tableA):
    rows = []
    for arch, base in ARCH.items():
        M = models(a.runs, "runs_ood_pf", base)
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; cl = [c[2:] for c in pc]; K = len(cl); n = cl.index("nv")
        cons, errs = [], []
        for V, T, O in M:
            Cv = condm(V["y"].values, V[pc].values.argmax(1), K); Ct = condm(O["y"].values, O[pc].values.argmax(1), K)
            pi = np.bincount(O["y"].values, minlength=K) / len(O)
            A = np.linalg.solve(Cv, Ct - Cv)
            cons.append(A[n, :] * pi * 100); errs.append((A @ pi)[n] * 100)
        c = np.mean(cons, 0)
        for j in range(K):
            rows.append({"arch": arch, "class": cl[j], "contribution": c[j]})
        P(f"fig2 {arch}: nevus-prior error from decomposition {np.mean(errs):+.2f} = sum of contributions {c.sum():+.2f}")
    R = pd.DataFrame(rows)
    P(R.pivot(index="class", columns="arch", values="contribution").round(2).to_string())
    archs = list(ARCH); cols = [C_19, C_D7, C_20, "#eda100"]
    order = ["mel", "bkl", "bcc", "akiec", "df", "vasc", "nv"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [7, 3.4]})
    w = 0.8 / len(archs); x = np.arange(len(order))
    for i, (ar, col) in enumerate(zip(archs, cols)):
        v = R[R["arch"] == ar].set_index("class").reindex(order)["contribution"].values
        axs[0].bar(x + (i - 1.5) * w, v, w * 0.9, color=col, label=ar, zorder=3)
    axs[0].axhline(0, color=INK2, lw=0.8); axs[0].set_xticks(x); axs[0].set_xticklabels(order)
    axs[0].set_xlabel("true class whose conditional prediction shift contributes")
    axs[0].set_ylabel("contribution to nevus-prior error (points)")
    axs[0].set_title("a  BBSE error by class, ISIC 2019 (mean over 15 models)", loc="left", fontsize=10, color=INK)
    axs[0].grid(axis="y", color=GRID, zorder=0); axs[0].legend(frameon=False, fontsize=8)
    xs = np.arange(3)
    for i, (ar, col) in enumerate(zip(archs, cols)):
        axs[1].bar(xs + (i - 1.5) * w, tableA[ar], w * 0.9, color=col, zorder=3)
    axs[1].axhline(0, color=INK2, lw=0.8); axs[1].set_xticks(xs)
    axs[1].set_xticklabels(["A\npure prior", "B\nconditional only", "C\nreal target"])
    axs[1].set_ylabel("BBSE nevus-prior error (points)")
    axs[1].set_title("b  Controlled conditions (seed 42)", loc="left", fontsize=10, color=INK)
    axs[1].grid(axis="y", color=GRID, zorder=0)
    fig.tight_layout(); save(fig, "fig2_bbse")


def fig3(a, P, recal):
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    curves = {}
    for tg, ood in TARGETS.items():
        M = models(a.runs, ood, "ham_baseline")
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; m = [c[2:] for c in pc].index("mel")
        for nm, getter in (("source", lambda V, T, O: T), (tg, lambda V, T, O: O)):
            if nm in curves:
                continue
            tprs, pts = [], []
            for V, T, O in M:
                D = getter(V, T, O); y = D["y"].values == m; s = D[pc].values[:, m]
                th = np.quantile(V[pc].values[V["y"].values != m, m], 0.8)
                ts = np.sort(np.unique(s))[::-1]
                tpr = np.array([0] + [np.mean(s[y] >= t) for t in ts[:: max(1, len(ts) // 400)]] + [1])
                fpr = np.array([0] + [np.mean(s[~y] >= t) for t in ts[:: max(1, len(ts) // 400)]] + [1])
                tprs.append(np.interp(grid, fpr, tpr)); pts.append((np.mean(s[~y] >= th), np.mean(s[y] >= th)))
            curves[nm] = (np.mean(tprs, 0), np.mean(pts, 0))
    style = {"source": (C_SRC, "HAM10000 test folds"), "ISIC2019": (C_19, "ISIC 2019"),
             "Derm7pt": (C_D7, "Derm7pt"), "ISIC2020": (C_20, "ISIC 2020")}
    for nm, (tpr, pt) in curves.items():
        col, lab = style[nm]
        ax.plot(grid, tpr, color=col, lw=1.8, label=lab, zorder=3)
        ax.scatter(*pt, s=46, color=col, edgecolor="white", lw=1.5, zorder=4)
        P(f"fig3 {nm}: transferred 80%-val threshold point FPR {pt[0]*100:.1f}, TPR {pt[1]*100:.1f}")
    if "ISIC2019" in curves:
        ax.scatter(1 - recal[1] / 100, recal[0] / 100, s=60, marker="D", facecolor="white", edgecolor=C_19, lw=1.8, zorder=5)
        ax.annotate("recalibrated on 1,000\nlocal images", (1 - recal[1] / 100, recal[0] / 100), xytext=(0.30, 0.42),
                    fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title("Melanoma ROC, ResNet-50 (mean over 15 fold models)\n"
                 "dots: threshold fixed for 80% specificity on source validation", fontsize=9, color=INK, loc="left")
    save(fig, "fig3_roc")


def fig4(a, P):
    e = np.linspace(0, 1, 16)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    done = set()
    for tg, col in (("ISIC2019", C_19), ("ISIC2020", C_20)):
        M = models(a.runs, TARGETS[tg], "ham_baseline")
        if M is None:
            continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]
        for nm, which, cc in (("HAM10000 test folds", "T", C_SRC), (TNAME[tg], "O", col)):
            if nm in done:
                continue
            done.add(nm)
            conf, corr = [], []
            for V, T, O in M:
                D = T if which == "T" else O; p = D[pc].values
                conf.append(p.max(1)); corr.append(p.argmax(1) == D["y"].values)
            conf, corr = np.concatenate(conf), np.concatenate(corr)
            xs, ys = [], []
            for lo, hi in zip(e[:-1], e[1:]):
                mk = (conf > lo) & (conf <= hi)
                if mk.sum() >= 30:
                    xs.append(conf[mk].mean()); ys.append(corr[mk].mean())
            ax.plot(xs, ys, color=cc, lw=1.8, marker="o", ms=5, mec="white", mew=1.2, label=nm, zorder=3)
            P(f"fig4 {nm}: signed (conf - acc) pooled = {(conf.mean() - corr.mean())*100:+.1f} points")
    ax.text(0.62, 0.30, "below the diagonal:\noverconfident", fontsize=8, color=INK2)
    ax.text(0.08, 0.78, "above the diagonal:\nunderconfident", fontsize=8, color=INK2)
    ax.set_xlabel("mean top-label confidence in bin"); ax.set_ylabel("accuracy in bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 0.72))
    ax.set_title("Reliability, ResNet-50 (15 fold models pooled;\nbins with at least 30 images)", fontsize=9, color=INK, loc="left")
    save(fig, "fig4_reliability")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--i20_meta", default="isic2020_metadata.csv")
    ap.add_argument("--deltaC", default="review5_deltaC.csv")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="review6_results.txt")
    a = ap.parse_args()
    L = []; P = L.append
    # Table 5, BBSE column, conditions A/B/C (seed 42)
    tableA = {"ResNet-50": [0.7, 6.0, 13.6], "EfficientNet-B3": [-0.5, 4.4, 11.4],
              "ConvNeXt-T": [1.6, 5.6, 13.1], "Swin-T": [1.7, 7.9, 16.0]}
    for name, fn in (("fig1", lambda: fig1(a.deltaC, P)), ("fig2", lambda: fig2(a, P, tableA)),
                     ("fig3", lambda: fig3(a, P, (61.6, 79.9))), ("fig4", lambda: fig4(a, P)),
                     ("contributors", lambda: contributors(a, P)), ("CI", lambda: roc_ci(a, P, a.n_boot))):
        try:
            fn(); print(f"{name} done", flush=True)
        except Exception as ex:          # one failing part must not stop the others
            P(f"!! {name} failed: {type(ex).__name__}: {ex}"); print(f"!! {name} failed: {ex}", flush=True)
        open(a.out, "w").write("\n".join(L) + "\n")
    open(a.out, "w").write("\n".join(L) + "\nDONE\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
