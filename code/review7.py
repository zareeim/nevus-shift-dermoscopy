import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from review5 import TARGETS, models
from review6 import INK, INK2, GRID, C_SRC, C_19, C_D7, C_20, save  # also applies the shared rcParams

RUNS = "runs_chromo_cv"
STYLE = {"source": (C_SRC, "HAM10000 test folds"), "ISIC2019": (C_19, "ISIC 2019"),
         "Derm7pt": (C_D7, "Derm7pt"), "ISIC2020": (C_20, "ISIC 2020")}


def main():
    L = []
    grid = np.linspace(0, 1, 201); e = np.linspace(0, 1, 16)
    roc, rel, low = {}, {}, {}
    for tg, ood in TARGETS.items():
        M = models(RUNS, ood, "ham_baseline")
        if M is None:
            L.append(f"{tg}: predictions missing"); continue
        pc = [c for c in M[0][0].columns if c.startswith("p_")]; m = [c[2:] for c in pc].index("mel")
        for nm, which in (("source", "T"), (tg, "O")):
            if nm in roc:
                continue
            tprs, pts, lo_share, conf, corr = [], [], [], [], []
            for V, T, O in M:
                D = T if which == "T" else O; p = D[pc].values; y = D["y"].values; s = p[:, m]; pos = y == m
                th = np.quantile(V[pc].values[V["y"].values != m, m], 0.8)
                ts = np.sort(np.unique(s))[::-1][:: max(1, len(np.unique(s)) // 400)]
                tpr = np.r_[0, [np.mean(s[pos] >= t) for t in ts], 1]
                fpr = np.r_[0, [np.mean(s[~pos] >= t) for t in ts], 1]
                tprs.append(np.interp(grid, fpr, tpr)); pts.append((np.mean(s[~pos] >= th), np.mean(s[pos] >= th)))
                lo_share.append(np.mean(s[pos] < np.quantile(s[~pos], 0.2)) * 100)
                conf.append(p.max(1)); corr.append(p.argmax(1) == y)
            roc[nm] = (np.mean(tprs, 0), np.mean(pts, 0))
            low[nm] = (np.mean(lo_share), np.min(lo_share), np.max(lo_share))
            if nm in ("source", "ISIC2019", "ISIC2020"):
                rel[nm] = (np.concatenate(conf), np.concatenate(corr))

    fig = plt.figure(figsize=(11, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3.2, 1], hspace=0.08, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    for nm, (tpr, pt) in roc.items():
        col, lab = STYLE[nm]
        ax.plot(grid, tpr, color=col, lw=1.8, label=lab, zorder=3)
        ax.scatter(*pt, s=40, color=col, edgecolor="white", lw=1.4, zorder=4)
    if "ISIC2019" in roc:
        x, y = 1 - 0.799, 0.616
        ax.scatter(x, y, s=52, marker="D", facecolor="white", edgecolor=C_19, lw=1.6, zorder=5)
        ax.annotate("recalibrated on 1,000\nlocal images", (x, y), xytext=(0.30, 0.40), fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("a", loc="left", fontsize=10, color=INK, fontweight="bold")

    axr = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[1, 1], sharex=axr)
    axr.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    width = e[1] - e[0]; offs = {"source": -width / 3, "ISIC2019": 0, "ISIC2020": width / 3}
    for nm, (conf, corr) in rel.items():
        col, lab = STYLE[nm]; xs, ys = [], []
        share = []
        for lo, hi in zip(e[:-1], e[1:]):
            mk = (conf > lo) & (conf <= hi); share.append(mk.mean() * 100)
            if mk.sum() >= 30:
                xs.append(conf[mk].mean()); ys.append(corr[mk].mean())
        axr.plot(xs, ys, color=col, lw=1.8, marker="o", ms=4.5, mec="white", mew=1.1, label=lab, zorder=3)
        axh.bar(e[:-1] + width / 2 + offs[nm], share, width / 3 * 0.9, color=col, zorder=3)
        L.append(f"{lab}: signed (conf - acc) {(conf.mean() - corr.mean())*100:+.1f}; share of images per bin "
                 + " ".join(f"{s_:.1f}" for s_ in share))
    axr.set_xlim(0, 1); axr.set_ylim(0, 1.01); axr.set_ylabel("accuracy in bin")
    axr.grid(color=GRID, lw=0.6, zorder=0); axr.legend(frameon=False, loc="upper left", fontsize=8)
    axr.text(0.97, 0.05, "below the diagonal = overconfident", ha="right", fontsize=8, color=INK2)
    axr.tick_params(labelbottom=False)
    axr.set_title("b", loc="left", fontsize=10, color=INK, fontweight="bold")
    axh.set_ylabel("% of images"); axh.set_xlabel("mean top-label confidence in bin")
    axh.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    save(fig, "fig3_roc_reliability")

    L.append("\nshare of melanomas with p(mel) below the 20th percentile of the benign images' p(mel), %, "
             "mean [min, max] over 15 ResNet-50 fold models (expected 20 for an uninformative score):")
    for nm, (mu, a, b) in low.items():
        L.append(f"  {STYLE[nm][1]}: {mu:.1f} [{a:.1f}, {b:.1f}]")
    open("review7_results.txt", "w").write("\n".join(L) + "\nDONE\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
