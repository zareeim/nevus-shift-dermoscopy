import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import review6, review7

WIDTH = 6.5          # inches, = text width in the elsarticle preprint layout
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})

_subplots, _figure = plt.subplots, plt.figure


def _scale(kw):
    if "figsize" in kw:
        w, h = kw["figsize"]; kw["figsize"] = (WIDTH, h * WIDTH / w)
    return kw


plt.subplots = lambda *a, **kw: _subplots(*a, **_scale(kw))
plt.figure = lambda *a, **kw: _figure(*a, **_scale(kw))

_text = matplotlib.axes.Axes.text
def text(self, *a, **kw):            # cell labels of the heatmap and annotations: never above 7 pt
    if "fontsize" in kw:
        kw["fontsize"] = min(kw["fontsize"], 7)
    return _text(self, *a, **kw)
matplotlib.axes.Axes.text = text

_title = matplotlib.axes.Axes.set_title
def set_title(self, label, *a, **kw):  # panel titles: letter + short text, 8.5 pt
    if label[:3] in ("a  ", "b  "):
        label = label.split(" (")[0]
    kw["fontsize"] = min(kw.get("fontsize", 8.5), 8.5)
    return _title(self, label, *a, **kw)
matplotlib.axes.Axes.set_title = set_title

_xtl = matplotlib.axes.Axes.set_xticklabels
def set_xticklabels(self, labels, *a, **kw):  # wrap the long condition names of Fig. 2b
    labels = [str(l).replace("pure prior", "pure\nprior").replace("conditional only", "conditional\nonly")
              .replace("real target", "real\ntarget") for l in labels]
    return _xtl(self, labels, *a, **kw)
matplotlib.axes.Axes.set_xticklabels = set_xticklabels

_ylab = matplotlib.axes.Axes.set_ylabel
def set_ylabel(self, label, *a, **kw):
    label = {"contribution to nevus-prior error (points)": "contribution (points)",
             "BBSE nevus-prior error (points)": "nevus-prior error (points)"}.get(label, label)
    return _ylab(self, label, *a, **kw)
matplotlib.axes.Axes.set_ylabel = set_ylabel

_legend = matplotlib.axes.Axes.legend
def legend(self, *a, **kw):
    kw["fontsize"] = min(kw.get("fontsize", 7), 7)
    return _legend(self, *a, **kw)
matplotlib.axes.Axes.legend = legend

if __name__ == "__main__":
    class A:  # same defaults as review6.py
        runs, i20_meta, deltaC, n_boot, out = "runs_chromo_cv", "isic2020_metadata.csv", "review5_deltaC.csv", 0, ""
    L = []
    review6.fig1(A.deltaC, L.append)
    review6.fig2(A, L.append, {"ResNet-50": [0.7, 6.0, 13.6], "EfficientNet-B3": [-0.5, 4.4, 11.4],
                               "ConvNeXt-T": [1.6, 5.6, 13.1], "Swin-T": [1.7, 7.9, 16.0]})
    review7.main()
    print("\n".join(L)); print("DONE: fig1_deltaC, fig2_bbse, fig3_roc_reliability (.pdf and .png)")
