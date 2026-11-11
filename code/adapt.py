import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

SRC_MIN = 300


def sources(paths):
    from label_shift import source_proxy
    s = pd.Series([source_proxy(p) for p in paths])
    small = s.value_counts()[lambda v: v < SRC_MIN].index
    return s.where(~s.isin(small), "other").values


# ------------------------------------------------------------------ step 1
def run(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from chromo import ChromoNet
    from ood_eval import predict
    from train import ImageDataset, build_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df0 = pd.read_csv(a.data)
    src_cache = os.path.join(a.out, "sources.csv")
    os.makedirs(a.out, exist_ok=True)

    def bns(m):
        return [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]

    def adabn(model, dl):
        m = copy.deepcopy(model).eval()
        for bn in bns(m):
            bn.reset_running_stats(); bn.momentum = None; bn.train()   # cumulative average
        with torch.no_grad():
            for x, _, _ in dl:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    m(x.to(device))
        return m.eval()

    def tent(model, dl):
        m = copy.deepcopy(model).eval()
        params = []
        for bn in bns(m):
            bn.train(); bn.track_running_stats = False; bn.running_mean = None; bn.running_var = None
            params += [bn.weight, bn.bias]
        for p in m.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=a.tent_lr)
        for x, _, _ in dl:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = m(x.to(device)).float()
            p = F.softmax(logits, 1)
            loss = -(p * F.log_softmax(logits, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return m  # BatchNorm keeps using batch statistics at prediction time

    for k in a.folds:
        run_dir = os.path.join(a.runs, f"{a.exp}_f{k}")
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
        df = df0[df0["label"].isin(classes)].reset_index(drop=True)
        if os.path.exists(src_cache):
            df["source"] = pd.read_csv(src_cache).set_index("image_path").loc[df["image_path"], "source"].values
        else:
            df["source"] = sources(df["image_path"]); df[["image_path", "source"]].to_csv(src_cache, index=False)
        _, tf = build_transforms(cfg["img_size"], color_constancy=True)
        base = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
        base.load_state_dict(torch.load(os.path.join(run_dir, "last.pt"), map_location=device))
        base.eval()
        for meth in a.methods:
            out = os.path.join(a.out, f"{meth}_f{k}")
            if os.path.exists(os.path.join(out, "ood_predictions.csv")):
                print(f"skip {out}"); continue
            parts = []
            for s, d in df.groupby("source"):
                d = d.reset_index(drop=True)
                ds = ImageDataset(d, c2i, tf)
                fit_dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
                pred_dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)
                m = adabn(base, fit_dl) if meth == "adabn" else tent(base, fit_dl)
                p = predict(m, pred_dl, device, tta=False)
                o = d[["image_path", "label", "group", "source"]].copy(); o["y"] = d["label"].map(c2i).values
                for j, c in enumerate(classes):
                    o[f"p_{c}"] = p[:, j]
                parts.append(o)
                print(f"fold {k} {meth} source={s} n={len(d)} done")
                del m; torch.cuda.empty_cache()
            os.makedirs(out, exist_ok=True)
            pd.concat(parts).to_csv(os.path.join(out, "ood_predictions.csv"), index=False)


# ------------------------------------------------------------------ step 2
def analyze(a):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from label_shift import cluster_bootstrap, reweight

    def load(path):
        d = pd.read_csv(path).sort_values("image_path").reset_index(drop=True); return d
    variants, ref_paths = {}, None
    for k in a.folds:
        V = pd.read_csv(os.path.join(a.runs, f"{a.exp}_f{k}", "val_predictions.csv"))
        pc = [c for c in V.columns if c.startswith("p_")]; pi = V[pc].values.mean(0)
        sets = {"none": load(os.path.join(a.ood_pf, f"{a.exp}_f{k}", "ood_predictions.csv"))}
        for meth in a.methods:
            f = os.path.join(a.out, f"{meth}_f{k}", "ood_predictions.csv")
            if os.path.exists(f):
                sets[meth] = load(f)
        for meth, d in sets.items():
            if ref_paths is None:
                ref_paths = d["image_path"].values; y = d["y"].values; g = d["group"].astype(str).values
            assert (d["image_path"].values == ref_paths).all(), f"image order differs ({meth})"
            p = d[pc].values
            for rule, q in (("raw", p), ("simple", reweight(p, 1 / pi))):
                key = f"{meth}+{rule}"
                variants.setdefault(key, []).append(q)
    classes = [c[2:] for c in pc]; K = len(classes)
    preds = {k: np.mean(v, 0) for k, v in variants.items() if len(v) == len(a.folds)}
    src = pd.read_csv(os.path.join(a.out, "sources.csv")).set_index("image_path").loc[ref_paths, "source"].values

    L = []; P = L.append
    pd.set_option("display.width", 220)
    P(f"Unsupervised appearance adaptation per source | {len(a.folds)}-fold ensemble | OOD n={len(y)}\n")
    rows = []
    for k, p in preds.items():
        r = {"variant": k, "BA": balanced_accuracy_score(y, p.argmax(1)),
             "macroF1": f1_score(y, p.argmax(1), average="macro", labels=list(range(K)), zero_division=0),
             "acc": float((p.argmax(1) == y).mean())}
        for s in pd.unique(src):
            mk = src == s; r[f"BA[{s}]"] = balanced_accuracy_score(y[mk], p[mk].argmax(1))
        rows.append(r)
    P(pd.DataFrame(rows).set_index("variant").round(4).to_string())
    for ref in ("none+simple", "none+raw"):
        if ref in preds:
            P(f"\npaired lesion-cluster bootstrap vs {ref} (points):")
            P(cluster_bootstrap(y, preds, g, ref, K, a.n_boot).round(3).to_string(index=False))
    if "adabn+simple" in preds:
        b = cluster_bootstrap(y, {"adabn+simple": preds["adabn+simple"], "none+simple": preds["none+simple"]},
                              g, "none+simple", K, a.n_boot)
        lo = float(b["BA_CI95"].iloc[0].strip("[]").split(",")[0])
        P(f"\nPRIMARY (adabn+simple minus none+simple, BA): {b.iloc[0, 1]:+.2f} pts, CI {b['BA_CI95'].iloc[0]} -> "
          f"{'MET' if lo > 0 else 'NOT MET'}")
    P("\nnote: adaptation is transductive (each source adapts on its own unlabeled images, then predicts them).")
    txt = "\n".join(L)
    open(a.results, "w").write(txt + "\nDONE\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--data", default="isic19_noham.csv"); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--exp", default="ham_baseline"); ap.add_argument("--ood_pf", default="runs_ood_pf")
    ap.add_argument("--out", default="runs_adapt"); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=["adabn", "tent"])
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tent_lr", type=float, default=1e-4); ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--results", default="adapt_results.txt")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)


if __name__ == "__main__":
    main()
