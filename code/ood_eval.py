import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from chromo import ChromoNet
from metrics import full_report, save_json
from summarize import paired_bootstrap, pooled_predictions
from train import ImageDataset, build_transforms


@torch.no_grad()
def predict(model, dl, device, tta):
    model.eval(); P = []
    flips = [None, [-1], [-2], [-1, -2]] if tta else [None]
    for x, _, _ in dl:
        x = x.to(device); p = 0
        for fl in flips:
            xx = x if fl is None else torch.flip(x, fl)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                p = p + torch.softmax(model(xx).float(), 1)
        P.append((p / len(flips)).cpu())
    return torch.cat(P).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--runs", default="runs_aug")
    ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out", default="runs_ood")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--primary", nargs=2, default=["ham_chromo", "ham_galdran"])
    ap.add_argument("--results", default="ood_results.txt")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg0 = json.load(open(os.path.join(a.runs, f"{a.exps[0]}_f{a.folds[0]}", "config.json")))
    classes = cfg0["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    df = pd.read_csv(a.data)
    dropped = df[~df["label"].isin(classes)]["label"].value_counts().to_dict()
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    print(f"OOD images: {len(df)} (dropped labels not in training classes: {dropped})")
    _, tf = build_transforms(cfg0["img_size"], color_constancy=True)  # same preprocessing as training
    dl = DataLoader(ImageDataset(df, c2i, tf), batch_size=a.bs, num_workers=4)

    for e in a.exps:
        d = os.path.join(a.out, e)
        if os.path.exists(os.path.join(d, "test_metrics.json")):
            print(f"skip {e}"); continue
        probs = 0
        for k in a.folds:
            run = os.path.join(a.runs, f"{e}_f{k}")
            cfg = json.load(open(os.path.join(run, "config.json")))
            m = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
            m.load_state_dict(torch.load(os.path.join(run, "last.pt"), map_location=device))
            probs = probs + predict(m, dl, device, a.tta)
            print(f"  {e} fold {k} done")
        probs = probs / len(a.folds)
        os.makedirs(d, exist_ok=True)
        o = df[["image_path", "label", "group"]].copy(); o["y"] = df["label"].map(c2i).values
        for j, c in enumerate(classes):
            o[f"p_{c}"] = probs[:, j]
        o.to_csv(os.path.join(d, "test_predictions.csv"), index=False)
        save_json(full_report(o["y"].values, probs, classes), os.path.join(d, "test_metrics.json"))

    lines = [f"OOD test: ISIC 2019 images not in HAM10000, {len(df)} images, 7 classes, fold-ensemble of "
             f"{len(a.folds)} models{' + TTA' if a.tta else ''}\n"]
    rows = []
    for e in a.exps:
        m = json.load(open(os.path.join(a.out, e, "test_metrics.json")))
        rows.append({"experiment": e, "balanced_accuracy": m["balanced_accuracy"],
                     "bacc_ci95": [round(v, 4) for v in m["ci95"]["balanced_accuracy"]],
                     "macro_f1": m["macro_f1"], "macro_auc": m["macro_auc"], "accuracy": m["accuracy"],
                     "mel_sens@spec80": m.get("mel_sens@spec80")})
    lines.append(pd.DataFrame(rows).round(4).to_string(index=False))
    pairs = [tuple(a.primary)] + [(x, y) for x, y in [("ham_chromo", "ham_none"),
                                                     ("ham_chromo", "ham_chromo_global"),
                                                     ("ham_galdran", "ham_none")]
                                  if x in a.exps and y in a.exps and (x, y) != tuple(a.primary)]
    for i, (x, y) in enumerate(pairs):
        A, B = pooled_predictions(a.out, x), pooled_predictions(a.out, y)
        n, res = paired_bootstrap(A, B)
        tag = "PRIMARY" if i == 0 else "secondary"
        lines.append(f"\n[{tag}] {x} minus {y} (paired bootstrap, n={n})")
        for k, r in res.items():
            lines.append(f"  {k:18s} {r['delta']*100:+.2f} pts  CI [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]  p={r['p_two_sided']:.3f}")
        if i == 0:
            ok = res["balanced_accuracy"]["ci95"][0] > 0
            lines.append(f"  PRIMARY ENDPOINT MET: {'YES' if ok else 'NO'}")
    txt = "\n".join(lines)
    open(a.results, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
