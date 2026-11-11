import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from chromo import ChromoNet
from ood_eval import predict
from train import ImageDataset, build_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="isic19_noham.csv"); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--exp", default="ham_baseline"); ap.add_argument("--out", default="runs_ood_pf")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in a.folds:
        run = os.path.join(a.runs, f"{a.exp}_f{k}")
        out = os.path.join(a.out, f"{a.exp}_f{k}")
        if os.path.exists(os.path.join(out, "ood_predictions.csv")):
            print(f"skip {out}"); continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
        df = pd.read_csv(a.data); df = df[df["label"].isin(classes)].reset_index(drop=True)
        _, tf = build_transforms(cfg["img_size"], color_constancy=True)
        dl = DataLoader(ImageDataset(df, c2i, tf), batch_size=a.bs, num_workers=4)
        m = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
        m.load_state_dict(torch.load(os.path.join(run, "last.pt"), map_location=device))
        p = predict(m, dl, device, tta=False)
        os.makedirs(out, exist_ok=True)
        o = df[["image_path", "label", "group"]].copy(); o["y"] = df["label"].map(c2i).values
        for j, c in enumerate(classes):
            o[f"p_{c}"] = p[:, j]
        o.to_csv(os.path.join(out, "ood_predictions.csv"), index=False)
        print(f"fold {k}: {len(o)} OOD predictions -> {out}")


if __name__ == "__main__":
    main()
