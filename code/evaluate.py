import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data import get_split
from metrics import full_report, save_json
from train import ImageDataset, build_model, build_transforms, predict, save_predictions


def load_run(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model = build_model(cfg, len(cfg["classes"]), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    return cfg, model.to(device).eval()


def overlay(img_rgb, heat):
    h = cv2.resize(heat, (img_rgb.shape[1], img_rgb.shape[0]))
    h = (255 * (h - h.min()) / (h.max() - h.min() + 1e-8)).astype(np.uint8)
    col = cv2.applyColorMap(h, cv2.COLORMAP_JET)[:, :, ::-1]
    return (0.55 * img_rgb + 0.45 * col).astype(np.uint8)


@torch.no_grad()
def hardness_analysis(model, df, cfg, device, out_dir, n_save, mask_dir, mask_suffix, top_q=0.2):
    os.makedirs(out_dir, exist_ok=True)
    _, eval_tf = build_transforms(cfg["img_size"], cfg.get("color_constancy", False))
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        pil = Image.open(r["image_path"]).convert("RGB")
        x = eval_tf(pil).unsqueeze(0).to(device)
        out = model(x, return_maps=True)
        if not out.get("hardness_maps"):
            raise SystemExit("this run has no hardness maps (baseline model)")
        size = cfg["img_size"]
        # average of all stage hardness maps upsampled to input size
        hm = torch.stack([F.interpolate(h, size=(size, size), mode="bilinear", align_corners=False)
                          for h in out["hardness_maps"]]).mean(0)[0, 0].cpu().numpy()
        # image (and mask) exactly as seen by the network: resize shorter side + center crop
        geo = T.Compose([T.Resize(size), T.CenterCrop(size)])
        vis = np.array(geo(pil))
        rec = {"image_path": r["image_path"], "label": r["label"]}
        if mask_dir:
            stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
            mp = os.path.join(mask_dir, stem + mask_suffix)
            if os.path.exists(mp):
                mgeo = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
                                  T.CenterCrop(size)])
                m = np.array(mgeo(Image.open(mp).convert("L"))) > 127
                e = hm / (hm.sum() + 1e-8)
                rec["energy_in_mask"] = float(e[m].sum())
                rec["mask_area_frac"] = float(m.mean())  # chance level for energy_in_mask
                yy, xx = np.unravel_index(hm.argmax(), hm.shape)
                rec["pointing_hit"] = bool(m[yy, xx])
                thr = np.quantile(hm, 1 - top_q)
                top = hm >= thr
                rec["iou_top%d" % int(top_q * 100)] = float((top & m).sum() / max(1, (top | m).sum()))
        rows.append(rec)
        if i < n_save:
            cv2.imwrite(os.path.join(out_dir, f"{i:03d}_{r['label']}.jpg"),
                        cv2.cvtColor(np.hstack([vis, overlay(vis, hm)]), cv2.COLOR_RGB2BGR))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "hardness_scores.csv"), index=False)
    if "energy_in_mask" in res:
        summ = res[[c for c in res.columns if c not in ("image_path", "label")]].mean(numeric_only=True)
        print(summ.to_string())
        save_json(summ.to_dict(), os.path.join(out_dir, "hardness_summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["all", "test"], default="all",
                    help="'test' = the run's own test fold (needs fold column)")
    ap.add_argument("--label_map", default=None, help="JSON dict external_label -> training label")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--save_maps", type=int, default=0)
    ap.add_argument("--mask_dir", default=None)
    ap.add_argument("--mask_suffix", default="_segmentation.png")
    ap.add_argument("--skip_metrics", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_run(a.run, device)
    classes = cfg["classes"]
    out_dir = a.out or os.path.join(a.run, "eval_" + os.path.splitext(os.path.basename(a.data))[0])
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(a.data)
    if a.split == "test":
        _, _, df = get_split(df, cfg["fold"], seed=cfg["seed"])
    if a.label_map:
        df["label"] = df["label"].map(lambda l: json.loads(a.label_map).get(l, l))
    unknown = df[~df["label"].isin(classes)]
    if len(unknown):
        print(f"[info] dropping {len(unknown)} images with labels not in training classes: "
              f"{unknown['label'].value_counts().to_dict()}")
    df = df[df["label"].isin(classes)].reset_index(drop=True)

    if not a.skip_metrics:
        _, eval_tf = build_transforms(cfg["img_size"], cfg.get("color_constancy", False))
        c2i = {c: i for i, c in enumerate(classes)}
        dl = DataLoader(ImageDataset(df, c2i, eval_tf), batch_size=a.bs, num_workers=4)
        p, y, idx = predict(model, dl, device, a.tta)
        save_predictions(df, p, y, idx, classes, os.path.join(out_dir, "predictions.csv"))
        m = full_report(y, p, classes)
        m["present_classes"] = sorted(df["label"].unique().tolist())
        save_json(m, os.path.join(out_dir, "metrics.json"))
        print(json.dumps({k: m[k] for k in ["n", "balanced_accuracy", "macro_f1", "macro_auc"]}, indent=1))

    if a.save_maps or a.mask_dir:
        hardness_analysis(model, df, cfg, device, os.path.join(out_dir, "hardness"),
                          a.save_maps, a.mask_dir, a.mask_suffix)


if __name__ == "__main__":
    main()
