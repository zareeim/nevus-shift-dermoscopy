import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from data import get_split
from metrics import save_json
from train import MEAN, STD, set_seed

H, W = 288, 384  # 4:3 like the HAM10000 images


class SegData(Dataset):
    def __init__(self, df, train):
        self.p = df["image_path"].tolist()
        self.m = df["mask_path"].tolist() if "mask_path" in df else [None] * len(df)
        self.train = train

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        size = img.size
        img = img.resize((W, H), Image.BILINEAR)
        m = Image.open(self.m[i]).convert("L").resize((W, H), Image.NEAREST) if self.m[i] else Image.new("L", (W, H))
        if self.train:
            if random.random() < 0.5:
                img, m = TF.hflip(img), TF.hflip(m)
            if random.random() < 0.5:
                img, m = TF.vflip(img), TF.vflip(m)
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            img = TF.adjust_hue(img, random.uniform(-0.03, 0.03))
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = (TF.to_tensor(m) > 0.5).float()
        return x, y, i, torch.tensor(size)


def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((1, 2, 3)) + eps
    return 1 - (num / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


def postprocess(prob):
    m = prob > 0.5
    lab, n = ndi.label(m)
    if n > 1:  # keep the largest connected component
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    if m.sum() == 0:  # nothing found: fall back to a central ellipse (rare)
        h, w = prob.shape
        m = np.zeros_like(prob, dtype=np.uint8)
        cv2.ellipse(m, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, 1, -1)
        m = m.astype(bool)
    return m


@torch.no_grad()
def predict_and_save(model, df, out_dir, device):
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    dl = DataLoader(SegData(df, False), batch_size=16, num_workers=4)
    model.eval()
    rows, scores = [], []
    for x, y, idx, size in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit = model(x.to(device))["out"].float()
        logit = (logit + torch.flip(model(torch.flip(x.to(device), [-1]))["out"].float(), [-1])) / 2
        prob = torch.sigmoid(logit).cpu()
        for b in range(len(idx)):
            i = int(idx[b]); w0, h0 = int(size[b][0]), int(size[b][1])
            p = F.interpolate(prob[b:b + 1], size=(h0, w0), mode="bilinear", align_corners=False)[0, 0].numpy()
            m = postprocess(p)
            stem = os.path.splitext(os.path.basename(df["image_path"].iloc[i]))[0]
            mp = os.path.join(out_dir, "masks", f"{stem}_segmentation.png")
            cv2.imwrite(mp, (m * 255).astype(np.uint8))
            rows.append({"image_path": df["image_path"].iloc[i], "mask_path": os.path.abspath(mp)})
            if "mask_path" in df:
                g = np.array(Image.open(df["mask_path"].iloc[i]).convert("L").resize((w0, h0), Image.NEAREST)) > 127
                inter, uni = (m & g).sum(), (m | g).sum()
                scores.append((2 * inter / max(1, m.sum() + g.sum()), inter / max(1, uni)))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "pred_masks.csv"), index=False)
    if scores:
        s = np.array(scores)
        res = {"n": len(s), "dice_mean": float(s[:, 0].mean()), "iou_mean": float(s[:, 1].mean()),
               "dice_below_0.7_frac": float((s[:, 0] < 0.7).mean())}
        save_json(res, os.path.join(out_dir, "seg_metrics.json"))
        print("segmentation vs dermatologist masks:", {k: round(v, 4) for k, v in res.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_all", action="store_true"); ap.add_argument("--predict_csv", default=None)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.data)
    if a.train_all:
        tr, target = df, (pd.read_csv(a.predict_csv) if a.predict_csv else None)
    else:
        tr, va, te = get_split(df, a.fold, seed=a.seed)
        target = pd.concat([va, te], ignore_index=True)

    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone="DEFAULT", num_classes=1, aux_loss=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(SegData(tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y, _, _ in dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logit = model(x)["out"]
            logit = logit.float()
            loss = F.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        print(f"seg ep {ep+1:02d} loss {tot / len(dl):.4f}")
    torch.save(model.state_dict(), os.path.join(a.out, "seg.pt"))
    if target is not None:
        predict_and_save(model, target, a.out, device)


if __name__ == "__main__":
    main()
