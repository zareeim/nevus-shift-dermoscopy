import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import get_split
from losses import make_class_weights
from metrics import core_metrics, full_report, save_json
from model import build_backbone
from rpr import N_DESC, descriptor_names, relative_physical_descriptors
from train import EMA, MEAN, STD, ShadesOfGray, set_seed

CHROMO_NAMES = ["melanin_epidermal", "melanin_dermal", "hemoglobin", "shading"]
# Approximate optical-density signatures in sRGB bands (R, G, B). They are only the
# INITIALISATION of the learnable unmixing (qualitative shapes from skin optics):
SPECTRA = {
    "melanin_epidermal": (0.35, 0.60, 1.00),  # absorption increases towards short wavelengths
    "melanin_dermal": (0.80, 0.70, 0.50),     # deep melanin looks blue-grey (Tyndall scattering)
    "hemoglobin": (0.10, 1.00, 0.65),         # strong green/blue (Soret, Q bands), weak red
    "shading": (1.00, 1.00, 1.00),            # illumination / intensity, wavelength-flat
}


# --------------------------------------------------------------------------- physics layer
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class ChromophoreLayer(nn.Module):
    def __init__(self, learnable=True, eps=1e-3):
        super().__init__()
        M = torch.tensor([SPECTRA[k] for k in CHROMO_NAMES], dtype=torch.float32).T  # 3 x 4
        M = M / M.norm(dim=0, keepdim=True)
        W0 = torch.linalg.pinv(M)  # 4 x 3, minimum-norm unmixing
        self.W = nn.Parameter(W0.clone(), requires_grad=learnable)
        self.register_buffer("W0", W0)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.eps = eps

    def optical_density(self, x_norm):
        rgb = (x_norm.float() * self.std + self.mean).clamp(0, 1)
        return -torch.log(srgb_to_linear(rgb).clamp_min(self.eps))  # Beer-Lambert

    def forward(self, x_norm):
        od = self.optical_density(x_norm)
        return torch.einsum("kc,bchw->bkhw", self.W, od), od

    def drift(self):
        return float((self.W - self.W0).norm() / self.W0.norm())


def masked_mean_std(c, w, eps=1e-6):
    s = w.sum((2, 3)).clamp_min(eps)
    m = (c * w).sum((2, 3)) / s
    v = (((c - m[..., None, None]) ** 2) * w).sum((2, 3)) / s
    return m, v.clamp_min(0).sqrt()


def border_ring(h, w, frac=0.1, device="cpu"):
    r = torch.zeros(1, 1, h, w, device=device)
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    r[..., :bh, :] = 1; r[..., -bh:, :] = 1; r[..., :, :bw] = 1; r[..., :, -bw:] = 1
    return r


def self_reference(c, masks):
    """c: (B,K,H,W) maps; masks: (B,2,H0,W0) = [lesion, valid].
    Returns maps relative to own skin and a statistics vector (B, 4K)."""
    B, K, H, W = c.shape
    m = F.interpolate(masks, size=(H, W), mode="nearest")
    lesion, valid = m[:, :1], m[:, 1:2]
    inside = lesion * valid
    outside = (1 - lesion) * valid
    # too little visible healthy skin -> use the outer border ring (still excluding invalid pixels)
    few = outside.mean((2, 3), keepdim=True) < 0.05
    outside = torch.where(few, border_ring(H, W, device=c.device) * valid, outside)
    empty = inside.sum((2, 3), keepdim=True) < 1
    inside = torch.where(empty, valid, inside)
    mi, si = masked_mean_std(c, inside)
    mo, so = masked_mean_std(c, outside)
    rel = (c - mo[..., None, None]) / (so[..., None, None] + 0.05)
    stats = torch.cat([mi - mo, si / (so + 0.05), mi, mo], 1)
    return rel, stats


# --------------------------------------------------------------------------- model
def conv_bn(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class ChromoNet(nn.Module):
    def __init__(self, num_classes, config="full", backbone="resnet50", pretrained=True, learn_chromo=True):
        super().__init__()
        assert config in ("baseline", "od_ref", "chromo", "full", "rpr")
        self.config = config
        self.stem, self.stages, chans = build_backbone(backbone, pretrained)
        d = chans[-1]
        # ConvNeXt's own head normalises the pooled features; other backbones: identity
        self.bb_norm = nn.LayerNorm(d) if backbone.lower().startswith("convnext") else nn.Identity()
        self.chromo = ChromophoreLayer(learn_chromo) if config != "baseline" else None
        if config != "baseline":
            k = 3 if config == "od_ref" else len(CHROMO_NAMES)
            self.k = k
            self.branch = nn.Sequential(conv_bn(k, 32), conv_bn(32, 64), conv_bn(64, 128), conv_bn(128, 256))
            d += 256
            if config in ("od_ref", "full"):
                self.stats_mlp = nn.Sequential(nn.Linear(4 * k, 64), nn.ReLU(inplace=True))
                d += 64
            elif config == "rpr":  # statistics + relative physical representation
                n = 4 * k + N_DESC
                self.stats_mlp = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 128), nn.ReLU(inplace=True))
                d += 128
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, num_classes))

    def backbone_params(self):
        return list(self.stem.parameters()) + list(self.stages.parameters())

    def chromo_maps(self, x, masks=None):
        """Physics maps used by the branch (for the sanity check)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            c, od = self.chromo(x)
            base = od if self.config == "od_ref" else c
            if self.config in ("od_ref", "full", "rpr"):
                rel, stats = self_reference(base, masks)
                if self.config == "rpr":
                    stats = torch.cat([stats, relative_physical_descriptors(rel, masks)], 1)
                return rel, stats, c
            return base, None, c

    def forward(self, x, masks=None):
        z = self.stem(x)
        for st in self.stages:
            z = st(z)
        feats = [self.bb_norm(z.mean((2, 3)))]
        if self.config != "baseline":
            maps, stats, _ = self.chromo_maps(x, masks)
            feats.append(self.branch(maps.to(z.dtype)).mean((2, 3)))
            if stats is not None:
                feats.append(self.stats_mlp(stats.to(z.dtype)))
        return self.fc(torch.cat([f.float() for f in feats], 1))


# --------------------------------------------------------------------------- data
class JointTransform:
    """Same geometry for image and lesion mask; colour augmentation only on the image.
    Returns image tensor and a 2-channel mask [lesion, valid] (valid = inside the image after
    rotation and not part of the dark dermoscope vignette)."""

    def __init__(self, size, train, color_constancy=True):
        self.size, self.train = size, train
        self.cc = ShadesOfGray() if color_constancy else None
        self.jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.02)
        self.norm = T.Normalize(MEAN, STD)

    def __call__(self, img, mask):
        if self.cc:
            img = self.cc(img)
        dark = Image.fromarray(((np.asarray(img.convert("L")) > 20) * 255).astype(np.uint8))
        ones = Image.new("L", img.size, 255)
        NEAREST = T.InterpolationMode.NEAREST
        s = self.size
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.6, 1.0), ratio=(0.8, 1.25))
            img = TF.resized_crop(img, i, j, h, w, [s, s], antialias=True)
            mask, dark, ones = (TF.resized_crop(m, i, j, h, w, [s, s], interpolation=NEAREST) for m in (mask, dark, ones))
            if random.random() < 0.5:
                img, mask, dark, ones = (TF.hflip(m) for m in (img, mask, dark, ones))
            if random.random() < 0.5:
                img, mask, dark, ones = (TF.vflip(m) for m in (img, mask, dark, ones))
            if random.random() < 0.5:
                a = random.uniform(-180, 180)
                img = TF.rotate(img, a)
                mask, dark, ones = (TF.rotate(m, a, interpolation=NEAREST) for m in (mask, dark, ones))
            img = self.jitter(img)
        else:
            img = TF.center_crop(TF.resize(img, s, antialias=True), s)
            mask, dark, ones = (TF.center_crop(TF.resize(m, s, interpolation=NEAREST), s) for m in (mask, dark, ones))
        les = (TF.to_tensor(mask) > 0.5).float()
        valid = ((TF.to_tensor(ones) > 0.5) & (TF.to_tensor(dark) > 0.5)).float()
        return self.norm(TF.to_tensor(img)), torch.cat([les, valid], 0)


class MaskDataset(Dataset):
    def __init__(self, df, c2i, tf):
        self.p, self.m = df["image_path"].tolist(), df["mask_path"].tolist()
        self.y = [c2i.get(l, -1) for l in df["label"]]
        self.tf = tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        mask = Image.open(self.m[i]).convert("L").resize(img.size, Image.NEAREST)
        x, m = self.tf(img, mask)
        return x, m, self.y[i], i


# --------------------------------------------------------------------------- train
@torch.no_grad()
def predict(model, loader, device, tta=False):
    """tta=True: average softmax over identity + 3 flips (image AND mask flipped together)."""
    model.eval()
    P, Y, I = [], [], []
    flips = [None, [-1], [-2], [-1, -2]] if tta else [None]
    for x, m, y, idx in loader:
        x, m = x.to(device), m.to(device)
        p = 0
        for fl in flips:
            xx, mm = (x, m) if fl is None else (torch.flip(x, fl), torch.flip(m, fl))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                p = p + torch.softmax(model(xx, mm).float(), 1)
        P.append((p / len(flips)).cpu()); Y.append(y); I.append(idx)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def train(a):
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.data)
    if "mask_path" not in df:
        raise SystemExit("data CSV has no mask_path column - run add_masks.py first")
    tr, va, te = get_split(df, a.fold, seed=a.seed)  # test fold used only with --eval_test
    if a.pred_masks:  # validation/test images use AUTOMATIC masks (as for any new image)
        pm = pd.read_csv(a.pred_masks).set_index("image_path")["mask_path"]
        for d in (va, te):
            d["mask_path"] = d["image_path"].map(pm)
            if d["mask_path"].isna().any():
                raise SystemExit("predicted masks missing for some validation/test images")
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}
    cfg = vars(a).copy(); cfg.update(classes=classes, n_train=len(tr), n_val=len(va))
    save_json(cfg, os.path.join(a.out, "config.json"))
    print(f"config={a.config} device={device} train={len(tr)} val={len(va)}")

    tl = DataLoader(MaskDataset(tr, c2i, JointTransform(a.img_size, True)), batch_size=a.bs,
                    shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(MaskDataset(va, c2i, JointTransform(a.img_size, False)), batch_size=a.bs,
                    num_workers=a.workers, pin_memory=True)

    model = ChromoNet(len(classes), a.config, a.backbone, not a.no_pretrained, not a.fixed_chromo).to(device)
    bb = model.backbone_params(); bb_ids = {id(p) for p in bb}
    W = [model.chromo.W] if model.chromo is not None and model.chromo.W.requires_grad else []
    w_ids = {id(p) for p in W}
    new = [p for p in model.parameters() if id(p) not in bb_ids | w_ids and p.requires_grad]
    groups = [{"params": bb, "lr": a.lr},
              {"params": [p for p in new if p.ndim > 1], "lr": a.lr * 10},
              {"params": [p for p in new if p.ndim <= 1], "lr": a.lr * 10, "weight_decay": 0.0}]
    if W:
        groups.append({"params": W, "lr": a.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    spe = len(tl) if not a.max_batches else min(len(tl), a.max_batches)
    total, warm = a.epochs * spe, a.warmup * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / max(1, warm) if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    cw = make_class_weights([c2i[l] for l in tr["label"]], len(classes), "inv_sqrt", device)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    from augment import build as build_aug
    augmenter = build_aug(a.aug, device, tr["image_path"].tolist())

    log = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b, (x, m, y, _) in enumerate(tl):
            if a.max_batches and b >= a.max_batches:
                break
            x, m, y = x.to(device, non_blocking=True), m.to(device, non_blocking=True), y.to(device)
            if augmenter is not None:
                x = augmenter(x, m)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x, m)
            loss = F.cross_entropy(logits.float(), y, weight=cw, label_smoothing=0.05)
            if W and a.phys_reg > 0:  # keep the unmixing close to its physical initialisation
                loss = loss + a.phys_reg * ((model.chromo.W - model.chromo.W0) ** 2).sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema:
                ema.update(model)
            run += loss.item(); nb += 1
        em = ema.model if ema else model
        pv, yv, _ = predict(em, vl, device)
        vm = core_metrics(yv, pv, classes)
        drift = f" W-drift={em.chromo.drift():.3f}" if em.chromo is not None else ""
        row = dict(epoch=ep + 1, loss=run / max(1, nb), val_bacc=vm["balanced_accuracy"],
                   val_f1=vm["macro_f1"], val_auc=vm["macro_auc"], time=time.time() - t0)
        log.append(row); pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)
        print(f"ep {ep+1:03d} loss {row['loss']:.4f} val bacc {row['val_bacc']:.4f} f1 {row['val_f1']:.4f} "
              f"auc {row['val_auc']:.4f}{drift} ({row['time']:.0f}s)")
        torch.save(em.state_dict(), os.path.join(a.out, "last.pt"))

    em = ema.model if ema else model
    pv, yv, iv = predict(em, vl, device)
    out = va.iloc[iv][["image_path", "label", "group"]].copy(); out["y"] = yv
    for k, c in enumerate(classes):
        out[f"p_{c}"] = pv[:, k]
    out.to_csv(os.path.join(a.out, "val_predictions.csv"), index=False)
    vm = core_metrics(yv, pv, classes)
    save_json(vm, os.path.join(a.out, "val_metrics.json"))
    print("VAL:", json.dumps({k: round(vm[k], 4) for k in ["balanced_accuracy", "macro_f1", "macro_auc"]}))
    if a.eval_test:  # touched once, with the final (last-epoch EMA) weights
        sl = DataLoader(MaskDataset(te, c2i, JointTransform(a.img_size, False)), batch_size=a.bs,
                        num_workers=a.workers, pin_memory=True)
        pt, yt, it = predict(em, sl, device)
        out = te.iloc[it][["image_path", "label", "group"]].copy(); out["y"] = yt
        for k, c in enumerate(classes):
            out[f"p_{c}"] = pt[:, k]
        out.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
        tm = full_report(yt, pt, classes)
        save_json(tm, os.path.join(a.out, "test_metrics.json"))
        print("TEST:", json.dumps({k: round(tm[k], 4) for k in ["balanced_accuracy", "macro_f1", "macro_auc"]}))


# --------------------------------------------------------------------------- physics sanity check
@torch.no_grad()
def check(a):
    """Do the learned maps behave like chromophores? Uses the run's VALIDATION images only.
    Expected if physics was learned:
      (1) haemoglobin, relative to own skin, is highest in vascular lesions (vasc)
      (2) dermal (deep) melanin, relative to own skin, is higher in melanoma than in nevi
      (3) epidermal melanin is higher inside pigmented lesions (nv, mel) than in vasc"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    if cfg["config"] not in ("chromo", "full", "rpr"):
        raise SystemExit("sanity check needs a run with chromophore maps (config chromo or full)")
    classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    model = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, "last.pt"), map_location=device))
    model.eval()
    df = pd.read_csv(cfg["data"]); _, va, _ = get_split(df, cfg["fold"], seed=cfg["seed"])
    dl = DataLoader(MaskDataset(va, c2i, JointTransform(cfg["img_size"], False)), batch_size=32, num_workers=4)
    rows, shown = [], {}
    os.makedirs(os.path.join(a.run, "maps"), exist_ok=True)
    for x, m, y, idx in dl:
        x, m = x.to(device), m.to(device)
        c, _ = model.chromo(x)
        rel, _ = self_reference(c, m)  # always evaluate relative to own skin, for both configs
        les = (m[:, :1] * m[:, 1:2])
        val = (rel * les).sum((2, 3)) / les.sum((2, 3)).clamp_min(1)
        for b in range(len(y)):
            rows.append({"label": classes[int(y[b])], **{n: float(val[b, k]) for k, n in enumerate(CHROMO_NAMES)}})
            lab = classes[int(y[b])]
            if shown.get(lab, 0) < 3:  # save example maps: image | 4 maps
                shown[lab] = shown.get(lab, 0) + 1
                img = ((x[b].cpu() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1))
                       .clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                tiles = [img[:, :, ::-1]]
                for k in range(len(CHROMO_NAMES)):
                    t = rel[b, k].cpu().numpy(); t = (255 * (t - t.min()) / (np.ptp(t) + 1e-6)).astype(np.uint8)
                    tiles.append(cv2.applyColorMap(t, cv2.COLORMAP_VIRIDIS))
                cv2.imwrite(os.path.join(a.run, "maps", f"{lab}_{shown[lab]}.jpg"), np.hstack(tiles))
    t = pd.DataFrame(rows).groupby("label")[CHROMO_NAMES].mean()
    print("mean chromophore inside lesion, relative to own skin (z-units), validation images:")
    print(t.round(3).to_string())
    res = {}
    if "vasc" in t.index:
        res["1_hemoglobin_highest_in_vasc"] = bool(t["hemoglobin"].idxmax() == "vasc")
    if {"mel", "nv"} <= set(t.index):
        res["2_dermal_melanin_mel_gt_nv"] = bool(t.loc["mel", "melanin_dermal"] > t.loc["nv", "melanin_dermal"])
    if {"nv", "mel", "vasc"} <= set(t.index):
        res["3_epidermal_melanin_pigmented_gt_vasc"] = bool(
            min(t.loc["nv", "melanin_epidermal"], t.loc["mel", "melanin_epidermal"]) > t.loc["vasc", "melanin_epidermal"])
    res["W_drift_from_physical_init"] = model.chromo.drift()
    res["passed"] = all(v for k, v in res.items() if k[0].isdigit())
    print(json.dumps(res, indent=1))
    t.to_csv(os.path.join(a.run, "chromo_by_class.csv"))
    save_json(res, os.path.join(a.run, "physics_check.json"))
    print(f"example maps (image | epidermal mel | dermal mel | hemoglobin | shading) -> {a.run}/maps/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--fold", type=int, default=0)
    t.add_argument("--out", required=True)
    t.add_argument("--config", choices=["baseline", "od_ref", "chromo", "full", "rpr"], default="full")
    t.add_argument("--backbone", default="resnet50"); t.add_argument("--no_pretrained", action="store_true")
    t.add_argument("--fixed_chromo", action="store_true", help="freeze unmixing at its physical initialisation")
    t.add_argument("--phys_reg", type=float, default=1.0, help="L2 pull of the unmixing towards physics")
    t.add_argument("--img_size", type=int, default=320); t.add_argument("--bs", type=int, default=24)
    t.add_argument("--epochs", type=int, default=30); t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--wd", type=float, default=0.05); t.add_argument("--warmup", type=int, default=2)
    t.add_argument("--ema", type=float, default=0.999); t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--max_batches", type=int, default=0)
    t.add_argument("--pred_masks", default=None, help="CSV image_path,mask_path from seg.py for val/test images")
    t.add_argument("--eval_test", action="store_true", help="also evaluate the fold's test images (once)")
    t.add_argument("--aug", choices=["none", "galdran", "chromo", "chromo_global"], default="none",
                   help="extra training-time colour augmentation (see augment.py)")
    c = sub.add_parser("check"); c.add_argument("--run", required=True)
    p = sub.add_parser("predict", help="re-predict a finished run's val+test folds with TTA (no retraining)")
    p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--tta", action="store_true"); p.add_argument("--bs", type=int, default=16)
    dsc = sub.add_parser("describe", help="relative physical descriptors of validation images, per class")
    dsc.add_argument("--run", required=True)
    a = ap.parse_args()
    {"train": train, "check": check, "predict": repredict, "describe": describe}[a.cmd](a)


@torch.no_grad()
def describe(a):
    """Writes <run>/rpr_features.csv (one row per validation image) and prints per-class means of
    the clinically interpretable descriptors. Uses the run's own masks (automatic if it had them)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    model = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, "last.pt"), map_location=device))
    model.eval()
    df = pd.read_csv(cfg["data"]); _, va, _ = get_split(df, cfg["fold"], seed=cfg["seed"])
    if cfg.get("pred_masks"):
        pm = pd.read_csv(cfg["pred_masks"]).set_index("image_path")["mask_path"]
        va["mask_path"] = va["image_path"].map(pm)
    dl = DataLoader(MaskDataset(va, c2i, JointTransform(cfg["img_size"], False)), batch_size=32, num_workers=4)
    names, rows = descriptor_names(), []
    for x, m, y, idx in dl:
        x, m = x.to(device), m.to(device)
        c, _ = model.chromo(x)
        rel, _ = self_reference(c, m)
        f = relative_physical_descriptors(rel, m).cpu().numpy()
        for b in range(len(y)):
            rows.append({"image_path": va["image_path"].iloc[int(idx[b])], "label": classes[int(y[b])],
                         **dict(zip(names, f[b]))})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.run, "rpr_features.csv"), index=False)
    t["mel_epi_border_step"] = t["mel_epi_ring_in1"] - t["mel_epi_ring_out1"]
    key = ["depth_derm_minus_epi", "hb_minus_mel_epi", "mel_epi_wasserstein", "mel_epi_het_ratio",
           "mel_epi_border_step", "mel_epi_asym_ax1", "shape_asym_ax1"]
    pd.set_option("display.width", 200)
    print("per-class mean of key relative physical descriptors (validation images):")
    print(t.groupby("label")[key].mean().round(3).to_string())


@torch.no_grad()
def repredict(a):
    """Loads <run>/last.pt and writes val/test predictions (+ test metrics) to --out, using the
    same automatic masks as the original run. The original run directory is not modified."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    model = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, "last.pt"), map_location=device))
    df = pd.read_csv(cfg["data"]); _, va, te = get_split(df, cfg["fold"], seed=cfg["seed"])
    if cfg.get("pred_masks"):
        pm = pd.read_csv(cfg["pred_masks"]).set_index("image_path")["mask_path"]
        for d in (va, te):
            d["mask_path"] = d["image_path"].map(pm)
    os.makedirs(a.out, exist_ok=True)
    save_json({**cfg, "tta": a.tta, "source_run": a.run}, os.path.join(a.out, "config.json"))
    for name, d in (("val", va), ("test", te)):
        dl = DataLoader(MaskDataset(d, c2i, JointTransform(cfg["img_size"], False)), batch_size=a.bs, num_workers=4)
        pr, y, idx = predict(model, dl, device, a.tta)
        out = d.iloc[idx][["image_path", "label", "group"]].copy(); out["y"] = y
        for k, c in enumerate(classes):
            out[f"p_{c}"] = pr[:, k]
        out.to_csv(os.path.join(a.out, f"{name}_predictions.csv"), index=False)
        if name == "test":
            tm = full_report(y, pr, classes)
            save_json(tm, os.path.join(a.out, "test_metrics.json"))
            print(f"{os.path.basename(a.out)}: test bacc {tm['balanced_accuracy']:.4f} f1 {tm['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
