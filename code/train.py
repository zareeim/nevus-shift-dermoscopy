import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data import get_split
from losses import hgpb_loss, make_class_weights
from metrics import core_metrics, full_report, save_json
from model import HGPBNet

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ShadesOfGray:
    """Colour constancy (Finlayson & Trezzi 2004), standard pre-processing for dermoscopy:
    estimates the illuminant with the Minkowski p-norm of each channel and rescales the
    channels so that the illuminant becomes neutral grey."""

    def __init__(self, p=6):
        self.p = p

    def __call__(self, img):
        a = np.asarray(img, dtype=np.float32)
        illum = np.power(np.mean(np.power(a, self.p), axis=(0, 1)), 1.0 / self.p)
        illum = illum / (np.sqrt((illum ** 2).sum()) + 1e-8)
        a = a / (illum * np.sqrt(3.0) + 1e-8)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def build_transforms(size, color_constancy=False):
    pre = [ShadesOfGray()] if color_constancy else []
    train_tf = T.Compose(pre + [
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose(pre + [
        T.Resize(int(size * 1.0)), T.CenterCrop(size),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class ImageDataset(Dataset):
    def __init__(self, df, class_to_idx, tf):
        self.paths = df["image_path"].tolist()
        self.y = [class_to_idx.get(l, -1) for l in df["label"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.y[i], i


# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    P, Y, I = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)["logits"].float()
            if tta:
                for flip in ([-1], [-2], [-1, -2]):
                    logits = logits + model(torch.flip(x, flip))["logits"].float()
                logits = logits / 4
        P.append(torch.softmax(logits, 1).cpu()); Y.append(y); I.append(idx)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def save_predictions(df, probs, y, idx, classes, path):
    out = df.iloc[idx][["image_id", "image_path", "label", "group"]].copy() if "image_id" in df else \
        df.iloc[idx][["image_path", "label", "group"]].copy()
    out["y"] = y
    for k, c in enumerate(classes):
        out[f"p_{c}"] = probs[:, k]
    out.to_csv(path, index=False)


def build_model(cfg, num_classes, pretrained=True):
    return HGPBNet(
        num_classes, backbone=cfg["backbone"], pretrained=pretrained,
        boost_stages=tuple(cfg["boost_stages"]),
        use_spatial=not cfg["no_spatial"], use_committee=not cfg["no_committee"],
        lse_r=cfg["lse_r"], baseline=cfg["model"] == "baseline",
        main_head=cfg.get("main_head", "committee"),  # old runs (v1) have no key -> committee
    )


class EMA:
    """Exponential moving average of all weights and buffers (used for both models alike)."""

    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # short warm-up
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepared CSV with fold column")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["hgpb", "baseline"], default="hgpb")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="backbone lr; new layers use 10x")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10, help="0 = no early stopping")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="checkpoint used for testing: best validation epoch, or the last epoch")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay, e.g. 0.999 (0 = off)")
    ap.add_argument("--main_head", choices=["committee", "gap"], default="committee",
                    help="hgpb only: 'gap' = standard GAP head is the output, boosting is auxiliary")
    ap.add_argument("--no_test", action="store_true",
                    help="tuning runs: evaluate on validation only, never touch the test fold")
    ap.add_argument("--class_weight", choices=["none", "inv_sqrt", "inv"], default="inv_sqrt")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    # HGPB hyper-parameters
    ap.add_argument("--boost_stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--no_spatial", action="store_true")
    ap.add_argument("--no_committee", action="store_true")
    ap.add_argument("--no_sample_boost", action="store_true")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--aux_w", type=float, default=0.5)
    ap.add_argument("--lse_r", type=float, default=4.0)
    ap.add_argument("--select_metric", default="balanced_accuracy",
                    choices=["balanced_accuracy", "macro_f1", "macro_auc"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--color_constancy", action="store_true", help="Shades-of-Gray colour constancy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    ap.add_argument("--train_all", action="store_true",
                    help="final model: train on ALL rows of --data (no validation/test); "
                         "requires --select last")
    ap.add_argument("--predict_csv", default=None,
                    help="with --train_all: write predictions for this prepared CSV "
                         "(e.g. the official ISIC 2018 test set) to <out>/external_predictions.csv")
    cfg = vars(ap.parse_args())
    if cfg["train_all"] and cfg["select"] != "last":
        raise SystemExit("--train_all needs --select last (there is no validation set)")

    os.makedirs(cfg["out"], exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"])
    if cfg["train_all"]:
        train_df, val_df, test_df = df.reset_index(drop=True), None, None
    else:
        train_df, val_df, test_df = get_split(df, cfg["fold"], seed=cfg["seed"])
    classes = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n = lambda d: 0 if d is None else len(d)
    cfg.update(classes=classes, n_train=n(train_df), n_val=n(val_df), n_test=n(test_df))
    save_json(cfg, os.path.join(cfg["out"], "config.json"))
    print(f"device={device} train={n(train_df)} val={n(val_df)} test={n(test_df)} classes={classes}")

    train_tf, eval_tf = build_transforms(cfg["img_size"], cfg["color_constancy"])
    mk = lambda d, tf, sh: None if d is None else DataLoader(
        ImageDataset(d, c2i, tf), batch_size=cfg["bs"], shuffle=sh,
        num_workers=cfg["workers"], pin_memory=True, drop_last=sh)
    tl, vl, sl = mk(train_df, train_tf, True), mk(val_df, eval_tf, False), mk(test_df, eval_tf, False)

    model = build_model(cfg, len(classes), pretrained=not cfg["no_pretrained"]).to(device)
    backbone_params = list(model.stem.parameters()) + list(model.stages.parameters())
    bb_ids = {id(p) for p in backbone_params}
    new_params = [p for p in model.parameters() if id(p) not in bb_ids]
    # gamma, alpha, biases and norm parameters of the new layers get no weight decay
    new_decay = [p for p in new_params if p.ndim > 1]
    new_nodecay = [p for p in new_params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": cfg["lr"]},
                             {"params": new_decay, "lr": cfg["lr"] * 10},
                             {"params": new_nodecay, "lr": cfg["lr"] * 10, "weight_decay": 0.0}],
                            weight_decay=cfg["wd"])
    steps_per_epoch = len(tl) if not cfg["max_batches"] else min(len(tl), cfg["max_batches"])
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / max(1, warm) if t < warm else
        0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    cw = make_class_weights([c2i[l] for l in train_df["label"]], len(classes), cfg["class_weight"], device)
    ema = EMA(model, cfg["ema"]) if cfg["ema"] > 0 else None
    main_is_committee = cfg["model"] != "baseline" and cfg["main_head"] == "committee"
    ckpt = os.path.join(cfg["out"], "best.pt")

    best, best_ep, log = -1.0, -1, []
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for b, (x, y, _) in enumerate(tl):
            if cfg["max_batches"] and b >= cfg["max_batches"]:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
            out = {k: ([t.float() for t in v] if isinstance(v, list) else v.float()) for k, v in out.items()}
            loss, _ = hgpb_loss(out, y, cw, cfg["label_smoothing"], cfg["aux_w"], cfg["eta"],
                                use_sample_boost=not cfg["no_sample_boost"],
                                use_committee=not cfg["no_committee"] and cfg["model"] != "baseline",
                                main_is_committee=main_is_committee)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema:
                ema.update(model)
            run += loss.item(); nb += 1

        eval_model = ema.model if ema else model
        if vl is None:  # --train_all: no validation, keep the last weights
            print(f"ep {ep+1:03d} loss {run / max(1, nb):.4f} ({time.time() - t0:.0f}s)")
            torch.save(eval_model.state_dict(), ckpt)
            continue
        pv, yv, _ = predict(eval_model, vl, device)
        vm = core_metrics(yv, pv, classes)
        score = vm[cfg["select_metric"]]
        extra = ""
        if hasattr(model, "gammas") and len(getattr(model, "gammas", {})):
            extra = " gamma=" + ",".join(f"{g.item():.3f}" for g in eval_model.gammas.values())
            extra += " alpha=" + ",".join(f"{a.item():.2f}" for a in eval_model.alphas())
        row = dict(epoch=ep + 1, loss=run / max(1, nb), val_bacc=vm["balanced_accuracy"],
                   val_f1=vm["macro_f1"], val_auc=vm["macro_auc"], time=time.time() - t0)
        log.append(row); pd.DataFrame(log).to_csv(os.path.join(cfg["out"], "log.csv"), index=False)
        print(f"ep {ep+1:03d} loss {row['loss']:.4f} val bacc {row['val_bacc']:.4f} "
              f"f1 {row['val_f1']:.4f} auc {row['val_auc']:.4f}{extra} ({row['time']:.0f}s)")
        if score > best:
            best, best_ep = score, ep
            if cfg["select"] == "best":
                torch.save(eval_model.state_dict(), ckpt)
        elif cfg["patience"] and ep - best_ep >= cfg["patience"]:
            print("early stopping"); break
        if cfg["select"] == "last":
            torch.save(eval_model.state_dict(), ckpt)

    # ---- final evaluation: test set is touched exactly once, with the selected checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    if cfg["train_all"]:
        if cfg["predict_csv"]:
            ext = pd.read_csv(cfg["predict_csv"]).reset_index(drop=True)
            el = mk(ext, eval_tf, False)
            pe, ye, ie = predict(model, el, device, cfg["tta"])
            save_predictions(ext, pe, ye, ie, classes, os.path.join(cfg["out"], "external_predictions.csv"))
            print(f"saved predictions for {len(ext)} images -> external_predictions.csv")
        return
    pv, yv, iv = predict(model, vl, device, cfg["tta"])
    save_predictions(val_df, pv, yv, iv, classes, os.path.join(cfg["out"], "val_predictions.csv"))
    vmf = core_metrics(yv, pv, classes)
    vmf["best_val_epoch"] = best_ep + 1
    save_json(vmf, os.path.join(cfg["out"], "val_metrics.json"))
    print("VAL:", json.dumps({k: round(vmf[k], 4) for k in ["balanced_accuracy", "macro_f1", "macro_auc"]}))
    if cfg["no_test"]:
        return
    pt, yt, it = predict(model, sl, device, cfg["tta"])
    save_predictions(test_df, pt, yt, it, classes, os.path.join(cfg["out"], "test_predictions.csv"))
    tm = full_report(yt, pt, classes)
    tm["best_epoch"] = best_ep + 1
    save_json(tm, os.path.join(cfg["out"], "test_metrics.json"))
    print(json.dumps({k: tm[k] for k in ["balanced_accuracy", "macro_f1", "macro_auc", "accuracy", "ece"]}, indent=1))
    print("95% CI:", tm["ci95"])


if __name__ == "__main__":
    main()
