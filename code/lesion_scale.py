import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SIZE = 320


# ------------------------------------------------------------------ geometry
def standard_view_fraction(mask, size=SIZE):
    """Lesion fraction after the standard eval transform: resize short side to size, centre crop."""
    w, h = mask.size; s = size / min(w, h)
    m = mask.resize((max(size, round(w * s)), max(size, round(h * s))), Image.NEAREST)
    W, H = m.size; l, t = (W - size) // 2, (H - size) // 2
    return float((np.asarray(m.crop((l, t, l + size, t + size))) > 127).mean())


def scale_box(mask_arr, f_star):
    """Lesion-centred square crop so the lesion covers ~f_star of it (clipped to the image)."""
    H, W = mask_arr.shape
    ys, xs = np.nonzero(mask_arr)
    if len(xs) == 0:
        side = min(W, H); return ((W - side) // 2, (H - side) // 2, side, False)
    area = len(xs); cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = np.sqrt(area / f_star)
    side = max(side, 1.05 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))  # keep whole lesion
    side = int(round(min(side, min(W, H))))
    l = int(round(np.clip(cx - side / 2, 0, W - side))); t = int(round(np.clip(cy - side / 2, 0, H - side)))
    return l, t, side, True


# ------------------------------------------------------------------ commands
def segment(a):
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    from seg import predict_and_save
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_cls = set(pd.read_csv(a.ham)["label"])
    df = pd.read_csv(a.data); df = df[df["label"].isin(ham_cls)].reset_index(drop=True)
    df = df[["image_path", "label", "group"]]
    if os.path.exists(os.path.join(a.seg_target, "pred_masks.csv")):
        print("target masks exist - skip"); return
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=1, aux_loss=False).to(device)
    model.load_state_dict(torch.load(os.path.join(a.seg_model, "seg.pt"), map_location=device))
    predict_and_save(model, df, a.seg_target, device)
    print(f"{len(df)} target masks -> {a.seg_target}")


def ham_masks(a):
    """Predicted (out-of-fold) HAM masks: fold k's seg model covers fold k's val + test images."""
    parts = []
    for k in a.folds:
        d = pd.read_csv(os.path.join("seg", f"f{k}", "pred_masks.csv")); d["fold_seg"] = k; parts.append(d)
    return pd.concat(parts, ignore_index=True)


def stats(a):
    from label_shift import source_proxy
    ham = pd.read_csv(a.ham)[["image_path", "label"]]
    hm = ham_masks(a).drop_duplicates("image_path")
    ham = ham.merge(hm[["image_path", "mask_path"]], on="image_path", how="inner")
    tg = pd.read_csv(os.path.join(a.seg_target, "pred_masks.csv"))
    lab = pd.read_csv(a.data).set_index("image_path")["label"]
    tg["label"] = tg["image_path"].map(lab)
    src = pd.read_csv(a.sources).set_index("image_path")["source"] if os.path.exists(a.sources) else None
    tg["source"] = tg["image_path"].map(src) if src is not None else [source_proxy(p) for p in tg["image_path"]]
    ham["source"] = "HAM10000"
    D = pd.concat([ham, tg], ignore_index=True)
    D["frac"] = [standard_view_fraction(Image.open(m).convert("L")) for m in D["mask_path"]]
    f_star = float(D.loc[D["source"] == "HAM10000", "frac"].median())
    L = [f"lesion fraction of the model input under the standard pipeline (predicted masks)\n",
         D.groupby("source")["frac"].describe(percentiles=[.25, .5, .75]).round(3).to_string(), "",
         "median by class:", D.pivot_table(index="label", columns="source", values="frac", aggfunc="median").round(3).to_string(), "",
         f"f* (HAM10000 median) = {f_star:.4f}"]
    # achieved fraction after scale normalisation, per source
    ach = []
    for m in D["mask_path"]:
        arr = np.asarray(Image.open(m).convert("L")) > 127
        l, t, s, ok = scale_box(arr, f_star)
        ach.append(arr[t:t + s, l:l + s].mean())
    D["frac_scaled"] = ach
    L += ["", "after lesion-scale normalisation:",
          D.groupby("source")["frac_scaled"].describe(percentiles=[.25, .5, .75]).round(3).to_string()]
    txt = "\n".join(L); print(txt)
    open("scale_stats.txt", "w").write(txt + "\nDONE\n")
    json.dump({"f_star": f_star}, open("scale_fstar.json", "w"))
    D.to_csv("scale_fractions.csv", index=False)
    # examples: 4 HAM + 8 target, standard view | scale-normalised view, lesion outline
    rng = np.random.default_rng(0)
    pick = pd.concat([D[D["source"] == "HAM10000"].sample(4, random_state=0),
                      D[D["source"] != "HAM10000"].sample(8, random_state=0)])
    tiles = []
    for _, r in pick.iterrows():
        img = Image.open(r["image_path"]).convert("RGB"); m = Image.open(r["mask_path"]).convert("L")
        w, h = img.size; s0 = min(w, h); std = img.crop(((w - s0) // 2, (h - s0) // 2, (w + s0) // 2, (h + s0) // 2))
        l, t, s, _ = scale_box(np.asarray(m) > 127, f_star); sc = img.crop((l, t, l + s, t + s))
        pair = Image.new("RGB", (2 * 200, 200))
        for j, (im, mm) in enumerate(((std, m.crop(((w - s0) // 2, (h - s0) // 2, (w + s0) // 2, (h + s0) // 2))),
                                      (sc, m.crop((l, t, l + s, t + s))))):
            im = im.resize((200, 200)); edge = np.asarray(mm.resize((200, 200), Image.NEAREST)) > 127
            e = edge ^ np.roll(edge, 1, 0) | edge ^ np.roll(edge, 1, 1)
            arr = np.asarray(im).copy(); arr[e] = (0, 255, 0); pair.paste(Image.fromarray(arr), (200 * j, 0))
        ImageDraw.Draw(pair).text((4, 4), str(r["source"])[:14], fill=(255, 255, 0))
        tiles.append(pair)
    grid = Image.new("RGB", (2 * 400, 6 * 200))
    for i, tl in enumerate(tiles):
        grid.paste(tl, ((i % 2) * 400, (i // 2) * 200))
    grid.save("scale_examples.jpg", quality=90)
    print("examples (left: standard view, right: scale-normalised; green = predicted lesion) -> scale_examples.jpg")


def predict(a):
    import torch
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader, Dataset
    from chromo import ChromoNet
    from ood_eval import predict as run_predict
    from train import MEAN, STD, ShadesOfGray

    f_star = json.load(open("scale_fstar.json"))["f_star"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cc = ShadesOfGray()

    class ScaleData(Dataset):
        def __init__(self, df, c2i):
            self.p, self.m = df["image_path"].tolist(), df["mask_path"].tolist()
            self.y = [c2i.get(l, -1) for l in df["label"]]

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            img = cc(Image.open(self.p[i]).convert("RGB"))
            arr = np.asarray(Image.open(self.m[i]).convert("L").resize(img.size, Image.NEAREST)) > 127
            l, t, s, _ = scale_box(arr, f_star)
            x = TF.to_tensor(img.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.BILINEAR))
            return TF.normalize(x, MEAN, STD), self.y[i], i

    tmask = pd.read_csv(os.path.join(a.seg_target, "pred_masks.csv")).set_index("image_path")["mask_path"]
    tgt0 = pd.read_csv(a.data)
    for k in a.folds:
        run = os.path.join(a.runs, f"{a.exp}_f{k}")
        out_r = os.path.join(a.out_runs, f"{a.exp}_scale_f{k}"); out_o = os.path.join(a.out_ood, f"{a.exp}_scale_f{k}")
        if os.path.exists(os.path.join(out_o, "ood_predictions.csv")):
            print(f"skip fold {k}"); continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        classes = cfg["classes"]; c2i = {c: i for i, c in enumerate(classes)}
        assert cfg["img_size"] == SIZE
        m = ChromoNet(len(classes), cfg["config"], cfg["backbone"], False).to(device)
        m.load_state_dict(torch.load(os.path.join(run, "last.pt"), map_location=device)); m.eval()
        hm = pd.read_csv(os.path.join("seg", f"f{k}", "pred_masks.csv")).set_index("image_path")["mask_path"]
        os.makedirs(out_r, exist_ok=True); os.makedirs(out_o, exist_ok=True)
        shutil.copy(os.path.join(run, "config.json"), out_r)
        jobs = [("val_predictions.csv", pd.read_csv(os.path.join(run, "val_predictions.csv")), hm, out_r),
                ("test_predictions.csv", pd.read_csv(os.path.join(run, "test_predictions.csv")), hm, out_r),
                ("ood_predictions.csv", tgt0[tgt0["label"].isin(classes)].reset_index(drop=True), tmask, out_o)]
        for name, d, mk, od in jobs:
            d = d[["image_path", "label", "group"]].copy(); d["mask_path"] = d["image_path"].map(mk)
            assert d["mask_path"].notna().all(), f"fold {k} {name}: masks missing"
            dl = DataLoader(ScaleData(d, c2i), batch_size=a.bs, num_workers=a.workers)
            p = run_predict(m, dl, device, tta=False)
            o = d[["image_path", "label", "group"]].copy(); o["y"] = d["label"].map(c2i).values
            for j, c in enumerate(classes):
                o[f"p_{c}"] = p[:, j]
            o.to_csv(os.path.join(od, name), index=False)
            print(f"fold {k}: {name} n={len(o)}")
        del m; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["segment", "stats", "predict"])
    ap.add_argument("--data", default="isic19_noham.csv"); ap.add_argument("--ham", default="ham.csv")
    ap.add_argument("--seg_model", default="seg/f0"); ap.add_argument("--seg_target", default="seg_target")
    ap.add_argument("--sources", default="runs_adapt/sources.csv")
    ap.add_argument("--exp", default="ham_baseline"); ap.add_argument("--runs", default="runs_chromo_cv")
    ap.add_argument("--out_runs", default="runs_scale"); ap.add_argument("--out_ood", default="runs_scale_ood")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"segment": segment, "stats": stats, "predict": predict}[a.cmd](a)


if __name__ == "__main__":
    main()
