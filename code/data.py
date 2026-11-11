import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Canonical lower-case class names shared by all datasets
ISIC2019_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl",
                "DF": "df", "VASC": "vasc", "SCC": "scc"}


def _find_image(img_dir, image_id, exts=(".jpg", ".jpeg", ".png")):
    for e in exts:
        p = os.path.join(img_dir, image_id + e)
        if os.path.exists(p):
            return p
    return None


def _attach_paths(df, img_dir, id_col):
    df = df.copy()
    df["image_path"] = [_find_image(img_dir, str(i)) for i in df[id_col]]
    missing = df["image_path"].isna().sum()
    if missing:
        print(f"[warn] {missing} images not found in {img_dir}; they are dropped.")
    return df.dropna(subset=["image_path"])


def prepare_ham10000(meta_csv, img_dir):
    m = pd.read_csv(meta_csv)
    m = _attach_paths(m, img_dir, "image_id")
    out = pd.DataFrame({
        "image_id": m["image_id"],
        "image_path": m["image_path"],
        "label": m["dx"].str.lower(),
        "group": m["lesion_id"].astype(str),
    })
    return out


def prepare_isic2019(gt_csv, meta_csv, img_dir, exclude_ids=None):
    gt = pd.read_csv(gt_csv)
    cls_cols = [c for c in ISIC2019_MAP if c in gt.columns]
    gt["label"] = gt[cls_cols].idxmax(axis=1).map(ISIC2019_MAP)
    meta = pd.read_csv(meta_csv)[["image", "lesion_id"]]
    df = gt[["image", "label"]].merge(meta, on="image", how="left")
    # images without lesion_id are their own group
    df["group"] = df["lesion_id"].fillna(df["image"]).astype(str)
    if exclude_ids is not None:
        before = len(df)
        df = df[~df["image"].isin(set(exclude_ids))]
        print(f"[info] excluded {before - len(df)} images that also appear in the external set")
    # ISIC 2019 file names sometimes carry the suffix _downsampled
    df["image_path"] = [_find_image(img_dir, i) or _find_image(img_dir, i + "_downsampled")
                        for i in df["image"]]
    df = df.dropna(subset=["image_path"])
    return df.rename(columns={"image": "image_id"})[["image_id", "image_path", "label", "group"]]


def prepare_isic2020(train_csv, img_dir):
    """ISIC 2020 is a BINARY task (benign vs malignant). Most benign images have
    diagnosis 'unknown', so the label must be the `target` column, not `diagnosis`."""
    t = pd.read_csv(train_csv)
    t = _attach_paths(t, img_dir, "image_name")
    out = pd.DataFrame({
        "image_id": t["image_name"],
        "image_path": t["image_path"],
        "label": np.where(t["target"] == 1, "malignant", "benign"),
        "group": t["patient_id"].astype(str),
    })
    return out


def prepare_isic2018_test(gt_csv, img_dir):
    """Official ISIC 2018 Task 3 test set (1512 images). Ground truth is one-hot
    (image,MEL,NV,BCC,AKIEC,BKL,DF,VASC). Images may sit in a sub-folder -> searched recursively."""
    gt = pd.read_csv(gt_csv)
    cls = [c for c in gt.columns if c != "image"]
    found = {}
    for root, _, files in os.walk(img_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                found[stem] = os.path.join(root, f)
    out = pd.DataFrame({
        "image_id": gt["image"],
        "image_path": gt["image"].map(found),
        "label": gt[cls].idxmax(axis=1).str.lower(),
        "group": gt["image"],
    })
    missing = out["image_path"].isna().sum()
    if missing:
        print(f"[warn] {missing} test images not found under {img_dir}")
    return out.dropna(subset=["image_path"])


def prepare_generic(csv, path_col, label_col, group_col=None, img_dir=None):
    d = pd.read_csv(csv)
    paths = d[path_col].astype(str)
    if img_dir:
        paths = [os.path.join(img_dir, p) for p in paths]
    out = pd.DataFrame({
        "image_id": [os.path.splitext(os.path.basename(p))[0] for p in paths],
        "image_path": paths,
        "label": d[label_col].astype(str).str.lower(),
        "group": d[group_col].astype(str) if group_col else [os.path.basename(p) for p in paths],
    })
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def assign_folds(df, n_folds=5, seed=42):
    """Adds column `fold` (0..n_folds-1). Groups never cross folds; class ratios are kept."""
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for f, (_, te) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
        df.loc[te, "fold"] = f
    return df


def get_split(df, test_fold, val_frac=0.125, seed=42):
    """Return train/val/test DataFrames for one outer fold.
    test = fold `test_fold`; val = a group-aware stratified part of the rest."""
    test = df[df["fold"] == test_fold].reset_index(drop=True)
    rest = df[df["fold"] != test_fold].reset_index(drop=True)
    n_inner = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + test_fold)
    tr_idx, va_idx = next(sgkf.split(rest, rest["label"], groups=rest["group"]))
    train, val = rest.iloc[tr_idx].reset_index(drop=True), rest.iloc[va_idx].reset_index(drop=True)
    check_no_leak(train, val, test)
    return train, val, test


def check_no_leak(train, val, test):
    g_tr, g_va, g_te = set(train["group"]), set(val["group"]), set(test["group"])
    assert not (g_tr & g_va), "group leak train/val"
    assert not (g_tr & g_te), "group leak train/test"
    assert not (g_va & g_te), "group leak val/test"
    i_tr, i_te = set(train["image_path"]), set(test["image_path"])
    assert not (i_tr & i_te), "image leak train/test"


def summary(df):
    s = df.groupby(["fold", "label"]).size().unstack(fill_value=0) if "fold" in df else df["label"].value_counts()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ham10000", "isic2019", "isic2020", "isic2018test", "generic"])
    ap.add_argument("--meta"); ap.add_argument("--gt"); ap.add_argument("--train_csv")
    ap.add_argument("--csv"); ap.add_argument("--path_col"); ap.add_argument("--label_col")
    ap.add_argument("--group_col")
    ap.add_argument("--img_dir")
    ap.add_argument("--exclude_ids", help="prepared CSV whose image_id's must be removed (e.g. ham.csv)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--no_folds", action="store_true", help="external test set: do not assign folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "ham10000":
        df = prepare_ham10000(a.meta, a.img_dir)
    elif a.dataset == "isic2019":
        ex = pd.read_csv(a.exclude_ids)["image_id"].tolist() if a.exclude_ids else None
        df = prepare_isic2019(a.gt, a.meta, a.img_dir, ex)
    elif a.dataset == "isic2020":
        df = prepare_isic2020(a.train_csv, a.img_dir)
    elif a.dataset == "isic2018test":
        df = prepare_isic2018_test(a.gt, a.img_dir)
    else:
        df = prepare_generic(a.csv, a.path_col, a.label_col, a.group_col, a.img_dir)

    if not a.no_folds:
        df = assign_folds(df, a.n_folds, a.seed)
    df.to_csv(a.out, index=False)
    print(f"saved {len(df)} rows, {df['group'].nunique()} groups -> {a.out}")
    print(summary(df))


if __name__ == "__main__":
    main()
