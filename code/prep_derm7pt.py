import argparse, os
import pandas as pd

def map_dx(d):
    d = str(d).strip().lower()
    if "metasta" in d: return None
    if d.startswith("melanoma"): return "mel"
    if "nevus" in d or "naevus" in d: return "nv"
    if "basal cell" in d: return "bcc"
    if "seborrheic" in d or "lentigo" in d: return "bkl"
    if "dermatofibroma" in d: return "df"
    if "vascular" in d: return "vasc"
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="derm7pt.csv")
a = ap.parse_args()
meta = pd.read_csv(os.path.join(a.root, "meta", "meta.csv"))
meta["label"] = meta["diagnosis"].map(map_dx)
excl = meta.loc[meta["label"].isna(), "diagnosis"].value_counts()
d = meta.dropna(subset=["label"]).copy()
d["image_path"] = [os.path.abspath(os.path.join(a.root, "images", p)) for p in d["derm"]]
miss = [p for p in d["image_path"] if not os.path.exists(p)]
if miss: raise SystemExit(f"{len(miss)} images not found, e.g. {miss[:3]}")
d["group"] = "d7_" + d["case_num"].astype(str)
d[["image_path", "label", "group"]].to_csv(a.out, index=False)
print(meta.groupby("diagnosis")["label"].first().fillna("EXCLUDED").to_string())
print(f"\nexcluded: {int(excl.sum())}\n{excl.to_string()}")
print(f"\nkept {len(d)} images, {d['group'].nunique()} lesions")
print((d["label"].value_counts(normalize=True) * 100).round(1).to_string())
