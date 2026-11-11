import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from data import assign_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic", default="isic19_noham.csv"); ap.add_argument("--sources", default="runs_adapt/sources.csv")
    ap.add_argument("--ham", default="ham.csv"); ap.add_argument("--mask", default="masks/blank.png")
    a = ap.parse_args()
    ham = pd.read_csv(a.ham)[["image_path", "label", "group"]]
    df = pd.read_csv(a.isic)
    extra = sorted(set(df["label"]) - set(ham["label"]))
    df = df[df["label"].isin(set(ham["label"]))].reset_index(drop=True)   # e.g. scc: not a HAM class
    print(f"dropped classes not in HAM10000: {extra}")
    src = pd.read_csv(a.sources).set_index("image_path")["source"] if os.path.exists(a.sources) else pd.Series(dtype=str)
    df["source"] = df["image_path"].map(src)
    miss = df["source"].isna()
    if miss.any():
        from label_shift import source_proxy
        print(f"{miss.sum()} images not in {a.sources}: reading their size")
        df.loc[miss, "source"] = [source_proxy(p) for p in df.loc[miss, "image_path"]]
    bcn = df[df["source"].str.startswith("BCN")][["image_path", "label", "group"]].reset_index(drop=True)
    bcn = assign_folds(bcn, 5, 42)
    os.makedirs(os.path.dirname(a.mask) or ".", exist_ok=True)
    Image.fromarray(np.full((64, 64), 255, np.uint8)).save(a.mask)
    bcn.to_csv("bcn.csv", index=False)
    bcn.assign(mask_path=a.mask).to_csv("bcn_masks.csv", index=False)
    ham.to_csv("ham_ood.csv", index=False)
    print(f"BCN-like training set: {len(bcn)} images, {bcn['group'].nunique()} lesions")
    print(pd.crosstab(bcn["fold"], bcn["label"]).to_string())
    print(f"HAM10000 target: {len(ham)} images")
    print("class prior (%)  BCN-like vs HAM:")
    print(pd.DataFrame({"BCN-like": bcn["label"].value_counts(normalize=True) * 100,
                        "HAM": ham["label"].value_counts(normalize=True) * 100}).round(1).to_string())
    missing = set(ham["label"]) - set(bcn["label"])
    assert not missing, f"classes absent from BCN-like training data: {missing}"


if __name__ == "__main__":
    main()
