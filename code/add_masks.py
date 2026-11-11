import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    found = {}
    for root, _, files in os.walk(a.mask_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".bmp")):
                key = f.split("_segmentation")[0].split(".")[0]
                found[key] = os.path.join(root, f)
    df = pd.read_csv(a.data)
    ids = df["image_id"] if "image_id" in df else df["image_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    df["mask_path"] = ids.map(found)
    miss = df["mask_path"].isna().sum()
    print(f"masks found for {len(df) - miss}/{len(df)} images")
    if miss:
        raise SystemExit(f"{miss} images have no mask - check --mask_dir")
    df.to_csv(a.out, index=False)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
