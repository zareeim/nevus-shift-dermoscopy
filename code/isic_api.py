import argparse
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

API = "https://api.isic-archive.com/api/v2/images/{}/"
GT2020 = "https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_GroundTruth_v2.csv"
DUP2020 = "https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_Duplicates.csv"
UA = {"User-Agent": "research-script/1.0"}


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0].replace("_downsampled", "")


def fetch_meta(ids, cache="isic_api_cache.jsonl", workers=8):
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            try:
                j = json.loads(line); done[j["isic_id"]] = j
            except Exception:
                pass
    todo = [i for i in ids if i not in done]
    print(f"{len(done)} cached, {len(todo)} to fetch")

    def one(i):
        try:
            return json.loads(get(API.format(i)))
        except Exception as e:
            return {"isic_id": i, "error": str(e)[:200]}
    with open(cache, "a") as f, ThreadPoolExecutor(workers) as ex:
        for n, j in enumerate(ex.map(one, todo)):
            f.write(json.dumps(j) + "\n"); done[j["isic_id"]] = j
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return done


def flatten(j):
    row = {"isic_id": j.get("isic_id"), "attribution": j.get("attribution"),
           "copyright_license": j.get("copyright_license"), "error": j.get("error")}
    for sec, d in (j.get("metadata") or {}).items():
        if isinstance(d, dict):
            for k, v in d.items():
                if not isinstance(v, (dict, list)):
                    row[f"{sec}.{k}"] = v
    return row


def cmd_meta(a):
    ids = sorted(set(pd.read_csv(a.csv)["image_path"].map(stem)))
    if a.limit:
        ids = ids[:a.limit]
    J = fetch_meta(ids)
    print("example record:", json.dumps(J[ids[0]])[:1500])
    M = pd.DataFrame([flatten(J[i]) for i in ids])
    M.to_csv(a.out, index=False)
    print(f"{len(M)} rows -> {a.out}; API errors: {M['error'].notna().sum()} ({M['error'].notna().mean():.1%})")
    print(f"non-empty attribution: {M['attribution'].notna().mean():.1%}; distinct attributions: {M['attribution'].nunique()}")
    print("metadata fields returned:", [c for c in M.columns if "." in c])
    print("attribution (top 15):"); print(M["attribution"].value_counts().head(15).to_string())
    c = [x for x in M.columns if "confirm" in x]
    if c:
        print(f"{c[0]}:"); print(M[c[0]].value_counts(dropna=False).to_string())


MAP2020 = {"nevus": "nv", "melanoma": "mel", "seborrheic keratosis": "bkl", "lentigo nos": "bkl",
           "solar lentigo": "bkl", "lichenoid keratosis": "bkl"}


def cmd_isic2020(a):
    gt = a.gt if os.path.exists(a.gt) else None
    if gt is None:
        open(a.gt, "wb").write(get(GT2020)); gt = a.gt
    G = pd.read_csv(gt)
    try:   # officially listed duplicate pairs: drop the second image of each pair
        if not os.path.exists("ISIC_2020_Training_Duplicates.csv"):
            open("ISIC_2020_Training_Duplicates.csv", "wb").write(get(DUP2020))
        Dd = pd.read_csv("ISIC_2020_Training_Duplicates.csv")
        drop = set(Dd[Dd.columns[1]].astype(str))
        print(f"official duplicate list: {len(Dd)} pairs; dropping {len(drop & set(G['image_name']))} images")
        G = G[~G["image_name"].isin(drop)].reset_index(drop=True)
    except Exception as e:
        print(f"WARNING: duplicate list not available ({e}); duplicates are checked later by dups.py")
    G["label"] = G["diagnosis"].str.strip().str.lower().map(MAP2020)
    print("diagnosis -> class:"); print(G.groupby("diagnosis")["label"].first().fillna("EXCLUDED").to_string())
    G = G.dropna(subset=["label"]).reset_index(drop=True)
    os.makedirs(a.img_dir, exist_ok=True)
    J = fetch_meta(G["image_name"].tolist())
    print("example record:", json.dumps(J[G["image_name"].iloc[0]])[:1500])
    from PIL import Image

    def dl(name):
        out = os.path.join(a.img_dir, name + ".jpg")
        if os.path.exists(out):
            return out
        j = J.get(name, {})
        f = j.get("files") or {}
        url = (f.get("full") or {}).get("url") if isinstance(f.get("full"), dict) else f.get("full")
        url = url or j.get("url") or (j.get("urls") or {}).get("full")
        if not url:
            return None
        im = Image.open(io.BytesIO(get(url))).convert("RGB")
        s = a.max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        im.save(out, quality=95)
        return out
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(dl, G["image_name"]))
    G["image_path"] = [os.path.abspath(p) if p else None for p in paths]
    miss = G["image_path"].isna().sum()
    G = G.dropna(subset=["image_path"])
    G["group"] = "p_" + G["patient_id"].astype(str)
    G["lesion"] = G["lesion_id"].astype(str) if "lesion_id" in G else G["image_name"]
    G[["image_path", "label", "group", "lesion"]].to_csv(a.out, index=False)
    pd.DataFrame([flatten(J[i]) for i in G["image_name"]]).to_csv("isic2020_metadata.csv", index=False)
    print(f"\nkept {len(G)} images, {G['group'].nunique()} patients, {G['lesion'].nunique()} lesions; {miss} not downloadable")
    print((G["label"].value_counts(normalize=True) * 100).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["meta", "isic2020"])
    ap.add_argument("--csv", default="isic19_noham.csv"); ap.add_argument("--out", default=None)
    ap.add_argument("--gt", default="ISIC_2020_Training_GroundTruth_v2.csv")
    ap.add_argument("--img_dir", default="isic2020_img"); ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="meta: only the first N images (feasibility probe)")
    a = ap.parse_args()
    if a.cmd == "meta":
        a.out = a.out or "isic_metadata.csv"; cmd_meta(a)
    else:
        a.out = a.out or "isic2020_target.csv"; cmd_isic2020(a)


if __name__ == "__main__":
    main()
