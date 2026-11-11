import itertools, os
import numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def feats(p):
    im = Image.open(p).convert("RGB")
    g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), np.int16)
    bits = np.packbits((g[:, 1:] > g[:, :-1]).flatten())
    th = np.asarray(im.resize((32, 32), Image.BILINEAR), np.float32).flatten()
    return bits, th

def load(paths):
    with ProcessPoolExecutor() as ex:
        r = list(ex.map(feats, paths, chunksize=64))
    return np.stack([x[0] for x in r]), np.stack([x[1] for x in r])

POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def main():
    sets = {"HAM10000": pd.read_csv("ham.csv")["image_path"].tolist(),
            "ISIC2019 non-HAM": pd.read_csv("isic19_noham.csv")["image_path"].tolist(),
            "Derm7pt": pd.read_csv("derm7pt.csv")["image_path"].tolist()}
    if os.path.exists("isic2020_target.csv"):
        sets["ISIC2020"] = pd.read_csv("isic2020_target.csv")["image_path"].tolist()
    F = {k: load(v) for k, v in sets.items()}
    rows = []
    for a, b in itertools.combinations(sets, 2):
        (Ha, Ta), (Hb, Tb) = F[a], F[b]
        cand = 0
        for s in range(0, len(Ha), 128):
            d = POP[np.bitwise_xor(Ha[s:s + 128, None, :], Hb[None, :, :])].sum(2)
            ii, jj = np.nonzero(d <= 20)                      # stage 1: 256-bit hash, <= 20 bits differ
            cand += len(ii)
            for i, j in zip(ii, jj):
                mae = np.abs(Ta[s + i] - Tb[j]).mean()        # stage 2: colour thumbnails must match
                rows.append((a, sets[a][s + i], b, sets[b][j], int(d[i, j]), round(float(mae), 2)))
        print(f"{a} vs {b}: {cand} hash candidates")
    D = pd.DataFrame(rows, columns=["set_a", "image_a", "set_b", "image_b", "hash_bits", "thumb_mae"])
    D.to_csv("dup_candidates.csv", index=False)
    dup = D[D["thumb_mae"] < 6]
    print("\nconfirmed near-duplicates (hash <= 20/256 bits AND mean colour difference < 6/255):")
    print(dup.groupby(["set_a", "set_b"]).size().to_string() if len(dup) else "none")
    show = D.sort_values("thumb_mae").head(12)
    if len(show):
        grid = Image.new("RGB", (400, 200 * len(show)))
        for i, r in enumerate(show.itertuples()):
            for j, p in enumerate((r.image_a, r.image_b)):
                grid.paste(Image.open(p).convert("RGB").resize((200, 200)), (200 * j, 200 * i))
        grid.save("dup_examples2.jpg", quality=85)
        print("12 closest candidates -> dup_examples2.jpg (left / right), thumb_mae:", show["thumb_mae"].tolist())
    print("DONE")


if __name__ == "__main__":
    main()
