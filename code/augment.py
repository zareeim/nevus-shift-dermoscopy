import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from chromo import CHROMO_NAMES, SPECTRA, srgb_to_linear
from train import MEAN, STD


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


class _Norm:
    def __init__(self, device):
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def to_rgb(self, x):
        return (x.float() * self.std + self.mean).clamp(0, 1)

    def to_norm(self, rgb):
        return (rgb - self.mean) / self.std


class ChromoAug:
    def __init__(self, device, separate=True, p=0.8, mel=(-0.3, 2.0), hb_skin=(0.0, 0.4),
                 shade=(-0.3, 0.3), contrast=(0.85, 1.15), eps=1e-3):
        M = torch.tensor([SPECTRA[k] for k in CHROMO_NAMES], dtype=torch.float32).T  # 3x4
        M = M / M.norm(dim=0, keepdim=True)
        self.M, self.W = M.to(device), torch.linalg.pinv(M).to(device)
        self.n, self.separate, self.p, self.eps = _Norm(device), separate, p, eps
        self.r = dict(mel=mel, hb=hb_skin, shade=shade, k=contrast)

    def _u(self, B, lo_hi, device):
        lo, hi = lo_hi
        return (torch.rand(B, device=device) * (hi - lo) + lo).view(B, 1, 1)

    @torch.no_grad()
    def __call__(self, x, masks):
        B, dev = x.shape[0], x.device
        rgb = self.n.to_rgb(x)
        od = -torch.log(srgb_to_linear(rgb).clamp_min(self.eps))
        c = torch.einsum("kc,bchw->bkhw", self.W, od)
        resid = od - torch.einsum("ck,bkhw->bchw", self.M, c)  # ~0; keeps reconstruction exact
        d_mel, d_shade = self._u(B, self.r["mel"], dev), self._u(B, self.r["shade"], dev)
        d_hb = self._u(B, self.r["hb"], dev)
        cn = c.clone()
        if self.separate:
            les = masks[:, :1].float()
            valid = masks[:, 1:2].float()
            # wide, smooth lesion zone: automatic masks are slightly too large or too small, and a
            # sharp lesion/skin switch creates an artificial halo around the lesion
            soft = F.avg_pool2d(F.max_pool2d(les, 9, 1, 4), 31, 1, 15, count_include_pad=False)
            skin = (1 - les) * valid
            skin_mean = (c * skin).sum((2, 3)) / skin.sum((2, 3)).clamp_min(1.0)
            k = self._u(B, self.r["k"], dev).unsqueeze(1)               # (B,1,1,1)
            dev_les = (c - skin_mean[..., None, None])[:, :3]           # melanins + haemoglobin
            cn[:, :3] = cn[:, :3] + (k - 1) * dev_les * soft
            cn[:, 2] = cn[:, 2] + d_hb * (1 - soft[:, 0])               # skin erythema only
        else:
            cn[:, 2] = cn[:, 2] + d_hb                                  # global
        cn[:, 0] = cn[:, 0] + d_mel                                     # constitutive skin tone
        cn[:, 3] = cn[:, 3] + d_shade                                   # illumination
        od_n = torch.einsum("ck,bkhw->bchw", self.M, cn) + resid
        out = self.n.to_norm(linear_to_srgb(torch.exp(-od_n)))
        keep = (torch.rand(B, device=dev) >= self.p).view(B, 1, 1, 1)
        return torch.where(keep, x, out.to(x.dtype))


class Galdran:
    def __init__(self, device, paths, n=800, p_norm=6, p=0.8, gamma_sd=0.1, seed=0):
        rng = random.Random(seed)
        paths = list(paths); rng.shuffle(paths)
        pool = []
        for pth in paths[:n]:
            a = np.asarray(Image.open(pth).convert("RGB").resize((160, 120)), dtype=np.float64) + 1.0
            e = np.power(np.mean(np.power(a, p_norm), axis=(0, 1)), 1.0 / p_norm)
            pool.append(e / np.linalg.norm(e) * math.sqrt(3))
        self.pool = torch.tensor(np.stack(pool), dtype=torch.float32, device=device)
        self.n, self.p, self.gsd = _Norm(device), p, gamma_sd

    @torch.no_grad()
    def __call__(self, x, masks=None):
        B, dev = x.shape[0], x.device
        rgb = self.n.to_rgb(x)
        e = self.pool[torch.randint(len(self.pool), (B,), device=dev)].view(B, 3, 1, 1)
        g = (1 + self.gsd * torch.randn(B, device=dev)).clamp(0.7, 1.4).view(B, 1, 1, 1)
        out = self.n.to_norm((rgb * e).clamp(0, 1) ** g)
        keep = (torch.rand(B, device=dev) >= self.p).view(B, 1, 1, 1)
        return torch.where(keep, x, out.to(x.dtype))


def build(name, device, train_paths):
    if name == "none":
        return None
    if name == "galdran":
        return Galdran(device, train_paths)
    if name == "chromo":
        return ChromoAug(device, separate=True)
    if name == "chromo_global":
        return ChromoAug(device, separate=False)
    raise ValueError(name)


def preview(a):
    """Grid: each row = one image; columns = original, 4x ChromoAug, 2x Galdran."""
    import pandas as pd
    from chromo import JointTransform
    df = pd.read_csv(a.data).sample(a.n, random_state=a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = JointTransform(a.size, False)
    xs, ms = [], []
    for _, r in df.iterrows():
        img = Image.open(r["image_path"]).convert("RGB")
        mask = Image.open(r["mask_path"]).convert("L").resize(img.size, Image.NEAREST)
        x, m = tf(img, mask); xs.append(x); ms.append(m)
    x, m = torch.stack(xs).to(dev), torch.stack(ms).to(dev)
    torch.manual_seed(a.seed)
    ca = ChromoAug(dev, p=1.0); ga = Galdran(dev, df["image_path"], n=len(df), p=1.0)
    nrm = _Norm(dev)
    cols = [x] + [ca(x, m) for _ in range(4)] + [ga(x) for _ in range(2)]
    rows = torch.cat([torch.cat([nrm.to_rgb(c)[i] for c in cols], 2) for i in range(len(df))], 1)
    Image.fromarray((rows.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(a.out)
    print(f"saved {a.out}: columns = original | ChromoAug x4 | Galdran x2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ham_masks.csv"); ap.add_argument("--out", default="aug_examples.jpg")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    preview(ap.parse_args())
