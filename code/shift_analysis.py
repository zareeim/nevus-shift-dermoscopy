import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from metrics import expected_calibration_error


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pc = [c for c in d.columns if c.startswith("p_")]
    return d, pc, [c[2:] for c in pc]


def scores(y, p, K):
    pr = p.argmax(1)
    return {"bacc": balanced_accuracy_score(y, pr),
            "macroF1": f1_score(y, pr, average="macro", labels=list(range(K)), zero_division=0),
            "acc": float((pr == y).mean())}


def reweight(p, w):
    q = p * w
    return q / q.sum(1, keepdims=True)


def em_prior(p, prior_train, iters=200, tol=1e-8):
    """Saerens, Latinne & Decaestecker (2002): EM estimate of the target prior from unlabeled data."""
    pi = prior_train.copy()
    for _ in range(iters):
        q = reweight(p, pi / prior_train)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    return pi


def image_sizes(paths):
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                out.append(f"{im.size[0]}x{im.size[1]}")
        except Exception:
            out.append("unreadable")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ind", default="runs_aug/ham_none_f*/test_predictions.csv")
    ap.add_argument("--ood", default="runs_ood/ham_none/test_predictions.csv")
    ap.add_argument("--train_csv", default="ham.csv")
    ap.add_argument("--meta", default=None, help="ISIC_2019_Training_Metadata.csv (optional)")
    ap.add_argument("--min_group", type=int, default=150)
    ap.add_argument("--out", default="shift_results.txt")
    a = ap.parse_args()

    I, pc, classes = load(a.ind)
    O, pc2, classes2 = load(a.ood)
    assert classes == classes2, "class order differs"
    K = len(classes)
    yI, pI, yO, pO = I["y"].values, I[pc].values, O["y"].values, O[pc].values
    ham = pd.read_csv(a.train_csv)
    prior_tr = ham["label"].value_counts().reindex(classes).fillna(0).values.astype(float) + 1
    prior_tr /= prior_tr.sum()
    prior_ood = np.bincount(yO, minlength=K) / len(yO)
    # the model was trained with class weights and label smoothing, so its outputs are NOT calibrated
    # to the raw training prior; the prior its probabilities actually encode is their mean on
    # in-distribution data. This "effective source prior" is used for all corrections.
    prior_src = pI.mean(0); prior_src = prior_src / prior_src.sum()

    L = []
    P = L.append
    P(f"in-distribution (HAM10000, 5-fold out-of-fold): {len(I)} images | OOD (ISIC 2019 non-HAM): {len(O)} images\n")

    # 1. distributions
    pred_ood = np.bincount(pO.argmax(1), minlength=K) / len(yO)
    t = pd.DataFrame({"train_prior_%": prior_tr * 100, "model_effective_prior_%": prior_src * 100,
                      "OOD_true_%": prior_ood * 100,
                      "OOD_predicted_%": pred_ood * 100}, index=classes)
    P("== 1. class distribution ==")
    P(t.round(1).to_string())
    tv = 0.5 * np.abs(prior_tr - prior_ood).sum()
    P(f"total-variation distance train vs OOD prior: {tv:.3f}\n")

    # 2. per-class recall and AUC
    rows = []
    for k, c in enumerate(classes):
        r = {"class": c}
        for nm, y, p in (("IND", yI, pI), ("OOD", yO, pO)):
            m = y == k
            r[f"{nm}_n"] = int(m.sum())
            r[f"{nm}_recall"] = float((p[m].argmax(1) == k).mean()) if m.any() else np.nan
            r[f"{nm}_AUC"] = float(roc_auc_score(m.astype(int), p[:, k])) if 0 < m.sum() < len(y) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index("class")
    t["recall_drop"] = t["IND_recall"] - t["OOD_recall"]
    t["AUC_drop"] = t["IND_AUC"] - t["OOD_AUC"]
    P("== 2. per-class recall and one-vs-rest AUC ==")
    P(t.round(3).to_string())
    P(f"mean recall drop {t['recall_drop'].mean():.3f} | mean AUC drop {t['AUC_drop'].mean():.3f}")
    P("(large recall drop with small AUC drop -> decision rule / prior problem; large AUC drop -> features do not transfer)\n")

    # 3. confusion
    cm = confusion_matrix(yO, pO.argmax(1), labels=list(range(K))).astype(float)
    cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    P("== 3. OOD confusion matrix, row-normalised (rows = true, columns = predicted) ==")
    P(pd.DataFrame(cm, index=classes, columns=classes).round(2).to_string() + "\n")

    # 4. decision-rule corrections
    pi_em = em_prior(pO, prior_src)
    variants = {
        "raw (as trained)": pO,
        "balanced rule p/prior_src (no labels)": reweight(pO, 1 / prior_src),
        "EM-estimated OOD prior (no labels)": reweight(pO, pi_em / prior_src),
        "ORACLE true OOD prior (upper bound)": reweight(pO, prior_ood / prior_src),
    }
    P("== 4. decision-rule corrections on OOD ==")
    rr = [{"rule": k, **scores(yO, v, K)} for k, v in variants.items()]
    P(pd.DataFrame(rr).set_index("rule").round(4).to_string())
    P("EM prior estimate vs true OOD prior (%):")
    P(pd.DataFrame({"EM": pi_em * 100, "true": prior_ood * 100}, index=classes).round(1).to_string())
    P("in-distribution reference: " + ", ".join(f"{k} {v:.4f}" for k, v in scores(yI, pI, K).items())
      + " | balanced rule: " + ", ".join(f"{k} {v:.4f}" for k, v in scores(yI, reweight(pI, 1 / prior_src), K).items()) + "\n")

    # 5. subgroups
    P("== 5. OOD performance by subgroup (groups with >= %d images) ==" % a.min_group)
    O["size"] = image_sizes(O["image_path"])
    O["downsampled"] = O["image_path"].str.contains("_downsampled")
    if a.meta and os.path.exists(a.meta):
        m = pd.read_csv(a.meta)
        m["image_id"] = m["image"].astype(str)
        O["image_id"] = O["image_path"].map(lambda s: os.path.splitext(os.path.basename(s))[0].replace("_downsampled", ""))
        O = O.merge(m[["image_id", "age_approx", "anatom_site_general", "sex"]], on="image_id", how="left")
        O["age_group"] = pd.cut(O["age_approx"], [0, 30, 45, 60, 75, 120], labels=["<30", "30-44", "45-59", "60-74", "75+"])
    sub_cols = ["size", "downsampled"] + [c for c in ("anatom_site_general", "age_group", "sex") if c in O]
    for col in sub_cols:
        rows = []
        for g, d in O.groupby(col, observed=True, dropna=False):
            if len(d) < a.min_group:
                continue
            yy, pp = d["y"].values, d[pc].values
            s = scores(yy, pp, K)
            rows.append({col: g, "n": len(d), "bacc": s["bacc"], "acc": s["acc"],
                         "%mel": (yy == classes.index("mel")).mean() * 100 if "mel" in classes else np.nan,
                         "%nv": (yy == classes.index("nv")).mean() * 100 if "nv" in classes else np.nan})
        if rows:
            P(pd.DataFrame(rows).sort_values("n", ascending=False).round(3).to_string(index=False) + "\n")

    # 6. confidence
    P("== 6. confidence ==")
    for nm, y, p in (("IND", yI, pI), ("OOD", yO, pO)):
        P(f"{nm}: mean max-probability {p.max(1).mean():.3f} | ECE {expected_calibration_error(p, y):.3f}")

    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\nDONE\n")
    print(txt)


if __name__ == "__main__":
    main()
