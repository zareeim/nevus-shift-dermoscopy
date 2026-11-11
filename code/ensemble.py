import argparse
import json
import os

import numpy as np
import pandas as pd

from adjust import TAUS, adjust
from metrics import full_report, save_json
from sklearn.metrics import balanced_accuracy_score


def _pcols(d):
    return [c for c in d.columns if c.startswith("p_")]


def average(frames):
    base = frames[0][["image_path", "label", "group", "y"]].copy()
    pc = _pcols(frames[0])
    probs = []
    for f in frames:
        f = base[["image_path"]].merge(f, on="image_path", how="left")
        assert f[pc].notna().all().all(), "ensemble members were not evaluated on the same images"
        probs.append(f[pc].values)
    avg = np.mean(probs, axis=0)
    out = base.copy()
    out[pc] = avg
    return out


def cv(a):
    taus = []
    for k in range(5):
        dirs = [os.path.join(a.runs, f"{e}_f{k}") for e in a.exps]
        missing = [d for d in dirs if not os.path.exists(os.path.join(d, "test_predictions.csv"))]
        if missing:
            raise SystemExit(f"missing runs: {missing}")
        v = average([pd.read_csv(os.path.join(d, "val_predictions.csv")) for d in dirs])
        t = average([pd.read_csv(os.path.join(d, "test_predictions.csv")) for d in dirs])
        pc = _pcols(v); classes = [c[2:] for c in pc]
        out = os.path.join(a.runs, f"{a.name}_f{k}")
        os.makedirs(out, exist_ok=True)
        v.to_csv(os.path.join(out, "val_predictions.csv"), index=False)
        t.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
        save_json(full_report(t["y"].values, t[pc].values, classes), os.path.join(out, "test_metrics.json"))

        yv = v["y"].values
        prior = np.bincount(yv, minlength=len(classes)).astype(float) + 1.0
        prior /= prior.sum()
        sc = [balanced_accuracy_score(yv, adjust(v[pc].values, prior, tau).argmax(1)) for tau in TAUS]
        tau = float(TAUS[int(np.argmax(sc))]); taus.append(tau)
        ta = t.copy(); ta[pc] = adjust(t[pc].values, prior, tau)
        ta.to_csv(os.path.join(out, "test_predictions_adj.csv"), index=False)
        m = full_report(t["y"].values, ta[pc].values, classes); m["tau"] = tau
        save_json(m, os.path.join(out, "test_metrics_adj.json"))
        print(f"fold {k}: tau={tau:.2f}  test bacc raw {balanced_accuracy_score(t['y'], t[pc].values.argmax(1)):.4f}"
              f"  adjusted {m['balanced_accuracy']:.4f}")
    final_tau = float(np.median(taus))
    save_json({"tau": final_tau, "per_fold": taus, "members": a.exps},
              os.path.join(a.runs, f"{a.name}_tau.json"))
    print(f"tau for the final model (median of folds) = {final_tau:.2f}")


def official(a):
    tau = json.load(open(a.tau_file))["tau"]
    ham = pd.read_csv(a.train_csv)
    frames, names = [], []
    for d in a.runs:
        frames.append(pd.read_csv(os.path.join(d, "external_predictions.csv")))
        names.append(os.path.basename(d.rstrip("/")))
    pc = _pcols(frames[0]); classes = [c[2:] for c in pc]
    counts = ham["label"].value_counts().reindex(classes).fillna(0).values.astype(float) + 1.0
    prior = counts / counts.sum()
    os.makedirs(a.out, exist_ok=True)
    rows, report = [], {"tau": tau}
    for nm, f in list(zip(names, frames)) + [("ENSEMBLE", average(frames))]:
        y = f["y"].values
        for variant, p in [("raw", f[pc].values), ("adjusted", adjust(f[pc].values, prior, tau))]:
            m = full_report(y, p, classes)
            report[f"{nm}_{variant}"] = m
            rows.append({"model": nm, "variant": variant, "n": m["n"],
                         "balanced_accuracy": m["balanced_accuracy"],
                         "bacc_ci95": m["ci95"]["balanced_accuracy"], "macro_f1": m["macro_f1"],
                         "macro_auc": m["macro_auc"], "accuracy": m["accuracy"],
                         "mel_sens@spec80": m.get("mel_sens@spec80"), "recall": m["recall_per_class"]})
            if nm == "ENSEMBLE":
                out = f.copy(); out[pc] = p
                out.to_csv(os.path.join(a.out, f"ensemble_{variant}_predictions.csv"), index=False)
    save_json(report, os.path.join(a.out, "official_metrics.json"))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(f"ISIC 2018 Task 3 official test set (tau={tau:.2f}, chosen by cross-validation)\n")
    print(tab.drop(columns=["recall"]).round(4).to_string(index=False))
    print("\nper-class recall (ENSEMBLE, adjusted):")
    print({k: round(v, 3) for k, v in report["ENSEMBLE_adjusted"]["recall_per_class"].items()})
    tab.to_csv(os.path.join(a.out, "official_table.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--runs", required=True); c.add_argument("--exps", nargs="+", required=True)
    c.add_argument("--name", default="ham_ensemble")
    o = sub.add_parser("official")
    o.add_argument("--runs", nargs="+", required=True); o.add_argument("--tau_file", required=True)
    o.add_argument("--train_csv", required=True); o.add_argument("--out", default="official_results")
    a = ap.parse_args()
    cv(a) if a.cmd == "cv" else official(a)


if __name__ == "__main__":
    main()
