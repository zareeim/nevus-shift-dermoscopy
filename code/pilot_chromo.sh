#!/usr/bin/env bash
set -euo pipefail
[ -f ham_masks.csv ] || { echo "ham_masks.csv missing - run add_masks.py first"; exit 1; }
for F in 0 1; do
  for C in baseline od_ref chromo full; do
    OUT=runs_chromo/${C}_f$F
    [ -f $OUT/val_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data ham_masks.csv --fold $F --config $C --out $OUT
  done
  python chromo.py check --run runs_chromo/full_f$F
done

python - <<'PY' | tee pilot_results.txt
import json
def bacc(c, f): return json.load(open(f"runs_chromo/{c}_f{f}/val_metrics.json"))["balanced_accuracy"]
print("validation balanced accuracy")
print(f"{'config':10s} {'fold0':>8s} {'fold1':>8s} {'mean':>8s}")
for c in ["baseline", "od_ref", "chromo", "full"]:
    v = [bacc(c, f) for f in (0, 1)]
    print(f"{c:10s} {v[0]:8.4f} {v[1]:8.4f} {sum(v)/2:8.4f}")
a = all(bacc("full", f) - bacc("baseline", f) >= 0.015 for f in (0, 1))
b = all(bacc("full", f) > bacc("od_ref", f) for f in (0, 1))
c = all(json.load(open(f"runs_chromo/full_f{f}/physics_check.json"))["passed"] for f in (0, 1))
print(f"\n(a) full - baseline >= 1.5 points on both folds : {a}")
print(f"(b) full > od_ref on both folds                  : {b}")
print(f"(c) physics sanity check passed on both folds    : {c}")
print("\nDECISION:", "GO - continue to the full study" if a and b and c else "NO-GO - stop this idea")
PY
notify-send "Skin" "chromophore pilot finished - see pilot_results.txt" || true
