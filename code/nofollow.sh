#!/usr/bin/env bash
set -uo pipefail
HM=$(find ~/workspace -name "HAM10000_metadata*.csv" 2>/dev/null | head -1)
[ -n "$HM" ] || { echo "HAM10000 metadata not found"; exit 1; }
python - "$HM" <<'EOF' || exit 1
import os, sys, pandas as pd
hm = pd.read_csv(sys.argv[1]); d = pd.read_csv("ham_masks.csv")
assert {"label", "fold", "image_path"} <= set(d.columns), f"unexpected columns: {list(d.columns)}"
assert (d["label"] == "nv").any(), f"labels are not class names: {d['label'].unique()[:7]}"
d["image_id"] = d["image_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
d = d.merge(hm[["image_id", "dx_type"]], on="image_id", how="left")
assert d["dx_type"].notna().all(), "some images have no dx_type"
drop = (d["label"] == "nv") & (d["dx_type"] == "follow_up")
print(f"all images {len(d)}; nevi {int((d['label'] == 'nv').sum())}; follow-up nevi removed {int(drop.sum())}")
k = d[~drop].drop(columns=["image_id", "dx_type"])
print("remaining class counts:"); print(k["label"].value_counts().to_string())
print("images per outer fold:"); print(k["fold"].value_counts().sort_index().to_string())
k.to_csv("ham_nofollow.csv", index=False)
EOF
E=ham_nofollow
for F in 0 1 2 3 4; do
  OUT=runs_chromo_cv/${E}_f$F
  [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
  python chromo.py train --data ham_nofollow.csv --fold $F --config baseline --seed 42 --eval_test --out $OUT || exit 1
done
python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp $E --out runs_ood_pf
python ood_perfold.py --data derm7pt.csv --runs runs_chromo_cv --exp $E --out runs_ood_d7
python matched.py --exps $E ham_baseline --targets ISIC2019=runs_ood_pf Derm7pt=runs_ood_d7 --n_boot 2000 --out matched_nofollow.txt > /dev/null
python - <<'EOF' > nofollow_all.txt
print(open("matched_nofollow.txt").read().split("===== SUMMARY")[1] if "===== SUMMARY" in open("matched_nofollow.txt").read() else "no summary")
import re
txt = open("matched_nofollow.txt").read()
for block in txt.split("===== ")[1:]:
    if "|" in block.split("\n")[0]:
        head = block.split("\n")[0]
        ci = block.split("-- matched fold-averaged differences")[1].split("-- M2")[0] if "-- matched fold-averaged differences" in block else ""
        print("=====", head); print(ci)
EOF
notify-send "Skin" "no-follow-up model finished" 2>/dev/null || true
echo "STEP DONE: nofollow_all.txt"
