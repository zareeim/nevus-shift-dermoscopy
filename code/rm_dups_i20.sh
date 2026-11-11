#!/usr/bin/env bash
set -euo pipefail
python - <<'EOF'
import pandas as pd
D = pd.read_csv("dup_candidates.csv")
D = D[D["thumb_mae"] < 6]                      # confirmed near-duplicates only (as in dups.py)
is19 = lambda s: s.str.startswith("ISIC2019")    # set name in dups.py is "ISIC2019 non-HAM"
pair = D[(is19(D.set_a) & (D.set_b == "ISIC2020")) | ((D.set_a == "ISIC2020") & is19(D.set_b))]
drop = set(pair.loc[pair.set_a == "ISIC2020", "image_a"]) | set(pair.loc[pair.set_b == "ISIC2020", "image_b"])
T = pd.read_csv("isic2020_target.csv")
keep = T[~T["image_path"].isin(drop)]
print(f"ISIC2019-ISIC2020 pairs: {len(pair)}; ISIC2020 images removed: {len(T) - len(keep)}; kept: {len(keep)}")
assert len(pair) > 0 and len(T) > len(keep), "no duplicate removed - check dup_candidates.csv"
keep.to_csv("isic2020_target_nodup.csv", index=False)
EOF
rm -rf runs_ood_i20_nodup
E=bcn_baseline
python ood_perfold.py --data isic2020_target_nodup.csv --runs runs_chromo_cv --exp $E --out runs_ood_i20_nodup
python safety.py --exp $E --ood runs_ood_i20_nodup --sources none --out i20_nodup_safety_$E.txt > /dev/null
python extra_analysis.py --ood runs_ood_i20_nodup --exps $E --prefix i20nodup_ --summary i20_nodup_extra_summary.txt > /dev/null
python matched.py --exps $E --targets ISIC2020=runs_ood_i20_nodup --out matched_results_i20_nodup.txt > /dev/null
echo "n_target: $(( $(wc -l < isic2020_target_nodup.csv) - 1 ))" > i20_nodup_n.txt
cat i20_nodup_n.txt i20_nodup_extra_summary.txt matched_results_i20_nodup.txt i20_nodup_safety_$E.txt > i20_nodup_all.txt
echo "STEP DONE: i20_nodup_all.txt"
