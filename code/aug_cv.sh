#!/usr/bin/env bash
set -euo pipefail
python augment.py --data ham_masks.csv --out aug_examples.jpg     # look at this image first!
mkdir -p runs_aug
for F in 0 1 2 3 4; do
  [ -e runs_aug/ham_none_f$F ] || ln -s "$(pwd)/runs_chromo_cv/ham_baseline_f$F" runs_aug/ham_none_f$F
  for A in galdran chromo chromo_global; do
    OUT=runs_aug/ham_${A}_f$F
    [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data ham_masks.csv --fold $F --config baseline --aug $A \
        --pred_masks seg/f$F/pred_masks.csv --eval_test --out $OUT
  done
done
{
  echo "##### in-distribution (HAM10000, 5-fold test, lesion-grouped) - SECONDARY"
  python summarize.py --runs runs_aug
  for P in "ham_chromo ham_galdran" "ham_chromo ham_none" "ham_chromo ham_chromo_global" "ham_galdran ham_none"; do
    python summarize.py --runs runs_aug --compare $P | sed -n '/minus/,$p'
  done
  echo DONE
} > aug_cv_results.txt 2>&1
notify-send "Skin" "stage A finished - see aug_cv_results.txt" || true
