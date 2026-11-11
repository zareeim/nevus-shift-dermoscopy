#!/usr/bin/env bash
set -euo pipefail
[ -f ham_masks.csv ] || { echo "ham_masks.csv missing"; exit 1; }
for F in 0 1 2 3 4; do
  [ -f seg/f$F/pred_masks.csv ] || python seg.py --data ham_masks.csv --fold $F --out seg/f$F
  for C in baseline od_ref chromo full; do
    OUT=runs_chromo_cv/ham_${C}_f$F
    [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data ham_masks.csv --fold $F --config $C \
        --pred_masks seg/f$F/pred_masks.csv --eval_test --out $OUT
  done
  [ -f runs_chromo_cv/ham_full_f$F/physics_check.json ] || python chromo.py check --run runs_chromo_cv/ham_full_f$F
done
{
  echo "##### segmentation quality (automatic vs dermatologist masks, val+test of each fold)"
  for F in 0 1 2 3 4; do echo "fold $F: $(cat seg/f$F/seg_metrics.json | tr -d '\n ')"; done
  echo; echo "##### 5-fold results (test folds, automatic masks)"
  python summarize.py --runs runs_chromo_cv
  echo; echo "##### PRIMARY: full minus baseline"
  python summarize.py --runs runs_chromo_cv --compare ham_full ham_baseline | sed -n '/minus/,$p'
  echo; echo "##### full minus od_ref (does the physics matter?)"
  python summarize.py --runs runs_chromo_cv --compare ham_full ham_od_ref | sed -n '/minus/,$p'
  echo; echo "##### full minus chromo (does the self-reference matter?)"
  python summarize.py --runs runs_chromo_cv --compare ham_full ham_chromo | sed -n '/minus/,$p'
  echo; echo "##### physics sanity check per fold"
  for F in 0 1 2 3 4; do echo "fold $F: $(cat runs_chromo_cv/ham_full_f$F/physics_check.json | tr -d '\n ')"; done
  echo DONE
} > study_results.txt 2>&1
notify-send "Skin" "chromophore study finished - see study_results.txt" || true
