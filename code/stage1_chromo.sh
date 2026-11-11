#!/usr/bin/env bash
set -euo pipefail
declare -A TAG=( [efficientnet_b3]=effb3 [convnext_tiny]=convnext )

echo "=== 1) train EfficientNet-B3 and ConvNeXt-T: baseline and full, 5 folds ==="
for BB in efficientnet_b3 convnext_tiny; do
  for F in 0 1 2 3 4; do
    for C in baseline full; do
      OUT=runs_chromo_cv/ham_${TAG[$BB]}_${C}_f$F
      [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
      python chromo.py train --data ham_masks.csv --fold $F --config $C --backbone $BB --bs 16 \
          --pred_masks seg/f$F/pred_masks.csv --eval_test --out $OUT
    done
  done
done

echo "=== 2) TTA predictions for all 30 runs (no retraining) ==="
for R in runs_chromo_cv/ham_{baseline,full,effb3_baseline,effb3_full,convnext_baseline,convnext_full}_f*; do
  OUT=runs_chromo_tta/$(basename $R)
  [ -f $OUT/test_metrics.json ] && continue
  python chromo.py predict --run $R --tta --out $OUT
done

echo "=== 3) ensembles of the three backbones ==="
python ensemble.py cv --runs runs_chromo_tta --name ham_ens_full --exps ham_full ham_effb3_full ham_convnext_full
python ensemble.py cv --runs runs_chromo_tta --name ham_ens_baseline --exps ham_baseline ham_effb3_baseline ham_convnext_baseline

echo "=== 4) decision rules chosen on validation (same for every model) ==="
python thresholds.py --runs runs_chromo_tta --objective bacc
python thresholds.py --runs runs_chromo_tta --objective f1

cmp () { python summarize.py --runs runs_chromo_tta "${@:3}" --compare $1 $2 | sed -n '/minus/,$p'; }
{
  echo "##### all models, TTA, raw predictions (5-fold test, automatic masks)"
  python summarize.py --runs runs_chromo_tta
  for V in "" bacc f1; do
    if [ -z "$V" ]; then ARGS=(); echo; echo "################ RAW"; else
      ARGS=(--file test_metrics_$V.json --pred_file test_predictions_$V.csv); echo; echo "################ decision rule: $V"
      python summarize.py --runs runs_chromo_tta "${ARGS[@]}"; fi
    for P in "ham_full ham_baseline" "ham_effb3_full ham_effb3_baseline" "ham_convnext_full ham_convnext_baseline" "ham_ens_full ham_ens_baseline"; do
      cmp $P "${ARGS[@]}"
    done
  done
  echo DONE
} > stage1_results.txt 2>&1
notify-send "Skin" "stage 1 finished - see stage1_results.txt" || true
