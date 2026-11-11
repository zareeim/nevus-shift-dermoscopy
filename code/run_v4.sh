#!/usr/bin/env bash
set -euo pipefail
[ -f isic2018_test.csv ] || { echo "isic2018_test.csv missing - prepare it with data.py isic2018test"; exit 1; }
COMMON="--data ham.csv --img_size 320 --color_constancy --epochs 30 --patience 0 --select last --ema 0.999 --tta"
done_run () { [ -f "$1/test_metrics.json" ] || [ -f "$1/external_predictions.csv" ]; }

echo "=== 1) 5-fold CV of the two new backbones (ResNet-50 is reused from runs_v3) ==="
for BB in efficientnet_b3 convnext_tiny; do
  for F in 0 1 2 3 4; do
    OUT=runs_v3/ham_${BB}_baseline_f$F
    done_run $OUT && { echo "skip $OUT"; continue; }
    python train.py $COMMON --backbone $BB --bs 16 --fold $F --model baseline --out $OUT
  done
done

echo "=== 2) ensemble in cross-validation (+ tau) ==="
EXPS="ham_resnet50_baseline ham_efficientnet_b3_baseline ham_convnext_tiny_baseline"
python ensemble.py cv --runs runs_v3 --name ham_ensemble3 --exps $EXPS
python adjust.py --runs runs_v3 > /dev/null
{
  echo "##### 5-fold CV on HAM10000 (lesion-grouped), RAW"
  python summarize.py --runs runs_v3
  echo; echo "##### 5-fold CV, AFTER logit adjustment"
  python summarize.py --runs runs_v3 --file test_metrics_adj.json
  echo; echo "##### ensemble vs ResNet-50 (adjusted, paired bootstrap)"
  python summarize.py --runs runs_v3 --file test_metrics_adj.json --pred_file test_predictions_adj.csv \
      --compare ham_ensemble3 ham_resnet50_baseline | sed -n '/minus/,$p'
} > cv_results_v4.txt 2>&1

echo "=== 3) final models on ALL of HAM10000 ==="
for BB in resnet50 efficientnet_b3 convnext_tiny; do
  OUT=runs_full/$BB
  done_run $OUT && { echo "skip $OUT"; continue; }
  BS=16; [ $BB = resnet50 ] && BS=32
  python train.py $COMMON --backbone $BB --bs $BS --model baseline --train_all \
      --predict_csv isic2018_test.csv --out $OUT
done

echo "=== 4) official ISIC 2018 test set (evaluated once) ==="
python ensemble.py official --runs runs_full/resnet50 runs_full/efficientnet_b3 runs_full/convnext_tiny \
    --tau_file runs_v3/ham_ensemble3_tau.json --train_csv ham.csv --out official_results \
    > official_results_v4.txt 2>&1
echo DONE >> official_results_v4.txt
notify-send "Skin" "v4 finished - see cv_results_v4.txt and official_results_v4.txt" || true
