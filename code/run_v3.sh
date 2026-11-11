#!/usr/bin/env bash
set -euo pipefail
COMMON="--data ham.csv --backbone resnet50 --img_size 320 --bs 32 --color_constancy --epochs 30 --patience 0 --select last --ema 0.999"

echo "=== 1) tuning on fold-0 validation ==="
python train.py $COMMON --fold 0 --no_test --model baseline --out runs_tune_v3/baseline
for ETA in 1 2; do
  for AUX in 0.3 1.0; do
    python train.py $COMMON --fold 0 --no_test --main_head gap --eta $ETA --aux_w $AUX \
        --out runs_tune_v3/hgpb_eta${ETA}_aux${AUX}
  done
done
python summarize.py --runs runs_tune_v3 --file val_metrics.json | tee tune_results_v3.txt

echo "=== 2) choose HGPB setting by validation balanced accuracy ==="
read ETA AUX < <(python - <<'PY'
import glob, json, re
best = max(glob.glob("runs_tune_v3/hgpb_*/val_metrics.json"),
           key=lambda f: json.load(open(f))["balanced_accuracy"])
eta, aux = re.search(r"eta([\d.]+)_aux([\d.]+)", best).groups()
print(eta, aux)
PY
)
echo "chosen: eta=$ETA aux_w=$AUX" | tee -a tune_results_v3.txt

echo "=== 3) confirmatory 5-fold run ==="
for F in 0 1 2 3 4; do
  python train.py $COMMON --tta --fold $F --model baseline --out runs_v3/ham_resnet50_baseline_f$F
  python train.py $COMMON --tta --fold $F --main_head gap --eta $ETA --aux_w $AUX --out runs_v3/ham_resnet50_hgpb_f$F
done

echo "=== 4) logit adjustment + comparison ==="
python adjust.py --runs runs_v3
{
  echo "##### chosen setting: eta=$ETA aux_w=$AUX"
  echo; echo "##### RAW predictions"
  python summarize.py --runs runs_v3 --compare ham_resnet50_hgpb ham_resnet50_baseline
  echo; echo "##### AFTER logit adjustment (tau chosen on validation)"
  python summarize.py --runs runs_v3 --file test_metrics_adj.json --pred_file test_predictions_adj.csv \
      --compare ham_resnet50_hgpb ham_resnet50_baseline
} > final_results_v3.txt 2>&1
echo "DONE" >> final_results_v3.txt
notify-send "HGPB" "v3 finished - see final_results_v3.txt" || true
