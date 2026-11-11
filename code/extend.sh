#!/usr/bin/env bash
set -euo pipefail
PART=${1:-all}
note() { notify-send "Skin" "$1" 2>/dev/null || true; echo ">>> $1"; }

train_ham() {  # $1 exp name, $2 backbone, $3 extra args
  for F in 0 1 2 3 4; do
    OUT=runs_chromo_cv/$1_f$F
    [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data ham_masks.csv --fold $F --config baseline --backbone $2 $3 \
        --pred_masks seg/f$F/pred_masks.csv --eval_test --out $OUT
  done
}

analyse_fwd() {  # $1 exp name, $2 tag
  python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp $1 --out runs_ood_pf
  [ -f label_shift_results_$2.txt ] || python label_shift.py --runs runs_chromo_cv --exp $1 --ood runs_ood_pf --out label_shift_results_$2.txt
  [ -f safety_results_$1.txt ] || python safety.py --exp $1 > /dev/null
}

if [ "$PART" = smoke ]; then
  rm -rf runs_smoke_swin
  python chromo.py train --data ham_masks.csv --fold 0 --config baseline --backbone swin_t --bs 16 --lr 1e-4 \
      --epochs 1 --max_batches 5 --pred_masks seg/f0/pred_masks.csv --out runs_smoke_swin
  echo "SMOKE OK"; exit 0
fi

if [ "$PART" = all ] || [ "$PART" = convnext ]; then
  echo "=== E1a ConvNeXt-T (already trained) ==="
  ls runs_chromo_cv/ham_convnext_baseline_f4/test_predictions.csv > /dev/null
  analyse_fwd ham_convnext_baseline convnext
  note "ConvNeXt analysis finished"
fi

if [ "$PART" = all ] || [ "$PART" = reverse ]; then
  echo "=== E2 reverse direction: BCN-like -> HAM10000 (ResNet-50) ==="
  [ -f bcn_masks.csv ] || python prep_reverse.py | tee reverse_data.txt
  for F in 0 1 2 3 4; do
    OUT=runs_chromo_cv/bcn_baseline_f$F
    [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data bcn_masks.csv --fold $F --config baseline --eval_test --out $OUT
  done
  python ood_perfold.py --data ham_ood.csv --runs runs_chromo_cv --exp bcn_baseline --out runs_ood_pf
  [ -f label_shift_results_bcn2ham.txt ] || python label_shift.py --runs runs_chromo_cv --exp bcn_baseline --ood runs_ood_pf --out label_shift_results_bcn2ham.txt
  python adapt.py run --data ham_ood.csv --exp bcn_baseline --out runs_adapt_bcn
  [ -f adapt_results_bcn2ham.txt ] || python adapt.py analyze --exp bcn_baseline --out runs_adapt_bcn --results adapt_results_bcn2ham.txt
  [ -f safety_results_bcn_baseline.txt ] || python safety.py --exp bcn_baseline --sources runs_adapt_bcn/sources.csv > /dev/null
  note "reverse direction finished"
fi

if [ "$PART" = all ] || [ "$PART" = swin ]; then
  echo "=== E1b Swin-T ==="
  train_ham ham_swin_baseline swin_t "--bs 16 --lr 1e-4"
  analyse_fwd ham_swin_baseline swin
  note "Swin-T finished"
fi

if [ "$PART" = all ] || [ "$PART" = seeds ]; then
  echo "=== E3 seeds 43, 44 (ResNet-50) ==="
  for S in 43 44; do
    train_ham ham_baseline_s$S resnet50 "--seed $S"
    analyse_fwd ham_baseline_s$S s$S
  done
  note "seeds finished"
fi

echo "ALL REQUESTED PARTS DONE"
ls -1 label_shift_results_*.txt safety_results_*.txt adapt_results_*.txt 2>/dev/null
