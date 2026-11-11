#!/usr/bin/env bash
set -uo pipefail
PART=${1:?give a step: matched | probe | meta | i20 | seeds}
note() { notify-send "Skin" "$1" 2>/dev/null || true; echo ">>> $1"; }
HM=$(find ~/workspace -name "HAM10000_metadata*.csv" 2>/dev/null | head -1)
D7=../derm7pt/release_v0
BASE="ham_baseline ham_baseline_s43 ham_baseline_s44 ham_effb3_baseline ham_convnext_baseline ham_swin_baseline bcn_baseline"
NEW="ham_effb3_baseline_s43 ham_effb3_baseline_s44 ham_convnext_baseline_s43 ham_convnext_baseline_s44 ham_swin_baseline_s43 ham_swin_baseline_s44"

other_targets() {  # $1 = experiment: Derm7pt and (if prepared) ISIC 2020
  local E=$1
  python ood_perfold.py --data derm7pt.csv --runs runs_chromo_cv --exp $E --out runs_ood_d7
  [ -f d7_label_shift_$E.txt ] || python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_d7 --out d7_label_shift_$E.txt > /dev/null
  [ -f d7_safety_$E.txt ] || python safety.py --exp $E --ood runs_ood_d7 --sources none --out d7_safety_$E.txt > /dev/null
  if [ -f isic2020_target.csv ]; then
    python ood_perfold.py --data isic2020_target.csv --runs runs_chromo_cv --exp $E --out runs_ood_i20
    [ -f i20_label_shift_$E.txt ] || python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_i20 --out i20_label_shift_$E.txt > /dev/null
    [ -f i20_safety_$E.txt ] || python safety.py --exp $E --ood runs_ood_i20 --sources none --out i20_safety_$E.txt > /dev/null
  fi
}

case $PART in
matched)
  python matched.py --exps $BASE --targets ISIC2019=runs_ood_pf Derm7pt=runs_ood_d7 --out matched_results.txt > /dev/null
  note "matched protocol finished -> matched_results.txt" ;;
probe)
  python isic_api.py meta --csv isic19_noham.csv --limit 50 --out isic_metadata_probe.csv ;;
meta)
  python isic_api.py meta --csv isic19_noham.csv --out isic_metadata.csv | tee isic_meta_log.txt
  python robustness.py --ham_meta "$HM" --d7_root $D7 --isic_meta isic_metadata.csv --skip_dups > /dev/null
  cat isic_meta_log.txt robustness_results.txt > meta_all.txt
  note "ISIC metadata audit finished -> meta_all.txt" ;;
i20)
  [ -f isic2020_target.csv ] || python isic_api.py isic2020 --out isic2020_target.csv | tee isic2020_log.txt
  for E in $BASE; do other_targets $E; done
  python extra_analysis.py --ood runs_ood_i20 --exps $BASE --prefix i20_ --summary i20_extra_summary.txt > /dev/null
  python matched.py --exps $BASE --targets ISIC2020=runs_ood_i20 --out matched_results_i20.txt > /dev/null
  python dups.py > dups_log2.txt 2>&1
  cat isic2020_log.txt i20_extra_summary.txt matched_results_i20.txt i20_safety_ham_baseline.txt dups_log2.txt > i20_all.txt
  note "ISIC 2020 finished -> i20_all.txt" ;;
seeds)
  for SPEC in "effb3 efficientnet_b3 --bs 16" "convnext convnext_tiny --bs 16" "swin swin_t --bs 16 --lr 1e-4"; do
    set -- $SPEC; TAG=$1; BB=$2; shift 2; EXTRA="$*"
    for S in 43 44; do
      E=ham_${TAG}_baseline_s$S
      for F in 0 1 2 3 4; do
        OUT=runs_chromo_cv/${E}_f$F
        [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
        python chromo.py train --data ham_masks.csv --fold $F --config baseline --backbone $BB $EXTRA --seed $S --eval_test --out $OUT || exit 1
      done
      python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp $E --out runs_ood_pf
      [ -f label_shift_results_${TAG}_s$S.txt ] || python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_pf --out label_shift_results_${TAG}_s$S.txt > /dev/null
      [ -f safety_results_$E.txt ] || python safety.py --exp $E > /dev/null
      other_targets $E
      note "$E finished"
    done
  done
  python extra_analysis.py --exps $BASE $NEW --summary extra_summary_all.txt > /dev/null
  python extra_analysis.py --ood runs_ood_d7 --exps $BASE $NEW --prefix d7_ --summary d7_extra_summary_all.txt > /dev/null
  [ -f isic2020_target.csv ] && python extra_analysis.py --ood runs_ood_i20 --exps $BASE $NEW --prefix i20_ --summary i20_extra_summary_all.txt > /dev/null
  python matched.py --exps $BASE $NEW --out matched_results_all.txt > /dev/null
  cat label_shift_results_effb3_s4*.txt label_shift_results_convnext_s4*.txt label_shift_results_swin_s4*.txt \
      extra_summary_all.txt d7_extra_summary_all.txt i20_extra_summary_all.txt matched_results_all.txt > seeds_arch_all.txt 2>/dev/null
  note "extra seeds finished -> seeds_arch_all.txt" ;;
*) echo "unknown step $PART"; exit 1 ;;
esac
echo "STEP DONE: $PART"
