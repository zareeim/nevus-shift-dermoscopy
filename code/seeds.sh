set -euo pipefail
for S in 43 44; do
  E=ham_baseline_s$S
  for F in 0 1 2 3 4; do
    OUT=runs_chromo_cv/${E}_f$F
    [ -f $OUT/test_metrics.json ] && { echo "skip $OUT"; continue; }
    python chromo.py train --data ham_masks.csv --fold $F --config baseline --seed $S --eval_test --out $OUT
  done
  python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp $E --out runs_ood_pf
  [ -f label_shift_results_s$S.txt ] || python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_pf --out label_shift_results_s$S.txt > /dev/null
  [ -f safety_results_$E.txt ] || python safety.py --exp $E > /dev/null
done
python extra_analysis.py > extra_log.txt 2>&1
cat label_shift_results_s43.txt safety_results_ham_baseline_s43.txt label_shift_results_s44.txt safety_results_ham_baseline_s44.txt extra_summary.txt > seeds_all.txt
notify-send "Skin" "seeds finished" 2>/dev/null || true
echo "SEEDS DONE -> seeds_all.txt"
