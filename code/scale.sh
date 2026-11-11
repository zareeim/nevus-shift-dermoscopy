#!/usr/bin/env bash
set -euo pipefail
python lesion_scale.py segment
[ -f scale_fstar.json ] || python lesion_scale.py stats
for E in ham_baseline ham_swin_baseline; do
  python lesion_scale.py predict --exp $E
  S=${E}_scale
  [ -f label_shift_results_$S.txt ] || python label_shift.py --runs runs_scale --exp $S --ood runs_scale_ood --out label_shift_results_$S.txt > /dev/null
  [ -f safety_results_$S.txt ] || python safety.py --runs runs_scale --ood runs_scale_ood --exp $S > /dev/null
  [ -f extra_results_$S.txt ] || python extra_analysis.py --runs runs_scale --ood runs_scale_ood --exps $S --summary extra_summary_$S.txt > /dev/null
done
cat scale_stats.txt label_shift_results_ham_baseline_scale.txt safety_results_ham_baseline_scale.txt extra_results_ham_baseline_scale.txt \
    label_shift_results_ham_swin_baseline_scale.txt safety_results_ham_swin_baseline_scale.txt extra_results_ham_swin_baseline_scale.txt > scale_all.txt
notify-send "Skin" "lesion-scale control finished" 2>/dev/null || true
echo "SCALE DONE -> scale_all.txt, scale_examples.jpg"
