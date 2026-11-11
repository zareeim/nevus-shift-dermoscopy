set -euo pipefail
ROOT=$1
[ -f derm7pt.csv ] || python prep_derm7pt.py --root "$ROOT" > derm7pt_data.txt
EXPS="ham_baseline ham_baseline_s43 ham_baseline_s44 ham_effb3_baseline ham_convnext_baseline ham_swin_baseline bcn_baseline"
for E in $EXPS; do
  python ood_perfold.py --data derm7pt.csv --runs runs_chromo_cv --exp $E --out runs_ood_d7
  [ -f d7_label_shift_$E.txt ] || python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_d7 --out d7_label_shift_$E.txt > /dev/null
  [ -f d7_safety_$E.txt ] || python safety.py --exp $E --ood runs_ood_d7 --sources none --out d7_safety_$E.txt > /dev/null
done
python extra_analysis.py --ood runs_ood_d7 --exps $EXPS --prefix d7_ --summary d7_extra_summary.txt > /dev/null
cat derm7pt_data.txt d7_label_shift_*.txt d7_safety_*.txt d7_extra_summary.txt > derm7pt_all.txt
notify-send "Skin" "Derm7pt finished" 2>/dev/null || true
echo "DERM7PT DONE -> derm7pt_all.txt"
