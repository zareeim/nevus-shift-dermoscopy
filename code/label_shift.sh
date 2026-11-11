#!/usr/bin/env bash
set -euo pipefail
python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp ham_baseline --out runs_ood_pf
python label_shift.py --runs runs_chromo_cv --exp ham_baseline --ood runs_ood_pf
notify-send "Skin" "label-shift study finished - see label_shift_results.txt" || true
