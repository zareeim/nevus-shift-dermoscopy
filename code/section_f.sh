#!/usr/bin/env bash
set -euo pipefail
python label_shift.py --runs runs_chromo_cv --exp ham_baseline --ood runs_ood_pf --out label_shift_results.txt
sed -n '/== F/,$p' label_shift_results.txt
