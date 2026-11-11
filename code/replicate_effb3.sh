#!/usr/bin/env bash
set -euo pipefail
E=ham_effb3_baseline
python ood_perfold.py --data isic19_noham.csv --runs runs_chromo_cv --exp $E --out runs_ood_pf
python label_shift.py --runs runs_chromo_cv --exp $E --ood runs_ood_pf --out label_shift_results_effb3.txt
python adapt.py run --data isic19_noham.csv --exp $E --out runs_adapt_effb3
python adapt.py analyze --exp $E --out runs_adapt_effb3 --results adapt_results_effb3.txt
notify-send "Skin" "EfficientNet-B3 replication finished" || true
