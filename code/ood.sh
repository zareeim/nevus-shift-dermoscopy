#!/usr/bin/env bash
set -euo pipefail
GT=${1:?GroundTruth csv}; META=${2:?Metadata csv}; IMG=${3:?image folder}
[ -f isic19_noham.csv ] || python data.py isic2019 --gt "$GT" --meta "$META" --img_dir "$IMG" \
    --exclude_ids ham.csv --no_folds --out isic19_noham.csv
python ood_eval.py --data isic19_noham.csv --runs runs_aug \
    --exps ham_none ham_galdran ham_chromo ham_chromo_global --out runs_ood
notify-send "Skin" "OOD test finished - see ood_results.txt" || true
