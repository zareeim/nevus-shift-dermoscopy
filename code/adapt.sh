#!/usr/bin/env bash
set -euo pipefail
python adapt.py run --data isic19_noham.csv
python adapt.py analyze
notify-send "Skin" "adaptation finished - see adapt_results.txt" || true
