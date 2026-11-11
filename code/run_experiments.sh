#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------- 0. paths
HAM_META=data/HAM10000/HAM10000_metadata.csv
HAM_IMG=data/HAM10000/images
I19_GT=data/ISIC2019/ISIC_2019_Training_GroundTruth.csv
I19_META=data/ISIC2019/ISIC_2019_Training_Metadata.csv
I19_IMG=data/ISIC2019/ISIC_2019_Training_Input
MASKS=data/ISIC2018_Task1_Training_GroundTruth   # lesion masks for HAM images (interpretability)
PH2_CSV=data/PH2/ph2.csv                          # columns: file,dx  (see README)
PH2_IMG=data/PH2/images
BACKBONES="resnet50 efficientnet_b0"
FOLDS="0 1 2 3 4"
EPOCHS=40

# ---------------------------------------------------------------- 1. prepare data (once)
python data.py ham10000 --meta $HAM_META --img_dir $HAM_IMG --out ham.csv
# ISIC 2019 contains the HAM10000 images -> remove them so HAM <-> ISIC2019 tests are truly external
python data.py isic2019 --gt $I19_GT --meta $I19_META --img_dir $I19_IMG --exclude_ids ham.csv --out isic19_noham.csv
python data.py generic --csv $PH2_CSV --path_col file --label_col dx --img_dir $PH2_IMG --no_folds --out ph2.csv

# ---------------------------------------------------------------- 2. main comparison + ablations
for DS in ham isic19_noham; do
  for BB in $BACKBONES; do
    for F in $FOLDS; do
      python train.py --data $DS.csv --fold $F --backbone $BB --epochs $EPOCHS --model baseline --out runs/${DS}_${BB}_baseline_f$F
      python train.py --data $DS.csv --fold $F --backbone $BB --epochs $EPOCHS                    --out runs/${DS}_${BB}_hgpb_f$F
    done
  done
  # ablations on the main backbone only
  for F in $FOLDS; do
    python train.py --data $DS.csv --fold $F --backbone resnet50 --epochs $EPOCHS --no_spatial      --out runs/${DS}_resnet50_noSpatial_f$F
    python train.py --data $DS.csv --fold $F --backbone resnet50 --epochs $EPOCHS --no_committee    --out runs/${DS}_resnet50_noCommittee_f$F
    python train.py --data $DS.csv --fold $F --backbone resnet50 --epochs $EPOCHS --no_sample_boost --out runs/${DS}_resnet50_noSampleBoost_f$F
    # hybrid baselines (the original pipeline, corrected)
    python hybrid_baseline.py --data $DS.csv --fold $F --out runs/${DS}_hybridRF_f$F
    python hybrid_baseline.py --data $DS.csv --fold $F --deep_run runs/${DS}_resnet50_hgpb_f$F --out runs/${DS}_hybridRFdeep_f$F
  done
done

# ---------------------------------------------------------------- 3. external validation (no fine-tuning)
for BB in $BACKBONES; do
  for M in baseline hgpb; do
    # HAM -> ISIC2019 (non-HAM part, all its folds) and HAM -> PH2
    python evaluate.py --run runs/ham_${BB}_${M}_f0 --data isic19_noham.csv --out runs/ext/ham2isic19_${BB}_${M}
    python evaluate.py --run runs/ham_${BB}_${M}_f0 --data ph2.csv \
        --label_map '{"common nevus":"nv","atypical nevus":"nv","melanoma":"mel"}' --out runs/ext/ham2ph2_${BB}_${M}
    # ISIC2019(non-HAM) -> HAM
    python evaluate.py --run runs/isic19_noham_${BB}_${M}_f0 --data ham.csv --out runs/ext/isic192ham_${BB}_${M}
  done
done

# ---------------------------------------------------------------- 4. interpretability (hardness maps vs masks)
python evaluate.py --run runs/ham_resnet50_hgpb_f0 --data ham.csv --split test --skip_metrics \
    --save_maps 60 --mask_dir $MASKS --mask_suffix _segmentation.png --out runs/maps_ham_f0

# ---------------------------------------------------------------- 5. tables
python summarize.py --runs runs --out results_by_fold.csv
python summarize.py --runs runs --compare ham_resnet50_hgpb ham_resnet50_baseline
python summarize.py --runs runs --compare isic19_noham_resnet50_hgpb isic19_noham_resnet50_baseline
