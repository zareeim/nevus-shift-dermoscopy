# Code for: A nevus-ward shift in cross-dataset dermoscopic classification: consequences for label-shift estimation and melanoma detection

Author: Mina Zareei (Independent Researcher, Bielefeld, Germany)

This archive contains the code, the analysis-plan file and the result summaries behind the paper.
It contains **no images and no dataset files**. The datasets must be downloaded from their original sources:
HAM10000 (Tschandl et al., 2018), ISIC 2019 and ISIC 2020 (ISIC Archive), Derm7pt (Kawahara et al., 2019).
Derm7pt is distributed under a CC BY-NC-ND licence and is therefore not redistributed here;
`code/prep_derm7pt.py` prepares it from the original download.

## Folders
- `code/` – all scripts (Python 3, PyTorch, scikit-learn, pandas, matplotlib)
- `results/` – text summaries and small tables produced by the scripts
- `PREREGISTRATION_extension.md` – analysis plan (pre-specified, not pre-registered; see Section 3.7 of the paper)

## Where each result comes from
| Paper | Script(s) |
| --- | --- |
| Model training (4 architectures x 3 seeds x 5 folds) | `chromo.py`, `model.py`, `data.py`, `losses.py`, `seeds.sh` |
| Predictions on the targets | `ood_perfold.py` |
| Data preparation (Derm7pt, reverse direction, ISIC metadata) | `prep_derm7pt.py`, `derm7pt.sh`, `prep_reverse.py`, `isic_api.py`, `add_masks.py` |
| Matched protocol, Tables 4 and 8 | `matched.py` |
| Primary tests and Holm correction (Section 4.8) | `pooled_tests.py`, `matched.py --primary` |
| Label-shift methods and controlled conditions (Table 5) | `label_shift.py`, `label_shift.sh` |
| Melanoma detection, calibration, ISIC 2020 by contributor, akiec taxonomy (Table 6, Sections 4.3-4.6) | `review5.py` |
| Confidence intervals for ROC-AUC and operating points; contributor CIs | `review6.py` |
| ROC tail share (Section 4.6) | `review7.py` |
| Histopathology-only case mix (Table 7) | `robustness.py`, `audit_meta.py` |
| Follow-up-nevus control | `nofollow.sh` |
| Lesion-scale control | `seg.py`, `lesion_scale.py`, `scale.sh` |
| Near-duplicate check | `dups.py`, `rm_dups_i20.sh` |
| Choice of denominator for the simple rule | `prior_check.py` |
| Test-time adaptation (AdaBN, TENT) | `adapt.py`, `adapt.sh` |
| Derm7pt subtypes and other descriptive analyses | `extra_analysis.py`, `safety.py` |
| Replication criteria (Supplementary Table S1) | `s1_table.py` |
| Figures 1-3 | `figures_final.py` (uses `review6.py`, `review7.py`, `results/review5_deltaC.csv`) |

Scripts are run from the folder that contains the prediction folders (`runs_chromo_cv/`, `runs_ood_*`).
Earlier ensemble-based scripts (`ensemble.py`, `evaluate.py`) are kept for the analysis history (Supplementary Appendix A).

## Licence
MIT (code). The datasets keep their own licences.

## Additional analyses (review8.py)

| Result in the paper | Script | Output |
| --- | --- | --- |
| Cluster counts (Table 1) | code/review8.py, section 1 | results/review8_results.txt |
| Decomposition check, predicted vs observed condition B (Section 4.2) | code/review8.py, section 2 | results/review8_results.txt |
| Oracle threshold at 80% specificity (Section 4.3) | code/review8.py, section 3 | results/review8_results.txt |
| Dark-vignette stratification (Section 4.4) | code/review8.py, section 5 | results/review8_results.txt |
