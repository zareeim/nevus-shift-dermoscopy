# Pre-registration: extension experiments (written 2026-11-03, before any of these results exist)

All models: fixed 30 epochs, last EMA checkpoint, no model selection on any target data.
Analyses use the unchanged scripts `label_shift.py` and `safety.py` (and `adapt.py` where the
network contains BatchNorm). Target labels are used for evaluation only.

## E1. Third and fourth architecture (HAM10000 -> ISIC 2019 non-HAM)
- ConvNeXt-Tiny baseline (already trained in stage 1; modern CNN, LayerNorm, no BatchNorm)
- Swin-Tiny baseline (vision transformer, LayerNorm, no BatchNorm; lr 1e-4, bs 16)

Replication criteria (each judged separately, per architecture):
- R1 OOD balanced accuracy drop > 20 points
- R2 controlled pure label shift on HAM test folds: |BA(raw) change| < 2 points
- R3 P(yhat=nv | mel) on OOD > on HAM
- R4 BBSE overestimates the nevus prior, and melanoma is the largest positive contributor
- R5 simple minus raw BA on OOD: 95% lesion-cluster CI above 0
- R6 simple BA >= best of {bcts_em, bbse, bcts_bbse}
AdaBN/TENT are not applicable (no BatchNorm) and are not run.

## E2. Reverse direction (BCN-like ISIC 2019 -> HAM10000), ResNet-50
Training data: BCN-like proxy (1024x1024) images of ISIC 2019 non-HAM, 5 lesion-grouped folds.
Target: all 10,015 HAM10000 images.
Questions (no direction is assumed in advance):
- Q1 Is there a BA drop, and does the controlled pure label shift reproduce it (R2)?
- Q2 Is the class-conditional shift one-sided, and towards which class (largest column change
  of C_T - C_S; sign and size of the BBSE prior error per class)?
- Q3 Do prior-estimation methods (EM, BCTS+EM, BBSE) fail relative to the simple rule (R6)?
- Q4 Does AdaBN (+simple) beat none+simple (pre-registered primary, as in the forward study)?
Interpretation fixed in advance: if the bias in the reverse direction is NOT towards nevus,
the paper will say the direction of the bias depends on the source/target pair, and will not
describe nevus bias as a general property.

## E3. Seed variation, ResNet-50 forward direction
Two extra training seeds (43, 44) for the 5 HAM folds (same test folds; the inner validation
split changes with the seed). Report mean and range over the three seeds of: OOD BA raw/simple,
simple minus raw, BBSE nevus-prior error, P(yhat=nv | mel). Criterion: simple minus raw > 0
for every seed.

## E4. Lesion-scale control (written 2026-11-06, before any result of it exists)
Concern: HAM10000 (600x450, centre-cropped) shows lesions larger in the model input than the
1024x1024 BCN-like images, and the source proxy is itself defined by image size. Same fold models,
no training; every image (source validation and test, target) is cropped to a lesion-centred square
so that the predicted lesion covers f* = the HAM10000 median fraction. Models: ResNet-50, Swin-T.
Outcomes, scale-normalised vs standard, forward direction:
- S1 raw target BA and the drop (source - target)
- S2 raw mel->nv and other->nv rate increase (target - source)
- S3 BBSE nevus-prior error, condition B (pure class-conditional) and C (real target)
Interpretation fixed in advance:
- if the increase in mel->nv AND the condition-B nevus-prior error both shrink by >= 50%,
  lesion scale is reported as a major driver of the nevus-ward shift;
- if both shrink by < 25%, scale is reported as not explaining the shift;
- otherwise, as a partial contributor.
Caveat recorded in advance: target masks are predicted (no reference masks for ISIC 2019), and the
crop also changes the framing of source images, so source BA under scale normalisation is reported.

## E5. Third independent target: Derm7pt (written 2026-11-07, before any result of it exists)
Same fold models (all forward models, the two extra seeds, and the reverse BCN-like model), no
training or selection on Derm7pt. Dermoscopic image only; diagnoses mapped to HAM10000 classes
(nevus subtypes incl. Reed/Spitz -> nv; melanoma subtypes -> mel; seborrheic keratosis and lentigo
-> bkl; BCC, DF, vascular as is; miscellaneous, melanosis and metastases excluded; no akiec).
Replication criteria for the forward models:
- D1 P(yhat=nv | mel) higher on Derm7pt than on the source test folds
- D2 P(yhat=nv | y) increases for the majority of the non-nevus classes present
- D3 BBSE overestimates the nevus prior on Derm7pt
- D4 simple minus raw BA: 95% lesion-cluster CI above 0
If D1-D3 fail, the paper will state that the nevus-ward shift was observed between HAM10000 and
ISIC 2019 only and is not a general property of cross-institutional transfer.
Caveat recorded in advance: Derm7pt is small (~1,000 images; DF and vascular lesions are rare), so
per-class estimates for rare classes are imprecise.

## E6-E9. Second review (written 2026-11-08, before any of these results exist; refined the same day
## after a methodological review of the protocol, still before any result was computed)

### E6 Matched protocol (forward models, ISIC 2019 and Derm7pt; reverse model on HAM10000)
Unit of comparison: fold model k on its own source test fold vs the same model on the whole target.
Per-fold quantities are computed first and then averaged over the 5 folds; target predictions of the
five models are never pooled as independent samples. CIs: lesion-cluster bootstrap (1,000 replicates)
in which the target lesions are resampled ONCE per replicate and shared by all five fold models
(paired), and the lesions of each source test fold are resampled within that fold. The five folds are
overlapping training splits and are not treated as independent samples (min/max are descriptive).
Outcomes compared with the ensemble-based values reported so far:
  O1 balanced-accuracy drop (source - target), O2 mel->nv increase, O3 other-non-nevus->nv increase,
  O4 simple minus raw balanced accuracy on the target.
Rule: if any of O1-O4 differs from the ensemble-based value by more than 2 points for any model, the
matched values become the main values for ALL models and all four outcomes; otherwise the ensemble
values stay in the main text. In both cases both sets are reported (the other in the supplement),
with the reason for the protocol change.
BBSE: re-derived per fold with the estimator's own validation confusion matrix, so the reported
decomposition is exactly the estimator's (unclipped) error; the clipped estimate is reported beside it;
the source test fold as C_S is reported as a sensitivity analysis.
Prior sensitivity of the simple rule: (a) effective prior, (b) raw label frequencies, (c) BCTS mean.

### E7 Source proxy audit (ISIC Archive API)
Feasibility first: 50 images; report the returned fields. Then all images. Reported: share with an API
record, share with non-empty attribution, the attributions. Attribution names the CONTRIBUTING
institution, which is not necessarily where the image was taken; the paper will say so.
Rule: the proxy is called validated if >= 90% of the 1024x1024 images are attributed to Hospital
Clinic de Barcelona; otherwise all per-source analyses are repeated by attribution and the size proxy
is dropped. If attribution coverage is < 80%, the audit is reported as inconclusive.

### E8 ISIC 2020 as a separate external test
Images with a labelled diagnosis only (nevus, melanoma, seborrheic keratosis, lentigo, lichenoid
keratosis); 'unknown', cafe-au-lait macule and atypical melanocytic proliferation excluded; images on
the official duplicate list removed. Unit of resampling: PATIENT (all images of a sampled patient enter
the replicate together). ISIC 2020 is a melanoma-vs-benign task, so it is reported SEPARATELY, as a
binary external test: primary outcomes melanoma ROC-AUC, melanoma sensitivity and specificity at the
thresholds fixed on source validation (80% / 90% specificity), and the melanoma->nevus rate.
Secondary (descriptive): 3-class directional errors (nv/mel/bkl). It is not pooled with ISIC 2019 or
Derm7pt and is not used as evidence for the 7-class one-sided shift.
Criterion: melanoma->nevus rate higher than on the source test folds (direction), as D1.

### E9 Extra seeds (43, 44) for EfficientNet-B3, ConvNeXt-T, Swin-T
Run only after E6 is complete. Criteria R1-R6 per seed, as in E1.
