# Dance Stress Physiological Analysis

This repository contains the physiological signal-processing and statistical
analysis code used for the dance momentary-stress study. It covers signal
harmonization, exercise-level feature extraction, momentary-stress prediction,
trait-covariate analyses, mixed-effects models, and the exercise-identity
classification control.

This README provides a short summary of the analysis and findings.

## Analysis overview

The physiological inputs were accelerometry (`ACC`), blood-volume pulse
(`BVP`), electrodermal activity (`EDA`), and temperature (`TEMP`) recorded
during three exercises. Two wearable devices were used, so signals were
resampled to common frequencies before feature extraction:

| Signal | Harmonized frequency |
| --- | ---: |
| ACC | 32 Hz |
| BVP | 64 Hz |
| EDA | 4 Hz |
| TEMP | 1 Hz |

The final analysis contained 88 participants and 262 valid exercise-level
observations. All predictive evaluations used subject-wise five-fold
cross-validation. Nested analyses performed model selection inside each
outer training fold.

The regression benchmark compared Ridge, Random Forest, and Histogram
Gradient Boosting models with an exercise-mean baseline. Exercise-identity
classification compared the corresponding classifier families with a global
majority-class baseline.

## Repository structure

```text
.
├── analysis/
│   ├── preprocessing/
│   │   ├── signal_preprocessing.py
│   │   ├── extract_features.py
│   │   └── merge_stress_labels.py
│   ├── shared/
│   │   └── harmonized_eval.py
│   ├── stress_models/
│   │   └── run_stress_regression.py
│   ├── exercise_control/
│   │   ├── run_exercise_classification.py
│   │   └── run_order_drift_control.py
│   └── trait_models/
│       ├── build_subject_labels.py
│       ├── run_trait_covariate_models.py
│       └── run_mixed_effects.py
├── .gitignore
└── requirements.txt
```

## Code guide

### Harmonized physiology pipeline

- `analysis/preprocessing/signal_preprocessing.py`: loads the four signal types, estimates sampling rates,
  performs signal-specific cleaning and filtering, and resamples the two
  devices to common frequencies.
- `analysis/preprocessing/extract_features.py`: extracts exercise-level ACC,
  BVP, EDA, and TEMP features and quality-control variables.
- `analysis/preprocessing/merge_stress_labels.py`: merges the extracted features with the
  exercise-level momentary-stress ratings and applies the validity rules.
- `analysis/shared/harmonized_eval.py`: contains shared feature selection,
  preprocessing, model definitions, baselines, and subject-wise evaluation
  utilities.
- `analysis/stress_models/run_stress_regression.py`: runs fixed-parameter and nested
  stress-regression benchmarks, including modality, exercise, and device
  ablations.
- `analysis/exercise_control/run_exercise_classification.py`: evaluates classification of
  exercise identity for single- and multi-modality feature sets.
- `analysis/exercise_control/run_order_drift_control.py`: repeats nested autonomic exercise
  classification after removing the within-participant linear trend across
  exercise order.

### Trait and mixed-effects analyses

- `analysis/trait_models/build_subject_labels.py`: extracts the subject-level STAI-Y,
  Cohen perceived-stress, and TAS-20 scores.
- `analysis/trait_models/run_trait_covariate_models.py`: compares STAI-Y, Cohen perceived
  stress, TAS-20, and selected combinations as stress-model covariates.
- `analysis/trait_models/run_mixed_effects.py`: fits participant-random-intercept models for
  exercise, trait, and selected physiological effects.

## Environment

Python 3.10 is recommended. The core analysis was verified with the package
versions pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Analysis summary

The values below correspond to the physiological analyses reported in the
manuscript dated October 4, 2026.

### Momentary-stress prediction

The exercise-mean baseline achieved MAE 1.4250 and RMSE 1.7300. Physiological
models did not consistently improve on that baseline.

| Feature setting | Best fixed MAE / RMSE | Best nested MAE / RMSE |
| --- | ---: | ---: |
| All physiological features | 1.4984 / 1.8100 | 1.4917 / 1.7840 |
| All physiology + exercise | 1.4872 / 1.7973 | 1.4806 / 1.7739 |
| ACC + EDA + exercise | 1.4807 / 1.7969 | 1.4749 / 1.7707 |

The modality, exercise, and device ablations followed the same fixed and
nested subject-wise evaluation protocol.

### Exercise-identity control

Exercise identity was strongly recoverable from accelerometry, as expected
from differences in movement tempo and amplitude. Autonomic modalities showed
more modest signal.

| Feature set | Nested accuracy | Nested macro F1 |
| --- | ---: | ---: |
| Majority-class baseline | 33.2% | 0.17 |
| ACC only | 82.8% | 0.82 |
| TEMP only | 59.7% | 0.58 |
| EDA only | 57.2% | 0.57 |
| BVP only | 52.6% | 0.51 |
| ACC + EDA | 83.5% | 0.83 |
| All physiological features | 80.5% | 0.80 |

After participant-wise linear-order residualization, the autonomic-only macro
F1 scores remained above baseline but were modest: BVP 0.54, TEMP 0.54, and
EDA 0.53. This bounds a monotonic session-drift account but does not eliminate
the fixed exercise-order confound.

### Trait-aware stress models

Exercise + STAI-Y + TAS-20 was the only configuration that consistently
improved on the exercise-mean baseline in both evaluation schemes.

| Variant | Fixed MAE / RMSE | Nested MAE / RMSE |
| --- | ---: | ---: |
| Exercise-mean baseline | 1.4250 / 1.7300 | 1.4250 / 1.7300 |
| Exercise + STAI-Y | 1.4216 / 1.7173 | 1.4257 / 1.7297 |
| Exercise + TAS-20 | 1.4356 / 1.7342 | 1.4358 / 1.7344 |
| Exercise + STAI-Y + TAS-20 | 1.3772 / 1.6850 | 1.3741 / 1.6815 |
| ACC + EDA + exercise | 1.5090 / 1.8658 | 1.4682 / 1.7623 |
| ACC + EDA + exercise + STAI-Y + TAS-20 | 1.4700 / 1.8359 | 1.4757 / 1.7731 |

The trait-aware run was separate from the main physiological benchmark, with
a different model-selection rule. Its ACC + EDA + exercise estimate is
therefore not interchangeable with the corresponding value above.

### Mixed-effects sensitivity analysis

The best fit on the 262 physiological observations was obtained for exercise
+ STAI-Y + TAS-20. Adding the representative ACC and EDA terms worsened fit.

| Model | AIC |
| --- | ---: |
| Exercise only | 959.25 |
| Exercise + STAI-Y | 957.72 |
| Exercise + TAS-20 | 960.39 |
| Exercise + STAI-Y + TAS-20 | 953.38 |
| Exercise + all traits | 955.37 |
| Exercise + ACC + EDA | 963.42 |
| Exercise + ACC + EDA + STAI-Y | 961.63 |

## Reproducibility notes

- Random seeds are fixed at zero.
- Predictive evaluations keep all observations from one participant in the
  same fold.
- Quality-control variables and bookkeeping fields are excluded from model
  features.
