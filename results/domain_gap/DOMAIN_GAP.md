# MNIST ↔ stylus domain gap

Generated from `results/domain_gap/runs.jsonl` and `reverse.jsonl` with `configs/domain_gap.json`. PyTorch CNN (team architecture), 5 epochs, accuracy in %, mean ± std over seeds 0–4. Δ is paired by seed against the baseline (pp). p: two-sided paired t-test; p (Holm): Holm-corrected over all exploratory recipe-vs-baseline tests in both directions. The headline recipe per direction was chosen on validation data before any test run and is the confirmatory test (no correction).

## 1. Why: measured differences (MNIST train 60,000 vs stylus 10,000)

| Group | Measure | Unit | MNIST mean ± std | Stylus mean ± std |
|---|---|---|---|---|
| Stroke width | Stroke width (ridge) | px | 2.563 ± 0.829 | 1.024 ± 0.101 |
| Intensity / anti-aliasing | Mean ink intensity | 0-1 | 0.689 ± 0.063 | 0.463 ± 0.053 |
| Intensity / anti-aliasing | Mid-gray share of ink pixels | share | 0.475 ± 0.091 | 0.832 ± 0.068 |
| Intensity / anti-aliasing | Edge softness | 0-1 | 0.298 ± 0.053 | 0.523 ± 0.068 |
| Size | Bounding-box height | px | 19.674 ± 1.117 | 19.778 ± 1.019 |
| Size | Bounding-box width | px | 15.199 ± 3.963 | 12.368 ± 3.901 |
| Size | Bounding-box longer side (MNIST convention: 20) | px | 19.997 ± 0.080 | 19.979 ± 0.274 |
| Size | Aspect ratio w / h | ratio | 0.781 ± 0.236 | 0.633 ± 0.240 |
| Position | Center-of-mass offset from center | px | 0.767 ± 0.287 | 1.483 ± 0.869 |
| Slant | Slant (second-moment angle) | deg | -5.902 ± 14.639 | -1.666 ± 8.903 |
| Coverage | Ink coverage (> 0.02) | share | 0.188 ± 0.052 | 0.101 ± 0.025 |
| Coverage | Ink coverage (> 0.25) | share | 0.156 ± 0.048 | 0.072 ± 0.019 |
| Device artifacts | Connected ink components | count | 1.030 ± 0.194 | 1.009 ± 0.092 |
| Device artifacts | Stray specks (components <= 3 px) | count | 0.017 ± 0.139 | 0.003 ± 0.058 |
| Shape | Closed loops (holes) | count | 0.586 ± 0.791 | 0.621 ± 0.857 |

Share of images with a closed loop, by class:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | 93% | 1% | 44% | 10% | 13% | 8% | 84% | 4% | 97% | 86% |
| Stylus | 93% | 3% | 28% | 4% | 49% | 1% | 91% | 5% | 97% | 82% |

## 2. Which factors cause the drop

### 2a. Stylus → MNIST: fix one mismatch at a time (stylus training images only)

| Factor | Recipe | MNIST test acc | Δ vs baseline (pp) | p | p (Holm) |
|---|---|---|---|---|---|
| none | baseline | 48.1 ± 7.6 | — | — | — |
| position (center of mass) | `f_center` | 64.6 ± 3.5 | +16.6 ± 5.8 | 0.003 | 0.058 |
| aspect (stretch ×1.3) | `f_aspect` | 50.9 ± 2.9 | +2.9 ± 5.5 | 0.31 | 1 |
| slant (random shear) | `f_slant` | 47.3 ± 3.5 | -0.8 ± 6.3 | 0.8 | 1 |
| stroke width (3×3 dilation) | `f_width` | 60.7 ± 2.6 | +12.7 ± 8.2 | 0.026 | 0.21 |
| intensity (peak → 1) | `f_intensity` | 47.8 ± 5.7 | -0.3 ± 2.1 | 0.79 | 1 |
| soft edges (blur σ 0.6) | `f_soft` | 53.5 ± 4.3 | +5.4 ± 3.7 | 0.031 | 0.21 |

### 2b. Cumulative attribution (fixed order: position → aspect → slant → width → intensity → soft edges)

| Step | Recipe | MNIST test acc | Step gain (pp) | Cumulative gain (pp) |
|---|---|---|---|---|
| + position | `f_center` | 64.6 ± 3.5 | +16.6 | +16.6 |
| + aspect | `c2_center_aspect` | 68.4 ± 2.6 | +3.7 | +20.3 |
| + slant | `c3_+slant` | 70.2 ± 2.4 | +1.9 | +22.2 |
| + stroke width | `c4_+width` | 72.5 ± 2.0 | +2.2 | +24.4 |
| + intensity | `c5_+intensity` | 71.8 ± 2.7 | -0.6 | +23.8 |
| + soft edges | `c6_+soft` | 70.2 ± 2.7 | -1.6 | +22.2 |

### 2c. Reverse: MNIST test images degraded towards stylus style, R1 (MNIST-trained) models

| Degradation | Accuracy | Δ vs clean (pp) | p |
|---|---|---|---|
| none | 98.8 ± 0.2 | — | — |
| thin | 95.1 ± 1.3 | -3.7 ± 1.3 | 0.003 |
| hard | 98.6 ± 0.2 | -0.2 ± 0.1 | 0.012 |
| dim | 98.8 ± 0.1 | -0.0 ± 0.1 | 0.43 |
| uncenter | 90.5 ± 1.5 | -8.3 ± 1.5 | 2.4e-04 |
| narrow | 96.0 ± 0.6 | -2.8 ± 0.6 | 5.8e-04 |
| upright | 98.8 ± 0.2 | +0.0 ± 0.1 | 0.61 |
| all | 78.3 ± 1.4 | -20.5 ± 1.3 | 4.2e-06 |

`thin`: Zhang–Suen skeleton scaled to the stylus median ink intensity. `hard`: binarized at 0.5. `dim`: intensity × (stylus / MNIST mean ink). `uncenter`: shifted so the center-of-mass offset follows the stylus distribution. `narrow`: squeezed to a stylus aspect ratio. `upright`: deslanted (stylus digits are nearly upright). `all`: upright + narrow + thin + uncenter.

## 3a. Fix: Stylus → MNIST (train 10,000 stylus, test MNIST test set)

Headline (pre-registered on validation data): **`normboth_dil_affine`**.

| Recipe | Selection val acc (3 seeds) | Test acc | Δ vs baseline (pp) | Seeds improved | p | p (Holm) |
|---|---|---|---|---|---|---|
| **`normboth_dil_affine`** | 80.9 | 82.8 ± 2.1 | +34.7 ± 6.1 | 5/5 | 2.1e-04 | confirmatory |
| `normboth_deslant_dil_affine` | 80.9 | 82.5 ± 2.6 | +34.5 ± 8.8 | 5/5 | 9.3e-04 | 0.025 |
| `normboth_rdil_affine` | 80.4 | 84.3 ± 1.9 | +36.3 ± 7.5 | 5/5 | 4.2e-04 | 0.014 |
| `normboth_dil_elastic` | 79.0 | 82.0 ± 1.2 | +33.9 ± 7.2 | 5/5 | 4.5e-04 | 0.015 |
| `normboth_dil_affine_noise_blur` | 78.2 | 81.0 ± 2.6 | +33.0 ± 6.2 | 5/5 | 2.8e-04 | 0.01 |
| `normboth_dil_affine_elastic` | 78.2 | 82.5 ± 0.9 | +34.4 ± 7.4 | 5/5 | 4.8e-04 | 0.015 |
| `normboth_deslant_dil` | 75.8 | 76.7 ± 2.3 | +28.6 ± 5.9 | 5/5 | 4.1e-04 | 0.014 |
| `normboth_rdil` | 73.8 | 73.5 ± 2.1 | +25.5 ± 8.7 | 5/5 | 0.0028 | 0.056 |
| `normboth_dil` | 70.7 | 71.9 ± 2.4 | +23.9 ± 5.8 | 5/5 | 7.7e-04 | 0.023 |
| `normboth_dil_blur` | 68.6 | 71.2 ± 3.6 | +23.2 ± 5.5 | 5/5 | 6.9e-04 | 0.021 |
| `normboth_dil_noise` | 68.4 | 71.5 ± 1.9 | +23.5 ± 6.5 | 5/5 | 0.0013 | 0.03 |
| `norm_dil_train` | 67.7 | 70.3 ± 2.4 | +22.3 ± 5.8 | 5/5 | 0.001 | 0.027 |
| `normboth` | 65.7 | 67.6 ± 4.0 | +19.6 ± 4.8 | 5/5 | 8.1e-04 | 0.023 |
| `norm_train` | 63.7 | 66.6 ± 3.9 | +18.5 ± 4.8 | 5/5 | 0.001 | 0.027 |
| `baseline` | 51.2 | 48.1 ± 7.6 | — | — | — | — |

Recipe definitions are in the config (`recipes`).

## 3b. Fix: MNIST → stylus (train 50,000 MNIST, test the writers' held-out splits, mean over 5 writers)

Headline (pre-registered on validation data): **`normboth_deslant`**.

| Recipe | Selection val acc (3 seeds) | Test acc | Δ vs baseline (pp) | Seeds improved | p | p (Holm) |
|---|---|---|---|---|---|---|
| **`normboth_deslant`** | 92.4 | 91.1 ± 2.7 | +10.3 ± 2.9 | 5/5 | 0.0014 | confirmatory |
| `normboth_noise` | 92.2 | 91.0 ± 1.5 | +10.1 ± 4.2 | 5/5 | 0.0056 | 0.085 |
| `normboth_affine_erode_contrast` | 91.4 | 89.4 ± 2.7 | +8.5 ± 4.1 | 5/5 | 0.0097 | 0.13 |
| `normboth` | 91.3 | 90.5 ± 1.6 | +9.6 ± 4.2 | 5/5 | 0.0067 | 0.094 |
| `normboth_affine` | 91.3 | 89.9 ± 3.0 | +9.0 ± 3.7 | 5/5 | 0.0053 | 0.085 |
| `normboth_contrast_hard` | 90.8 | 92.3 ± 1.0 | +11.4 ± 4.5 | 5/5 | 0.0046 | 0.079 |
| `normboth_testdil_affine` | 90.5 | 87.1 ± 3.9 | +6.3 ± 3.2 | 5/5 | 0.012 | 0.14 |
| `normboth_rerode` | 90.3 | 90.3 ± 1.6 | +9.5 ± 4.6 | 5/5 | 0.0099 | 0.13 |
| `normboth_testdil2` | 90.2 | 89.5 ± 2.6 | +8.6 ± 4.8 | 5/5 | 0.016 | 0.16 |
| `normboth_deslant_testdil_affine` | 89.0 | 84.2 ± 5.6 | +3.3 ± 2.8 | 5/5 | 0.054 | 0.33 |
| `normboth_testdil` | 87.7 | 87.0 ± 1.3 | +6.2 ± 3.7 | 5/5 | 0.02 | 0.18 |
| `normboth_testdil_affine_elastic` | 87.1 | 82.9 ± 3.3 | +2.0 ± 5.3 | 4/5 | 0.44 | 1 |
| `normboth_elastic` | 84.7 | 86.4 ± 2.6 | +5.5 ± 4.6 | 5/5 | 0.055 | 0.33 |
| `baseline` | 82.3 | 80.9 ± 3.7 | — | — | — | — |

Recipe definitions are in the config (`recipes`).

## 4. Per class, stylus → MNIST (confusion matrices, mean ± std over seeds)

| Recipe | recall 7 | precision 8 | predicted-8 share | 9 → 4 rate | recall 9 | precision 4 |
|---|---|---|---|---|---|---|
| `baseline` | 1.1 ± 1.1 | 31.9 ± 13.7 | 33.3 ± 13.1 | 49.8 ± 10.2 | 13.9 ± 9.3 | 45.3 ± 5.9 |
| `f_width` | 22.2 ± 9.2 | 88.5 ± 4.7 | 5.6 ± 1.6 | 74.8 ± 7.9 | 1.6 ± 1.6 | 40.6 ± 1.6 |
| `norm_dil_train` | 72.5 ± 7.3 | 91.8 ± 2.0 | 6.5 ± 1.4 | 31.6 ± 10.9 | 22.3 ± 9.5 | 62.2 ± 3.0 |
| `normboth_dil_affine` | 91.8 ± 2.4 | 89.2 ± 4.5 | 7.9 ± 1.5 | 10.9 ± 7.2 | 44.8 ± 13.0 | 82.4 ± 7.9 |
