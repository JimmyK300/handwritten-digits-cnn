# Holm correction across all exploratory tests

Built by `python -m digitstudy.holm_report` from `results/summary.json`, `results/regimes/runs.jsonl` and `results/domain_gap/summary.json`. All tests are two-sided paired t-tests over 5 seeds.

- Exploratory tests: **185**, one Holm family. Significant at p < 0.05 before correction: 137; after Holm: **50**.
- Confirmatory tests (pre-registered, not corrected): 6.

## Confirmatory

| Family | Test | Effect | p |
|---|---|---|---|
| Dilation study | torch, custom_to_mnist: dilate_k3_i1 vs baseline | +12.59 pp | 0.024 |
| Dilation study | torch, lowo: dilate_k3_i1 vs baseline | +0.81 pp | 0.074 |
| Regimes | diversity (size-matched) k=4 vs k=1: unseen_acc | +12.3 pp | 0.00017 |
| Regimes | diversity (size-matched) k=4 vs k=1: writer_bias | -18.3 pp | 4.9e-05 |
| Domain gap | s2m: normboth_dil_affine vs baseline | +34.73 pp | 0.00021 |
| Domain gap | m2s: normboth_deslant vs baseline | +10.27 pp | 0.0014 |

## Exploratory (Holm-adjusted)

| Family | Test | Effect | p | p (Holm) | Survives |
|---|---|---|---|---|---|
| Dilation study | numpy, custom_to_mnist: dilate_k3_i1 vs baseline | +6.52 pp | 0.04 | 1 | no |
| Dilation study | numpy, custom_to_mnist: norm_mnist vs baseline | +12.20 pp | 0.0025 | 0.24 | no |
| Dilation study | numpy, custom_to_mnist: norm_mnist_dilate_k3_i1 vs baseline | +12.61 pp | 0.0012 | 0.13 | no |
| Dilation study | torch, custom_to_mnist: aug_stroke_width vs baseline | +7.28 pp | 0.14 | 1 | no |
| Dilation study | torch, custom_to_mnist: dilate_k2_i1 vs baseline | +7.77 pp | 0.056 | 1 | no |
| Dilation study | torch, custom_to_mnist: dilate_k2_i1_gain_1.5 vs baseline | +9.58 pp | 0.04 | 1 | no |
| Dilation study | torch, custom_to_mnist: dilate_k2_i2 vs baseline | +2.29 pp | 0.51 | 1 | no |
| Dilation study | torch, custom_to_mnist: dilate_k3_i2 vs baseline | -5.86 pp | 0.14 | 1 | no |
| Dilation study | torch, custom_to_mnist: dilate_k5_i1 vs baseline | -5.86 pp | 0.14 | 1 | no |
| Dilation study | torch, custom_to_mnist: gain_1.5 vs baseline | +4.93 pp | 0.044 | 1 | no |
| Dilation study | torch, custom_to_mnist: gain_2.0 vs baseline | +7.24 pp | 0.049 | 1 | no |
| Dilation study | torch, custom_to_mnist: norm_center vs baseline | +17.42 pp | 0.0028 | 0.26 | no |
| Dilation study | torch, custom_to_mnist: norm_mnist vs baseline | +19.87 pp | 0.0014 | 0.15 | no |
| Dilation study | torch, custom_to_mnist: norm_mnist_dilate_k2_i1 vs baseline | +18.37 pp | 0.0065 | 0.54 | no |
| Dilation study | torch, custom_to_mnist: norm_mnist_dilate_k3_i1 vs baseline | +22.11 pp | 0.00071 | 0.086 | no |
| Dilation study | numpy, lowo: dilate_k2_i1 vs baseline | +1.10 pp | 0.089 | 1 | no |
| Dilation study | numpy, lowo: dilate_k3_i1 vs baseline | +1.31 pp | 0.062 | 1 | no |
| Dilation study | torch, lowo: aug_stroke_width vs baseline | -3.07 pp | 0.011 | 0.81 | no |
| Dilation study | torch, lowo: dilate_k2_i1 vs baseline | +0.04 pp | 0.94 | 1 | no |
| Dilation study | torch, lowo: dilate_k2_i1_gain_1.5 vs baseline | +0.92 pp | 0.031 | 1 | no |
| Dilation study | torch, lowo: dilate_k2_i2 vs baseline | +0.59 pp | 0.18 | 1 | no |
| Dilation study | torch, lowo: dilate_k3_i2 vs baseline | -2.11 pp | 0.018 | 1 | no |
| Dilation study | torch, lowo: dilate_k5_i1 vs baseline | -2.11 pp | 0.018 | 1 | no |
| Dilation study | torch, lowo: erode_k2_i1 vs baseline | -14.25 pp | 0.0002 | 0.029 | yes |
| Dilation study | torch, lowo: gain_0.7 vs baseline | -0.82 pp | 0.1 | 1 | no |
| Dilation study | torch, lowo: gain_1.5 vs baseline | +0.32 pp | 0.39 | 1 | no |
| Dilation study | torch, lowo: gain_2.0 vs baseline | +0.43 pp | 0.39 | 1 | no |
| Dilation study | numpy, mnist_to_stylus: dilate_k2_i1 vs baseline | -0.74 pp | 0.63 | 1 | no |
| Dilation study | numpy, mnist_to_stylus: dilate_k3_i1 vs baseline | +3.43 pp | 0.085 | 1 | no |
| Dilation study | torch, mnist_to_stylus: aug_stroke_width vs baseline | +1.38 pp | 0.49 | 1 | no |
| Dilation study | torch, mnist_to_stylus: dilate_k2_i1 vs baseline | -5.01 pp | 0.00082 | 0.094 | no |
| Dilation study | torch, mnist_to_stylus: dilate_k2_i1_gain_1.5 vs baseline | -4.94 pp | 0.00045 | 0.059 | no |
| Dilation study | torch, mnist_to_stylus: dilate_k2_i2 vs baseline | -15.29 pp | 2.3e-05 | 0.0037 | yes |
| Dilation study | torch, mnist_to_stylus: dilate_k3_i1 vs baseline | -2.36 pp | 0.0022 | 0.22 | no |
| Dilation study | torch, mnist_to_stylus: dilate_k3_i2 vs baseline | -17.68 pp | 6.1e-05 | 0.0094 | yes |
| Dilation study | torch, mnist_to_stylus: dilate_k5_i1 vs baseline | -17.68 pp | 6.1e-05 | 0.0094 | yes |
| Dilation study | torch, mnist_to_stylus: erode_k2_i1 vs baseline | -59.84 pp | 1.3e-06 | 0.00023 | yes |
| Dilation study | torch, mnist_to_stylus: gain_0.7 vs baseline | -0.64 pp | 0.23 | 1 | no |
| Dilation study | torch, mnist_to_stylus: gain_1.5 vs baseline | +0.54 pp | 0.13 | 1 | no |
| Dilation study | torch, mnist_to_stylus: gain_2.0 vs baseline | +0.44 pp | 0.39 | 1 | no |
| Dilation study | torch, sample_split: aug_stroke_width vs baseline | -1.83 pp | 0.0089 | 0.71 | no |
| Dilation study | torch, sample_split: dilate_k2_i1 vs baseline | +0.17 pp | 0.35 | 1 | no |
| Dilation study | torch, sample_split: dilate_k2_i1_gain_1.5 vs baseline | +0.39 pp | 0.066 | 1 | no |
| Dilation study | torch, sample_split: dilate_k2_i2 vs baseline | +0.18 pp | 0.55 | 1 | no |
| Dilation study | torch, sample_split: dilate_k3_i1 vs baseline | +0.18 pp | 0.47 | 1 | no |
| Dilation study | torch, sample_split: dilate_k3_i2 vs baseline | -0.90 pp | 0.056 | 1 | no |
| Dilation study | torch, sample_split: dilate_k5_i1 vs baseline | -0.90 pp | 0.056 | 1 | no |
| Dilation study | torch, sample_split: erode_k2_i1 vs baseline | -8.43 pp | 1.1e-06 | 0.0002 | yes |
| Dilation study | torch, sample_split: gain_0.7 vs baseline | -0.45 pp | 0.02 | 1 | no |
| Dilation study | torch, sample_split: gain_1.5 vs baseline | +0.07 pp | 0.82 | 1 | no |
| Dilation study | torch, sample_split: gain_2.0 vs baseline | +0.28 pp | 0.11 | 1 | no |
| Regimes | diversity (full data) k=4 vs k=1: unseen_acc | +17.9 pp | 5e-05 | 0.0079 | yes |
| Regimes | diversity (full data) k=4 vs k=1: writer_bias | -22.8 pp | 9.6e-06 | 0.0016 | yes |
| Regimes | diversity (full data) k=4 vs k=1: gap_acc | +0.303 pp | 0.3 | 1 | no |
| Regimes | diversity (full data) k=4 vs k=1: cross_gap | -17.3 pp | 5.4e-05 | 0.0084 | yes |
| Regimes | diversity (size-matched) k=4 vs k=1: gap_acc | +0.571 pp | 0.03 | 1 | no |
| Regimes | diversity (size-matched) k=4 vs k=1: cross_gap | -16.6 pp | 0.00013 | 0.019 | yes |
| Regimes | size sweep (5 epochs) R3 k=3 100% vs 10%: gap_acc | -1.09 pp | 0.1 | 1 | no |
| Regimes | size sweep (5 epochs) R3 k=3 100% vs 10%: gap_loss | +0.0296 nats | 0.054 | 1 | no |
| Regimes | size sweep (5 epochs) R3 k=3 100% vs 10%: writer_bias | -4.48 pp | 0.00072 | 0.087 | no |
| Regimes | size sweep (5 epochs) R3 k=3 100% vs 10%: seen_acc | +34.9 pp | 5.8e-05 | 0.009 | yes |
| Regimes | size sweep (5 epochs) R4 100% vs 10%: gap_acc | -1.03 pp | 0.3 | 1 | no |
| Regimes | size sweep (5 epochs) R4 100% vs 10%: gap_loss | -0.0232 nats | 0.062 | 1 | no |
| Regimes | size sweep (5 epochs) R4 100% vs 10%: writer_bias | -8.32 pp | 0.0056 | 0.47 | no |
| Regimes | size sweep (5 epochs) R4 100% vs 10%: seen_acc | +14.6 pp | 0.00038 | 0.051 | no |
| Regimes | size sweep (equal steps) R3 k=3 100% vs 10%: gap_acc | -5.43 pp | 1.9e-05 | 0.0032 | yes |
| Regimes | size sweep (equal steps) R3 k=3 100% vs 10%: gap_loss | -0.32 nats | 0.0002 | 0.029 | yes |
| Regimes | size sweep (equal steps) R3 k=3 100% vs 10%: writer_bias | -3.36 pp | 0.012 | 0.88 | no |
| Regimes | size sweep (equal steps) R3 k=3 100% vs 10%: seen_acc | +3.17 pp | 0.00051 | 0.065 | no |
| Regimes | size sweep (equal steps) R4 100% vs 10%: gap_acc | -4.78 pp | 2.2e-05 | 0.0035 | yes |
| Regimes | size sweep (equal steps) R4 100% vs 10%: gap_loss | -0.414 nats | 0.00017 | 0.025 | yes |
| Regimes | size sweep (equal steps) R4 100% vs 10%: writer_bias | -2.45 pp | 0.058 | 1 | no |
| Regimes | size sweep (equal steps) R4 100% vs 10%: seen_acc | +2.28 pp | 0.00084 | 0.095 | no |
| Regimes | R5 (1 writer) vs R2: unseen_acc | +25 pp | 7.8e-06 | 0.0014 | yes |
| Regimes | R5 (1 writer) vs R2: mnist_acc | +64.2 pp | 4.8e-07 | 8.7e-05 | yes |
| Regimes | R5 (1 writer) vs R2: writer_bias | -25.1 pp | 1.7e-05 | 0.0029 | yes |
| Regimes | R5 (3 writers) vs R3 k=3: unseen_acc | +11 pp | 5.1e-06 | 0.0009 | yes |
| Regimes | R5 (3 writers) vs R3 k=3: mnist_acc | +54.3 pp | 4.5e-07 | 8.3e-05 | yes |
| Regimes | R5 (3 writers) vs R3 k=3: writer_bias | -11.1 pp | 2.5e-07 | 4.6e-05 | yes |
| Regimes | arm dilate_k3_i1 vs baseline, R1: unseen_acc | -2.55 pp | 0.0027 | 0.26 | no |
| Regimes | arm norm_mnist vs baseline, R1: unseen_acc | +8.1 pp | 0.00036 | 0.05 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R1: unseen_acc | +3.59 pp | 0.013 | 0.94 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R2: unseen_acc | -0.72 pp | 0.18 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R2: unseen_acc | +1.5 pp | 0.015 | 1 | no |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R2: unseen_acc | +1.91 pp | 0.016 | 1 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R3 k=2: unseen_acc | +0.117 pp | 0.86 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R3 k=2: unseen_acc | +2.71 pp | 0.00055 | 0.069 | no |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R3 k=2: unseen_acc | +3.18 pp | 0.0022 | 0.22 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R3 k=3: unseen_acc | +0.92 pp | 0.028 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R3 k=3: unseen_acc | +2.21 pp | 7.6e-05 | 0.012 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R3 k=3: unseen_acc | +2.7 pp | 0.0023 | 0.23 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R3 k=4: unseen_acc | +1.03 pp | 0.032 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R3 k=4: unseen_acc | +1.89 pp | 0.0093 | 0.72 | no |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R3 k=4: unseen_acc | +2.41 pp | 0.023 | 1 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R4: mnist_acc | +11.2 pp | 0.0068 | 0.55 | no |
| Regimes | arm norm_mnist vs baseline, R4: mnist_acc | +16 pp | 0.00074 | 0.089 | no |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R4: mnist_acc | +22.3 pp | 9.6e-05 | 0.014 | yes |
| Regimes | arm dilate_k3_i1 vs baseline, R3s k=2: unseen_acc | +0.168 pp | 0.73 | 1 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R3s k=3: unseen_acc | -0.762 pp | 0.2 | 1 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R3s k=4: unseen_acc | +0.476 pp | 0.35 | 1 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R5 (1 writer): unseen_acc | -1.33 pp | 0.0011 | 0.12 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R5 (3 writers): unseen_acc | -0.874 pp | 0.00078 | 0.092 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R1m n=1712: unseen_acc | -1.26 pp | 0.55 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R1m n=1712: unseen_acc | +20.5 pp | 1.9e-05 | 0.0031 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R1m n=1712: unseen_acc | +16.7 pp | 0.00032 | 0.044 | yes |
| Regimes | arm dilate_k3_i1 vs baseline, R1m n=3424: unseen_acc | -2.66 pp | 0.33 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R1m n=3424: unseen_acc | +19.5 pp | 3e-05 | 0.0049 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R1m n=3424: unseen_acc | +11.8 pp | 0.0012 | 0.13 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R1m n=5136: unseen_acc | -1.51 pp | 0.49 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R1m n=5136: unseen_acc | +16.4 pp | 0.00011 | 0.017 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R1m n=5136: unseen_acc | +11 pp | 0.0089 | 0.71 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R1m n=6848: unseen_acc | -1.52 pp | 0.036 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R1m n=6848: unseen_acc | +17.5 pp | 3.8e-05 | 0.006 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R1m n=6848: unseen_acc | +12.9 pp | 0.00052 | 0.066 | no |
| Regimes | arm dilate_k3_i1 vs baseline, R2: mnist_acc | +4.8 pp | 0.017 | 1 | no |
| Regimes | arm norm_mnist vs baseline, R2: mnist_acc | +8.23 pp | 0.00031 | 0.043 | yes |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R2: mnist_acc | +11.7 pp | 0.00036 | 0.05 | yes |
| Regimes | arm dilate_k3_i1 vs baseline, R3 k=3: mnist_acc | +10.7 pp | 0.0033 | 0.3 | no |
| Regimes | arm norm_mnist vs baseline, R3 k=3: mnist_acc | +11.7 pp | 0.00052 | 0.066 | no |
| Regimes | arm norm_mnist_dilate_k3_i1 vs baseline, R3 k=3: mnist_acc | +19.6 pp | 0.00039 | 0.052 | no |
| Regimes | R5 sweep 1 writer lr×0.1 5 ep: forgetting > 0 | +0.268 pp | 0.034 | 1 | no |
| Regimes | R5 sweep 1 writer lr×0.1 15 ep: unseen gain vs R2 | +24.8 pp | 8.1e-06 | 0.0014 | yes |
| Regimes | R5 sweep 1 writer lr×0.1 15 ep: forgetting > 0 | +0.879 pp | 0.0053 | 0.46 | no |
| Regimes | R5 sweep 1 writer lr×0.3 5 ep: unseen gain vs R2 | +24.8 pp | 8.8e-06 | 0.0015 | yes |
| Regimes | R5 sweep 1 writer lr×0.3 5 ep: forgetting > 0 | +0.985 pp | 0.0024 | 0.24 | no |
| Regimes | R5 sweep 1 writer lr×0.3 15 ep: unseen gain vs R2 | +24.1 pp | 9.9e-06 | 0.0017 | yes |
| Regimes | R5 sweep 1 writer lr×0.3 15 ep: forgetting > 0 | +2.84 pp | 0.00026 | 0.036 | yes |
| Regimes | R5 sweep 1 writer lr×1 5 ep: unseen gain vs R2 | +23.7 pp | 1.4e-05 | 0.0024 | yes |
| Regimes | R5 sweep 1 writer lr×1 5 ep: forgetting > 0 | +4.01 pp | 0.00017 | 0.026 | yes |
| Regimes | R5 sweep 1 writer lr×1 15 ep: unseen gain vs R2 | +23.5 pp | 8.9e-06 | 0.0015 | yes |
| Regimes | R5 sweep 1 writer lr×1 15 ep: forgetting > 0 | +5.29 pp | 0.00017 | 0.025 | yes |
| Regimes | R5 sweep 3 writers lr×0.1 5 ep: forgetting > 0 | +0.344 pp | 0.022 | 1 | no |
| Regimes | R5 sweep 3 writers lr×0.1 15 ep: unseen gain vs R3 k=3 | +10.8 pp | 3.7e-06 | 0.00066 | yes |
| Regimes | R5 sweep 3 writers lr×0.1 15 ep: forgetting > 0 | +2.14 pp | 0.0013 | 0.13 | no |
| Regimes | R5 sweep 3 writers lr×0.3 5 ep: unseen gain vs R3 k=3 | +10.8 pp | 5e-06 | 0.00088 | yes |
| Regimes | R5 sweep 3 writers lr×0.3 5 ep: forgetting > 0 | +2.02 pp | 0.0016 | 0.16 | no |
| Regimes | R5 sweep 3 writers lr×0.3 15 ep: unseen gain vs R3 k=3 | +10.3 pp | 7.9e-06 | 0.0014 | yes |
| Regimes | R5 sweep 3 writers lr×0.3 15 ep: forgetting > 0 | +5.38 pp | 8.1e-05 | 0.012 | yes |
| Regimes | R5 sweep 3 writers lr×1 5 ep: unseen gain vs R3 k=3 | +10.1 pp | 5.8e-06 | 0.001 | yes |
| Regimes | R5 sweep 3 writers lr×1 5 ep: forgetting > 0 | +6.82 pp | 0.00024 | 0.034 | yes |
| Regimes | R5 sweep 3 writers lr×1 15 ep: unseen gain vs R3 k=3 | +10 pp | 3.2e-05 | 0.0051 | yes |
| Regimes | R5 sweep 3 writers lr×1 15 ep: forgetting > 0 | +8.8 pp | 4.3e-06 | 0.00077 | yes |
| Domain gap | s2m: norm_train vs baseline | +18.50 pp | 0.001 | 0.11 | no |
| Domain gap | s2m: norm_dil_train vs baseline | +22.28 pp | 0.001 | 0.12 | no |
| Domain gap | s2m: normboth vs baseline | +19.59 pp | 0.00081 | 0.094 | no |
| Domain gap | s2m: normboth_dil vs baseline | +23.85 pp | 0.00077 | 0.092 | no |
| Domain gap | s2m: normboth_rdil vs baseline | +25.48 pp | 0.0028 | 0.26 | no |
| Domain gap | s2m: normboth_dil_elastic vs baseline | +33.94 pp | 0.00045 | 0.059 | no |
| Domain gap | s2m: normboth_dil_noise vs baseline | +23.46 pp | 0.0013 | 0.14 | no |
| Domain gap | s2m: normboth_dil_blur vs baseline | +23.20 pp | 0.00069 | 0.085 | no |
| Domain gap | s2m: normboth_deslant_dil vs baseline | +28.63 pp | 0.00041 | 0.054 | no |
| Domain gap | s2m: normboth_dil_affine_elastic vs baseline | +34.43 pp | 0.00048 | 0.062 | no |
| Domain gap | s2m: normboth_dil_affine_noise_blur vs baseline | +32.99 pp | 0.00028 | 0.04 | yes |
| Domain gap | s2m: normboth_deslant_dil_affine vs baseline | +34.49 pp | 0.00093 | 0.11 | no |
| Domain gap | s2m: normboth_rdil_affine vs baseline | +36.29 pp | 0.00042 | 0.055 | no |
| Domain gap | s2m: f_center vs baseline | +16.57 pp | 0.003 | 0.28 | no |
| Domain gap | s2m: f_aspect vs baseline | +2.89 pp | 0.31 | 1 | no |
| Domain gap | s2m: f_slant vs baseline | -0.76 pp | 0.8 | 1 | no |
| Domain gap | s2m: f_width vs baseline | +12.69 pp | 0.026 | 1 | no |
| Domain gap | s2m: f_intensity vs baseline | -0.28 pp | 0.79 | 1 | no |
| Domain gap | s2m: f_soft vs baseline | +5.42 pp | 0.031 | 1 | no |
| Domain gap | s2m: c2_center_aspect vs baseline | +20.30 pp | 0.0013 | 0.14 | no |
| Domain gap | s2m: c3_+slant vs baseline | +22.17 pp | 0.001 | 0.12 | no |
| Domain gap | s2m: c4_+width vs baseline | +24.42 pp | 0.0031 | 0.28 | no |
| Domain gap | s2m: c5_+intensity vs baseline | +23.77 pp | 0.00079 | 0.093 | no |
| Domain gap | s2m: c6_+soft vs baseline | +22.19 pp | 0.0018 | 0.18 | no |
| Domain gap | m2s: normboth vs baseline | +9.61 pp | 0.0067 | 0.55 | no |
| Domain gap | m2s: normboth_testdil vs baseline | +6.17 pp | 0.02 | 1 | no |
| Domain gap | m2s: normboth_testdil2 vs baseline | +8.62 pp | 0.016 | 1 | no |
| Domain gap | m2s: normboth_rerode vs baseline | +9.45 pp | 0.0099 | 0.75 | no |
| Domain gap | m2s: normboth_contrast_hard vs baseline | +11.38 pp | 0.0046 | 0.41 | no |
| Domain gap | m2s: normboth_affine vs baseline | +9.05 pp | 0.0053 | 0.46 | no |
| Domain gap | m2s: normboth_elastic vs baseline | +5.51 pp | 0.055 | 1 | no |
| Domain gap | m2s: normboth_noise vs baseline | +10.13 pp | 0.0056 | 0.47 | no |
| Domain gap | m2s: normboth_affine_erode_contrast vs baseline | +8.49 pp | 0.0097 | 0.74 | no |
| Domain gap | m2s: normboth_testdil_affine vs baseline | +6.27 pp | 0.012 | 0.9 | no |
| Domain gap | m2s: normboth_testdil_affine_elastic vs baseline | +2.01 pp | 0.44 | 1 | no |
| Domain gap | m2s: normboth_deslant_testdil_affine vs baseline | +3.33 pp | 0.054 | 1 | no |
| Domain gap | reverse degradation: thin vs clean (R1 models) | -3.70 pp | 0.003 | 0.28 | no |
| Domain gap | reverse degradation: hard vs clean (R1 models) | -0.22 pp | 0.012 | 0.86 | no |
| Domain gap | reverse degradation: dim vs clean (R1 models) | -0.02 pp | 0.43 | 1 | no |
| Domain gap | reverse degradation: uncenter vs clean (R1 models) | -8.29 pp | 0.00024 | 0.034 | yes |
| Domain gap | reverse degradation: narrow vs clean (R1 models) | -2.77 pp | 0.00058 | 0.072 | no |
| Domain gap | reverse degradation: upright vs clean (R1 models) | +0.02 pp | 0.61 | 1 | no |
| Domain gap | reverse degradation: all vs clean (R1 models) | -20.50 pp | 4.2e-06 | 0.00075 | yes |
