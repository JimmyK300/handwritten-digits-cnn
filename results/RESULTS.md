# Results

Generated from `results/runs.jsonl` (600 runs) with config `configs/study.json`.

Accuracy is mean ± std over seeds, in %. Δ is paired by seed against `baseline`, in percentage points (pp); `rel` is Δ / baseline accuracy; p is a two-sided paired t-test over seeds, not corrected for the number of conditions tested. Pre-registered primary condition: `dilate_k3_i1`.

## PyTorch CNN

### Sample split (team's original) (5 seeds)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 96.13 ± 0.52 | — |
| dilate_k2_i1 | 96.29 ± 0.24 | +0.17 ± 0.35 pp (+0.2% rel, 3/5 seeds, p=0.346) |
| dilate_k2_i2 | 96.31 ± 0.29 | +0.18 ± 0.62 pp (+0.2% rel, 3/5 seeds, p=0.549) |
| **dilate_k3_i1** | 96.31 ± 0.26 | +0.18 ± 0.52 pp (+0.2% rel, 3/5 seeds, p=0.474) |
| dilate_k3_i2 | 95.23 ± 0.30 | -0.90 ± 0.75 pp (-0.9% rel, 0/5 seeds, p=0.056) |
| dilate_k5_i1 | 95.23 ± 0.30 | -0.90 ± 0.75 pp (-0.9% rel, 0/5 seeds, p=0.056) |
| erode_k2_i1 | 87.69 ± 0.37 | -8.43 ± 0.39 pp (-8.8% rel, 0/5 seeds, p=0.000) |
| gain_0.7 | 95.68 ± 0.47 | -0.45 ± 0.27 pp (-0.5% rel, 0/5 seeds, p=0.020) |
| gain_1.5 | 96.20 ± 1.04 | +0.07 ± 0.66 pp (+0.1% rel, 4/5 seeds, p=0.825) |
| gain_2.0 | 96.41 ± 0.48 | +0.28 ± 0.30 pp (+0.3% rel, 3/5 seeds, p=0.106) |
| dilate_k2_i1_gain_1.5 | 96.52 ± 0.32 | +0.39 ± 0.35 pp (+0.4% rel, 4/5 seeds, p=0.066) |
| aug_stroke_width | 94.29 ± 1.02 | -1.83 ± 0.86 pp (-1.9% rel, 0/5 seeds, p=0.009) |

### Leave-one-writer-out (5 seeds, 5 held-out writers)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 88.63 ± 1.36 | — |
| dilate_k2_i1 | 88.67 ± 1.90 | +0.04 ± 1.01 pp (+0.0% rel, 3/5 seeds, p=0.939) |
| dilate_k2_i2 | 89.23 ± 1.30 | +0.59 ± 0.82 pp (+0.7% rel, 4/5 seeds, p=0.183) |
| **dilate_k3_i1** | 89.44 ± 1.37 | +0.81 ± 0.75 pp (+0.9% rel, 4/5 seeds, p=0.074) |
| dilate_k3_i2 | 86.53 ± 1.60 | -2.11 ± 1.22 pp (-2.4% rel, 0/5 seeds, p=0.018) |
| dilate_k5_i1 | 86.53 ± 1.60 | -2.11 ± 1.22 pp (-2.4% rel, 0/5 seeds, p=0.018) |
| erode_k2_i1 | 74.39 ± 1.22 | -14.25 ± 2.44 pp (-16.0% rel, 0/5 seeds, p=0.000) |
| gain_0.7 | 87.82 ± 1.33 | -0.82 ± 0.86 pp (-0.9% rel, 1/5 seeds, p=0.100) |
| gain_1.5 | 88.96 ± 1.69 | +0.32 ± 0.76 pp (+0.4% rel, 3/5 seeds, p=0.394) |
| gain_2.0 | 89.06 ± 1.61 | +0.43 ± 0.99 pp (+0.5% rel, 3/5 seeds, p=0.385) |
| dilate_k2_i1_gain_1.5 | 89.55 ± 1.05 | +0.92 ± 0.63 pp (+1.0% rel, 4/5 seeds, p=0.031) |
| aug_stroke_width | 85.56 ± 0.55 | -3.07 ± 1.53 pp (-3.4% rel, 0/5 seeds, p=0.011) |

### MNIST → stylus (5 seeds)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 82.72 ± 2.13 | — |
| dilate_k2_i1 | 77.71 ± 2.98 | -5.01 ± 1.24 pp (-6.1% rel, 0/5 seeds, p=0.001) |
| dilate_k2_i2 | 67.43 ± 1.92 | -15.29 ± 1.51 pp (-18.5% rel, 0/5 seeds, p=0.000) |
| **dilate_k3_i1** | 80.36 ± 1.90 | -2.36 ± 0.76 pp (-2.9% rel, 0/5 seeds, p=0.002) |
| dilate_k3_i2 | 65.05 ± 2.33 | -17.68 ± 2.24 pp (-21.4% rel, 0/5 seeds, p=0.000) |
| dilate_k5_i1 | 65.05 ± 2.33 | -17.68 ± 2.24 pp (-21.4% rel, 0/5 seeds, p=0.000) |
| erode_k2_i1 | 22.88 ± 2.55 | -59.84 ± 2.88 pp (-72.3% rel, 0/5 seeds, p=0.000) |
| gain_0.7 | 82.08 ± 3.10 | -0.64 ± 1.01 pp (-0.8% rel, 2/5 seeds, p=0.228) |
| gain_1.5 | 83.26 ± 1.76 | +0.54 ± 0.64 pp (+0.7% rel, 4/5 seeds, p=0.134) |
| gain_2.0 | 83.17 ± 1.83 | +0.44 ± 1.02 pp (+0.5% rel, 3/5 seeds, p=0.388) |
| dilate_k2_i1_gain_1.5 | 77.79 ± 2.85 | -4.94 ± 1.05 pp (-6.0% rel, 0/5 seeds, p=0.000) |
| aug_stroke_width | 84.10 ± 2.50 | +1.38 ± 4.02 pp (+1.7% rel, 3/5 seeds, p=0.487) |

### Stylus → MNIST (team's regime 4) (5 seeds)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 47.35 ± 7.98 | — |
| dilate_k2_i1 | 55.12 ± 3.28 | +7.77 ± 6.52 pp (+18.8% rel, 5/5 seeds, p=0.056) |
| dilate_k2_i2 | 49.64 ± 3.70 | +2.29 ± 7.11 pp (+6.9% rel, 4/5 seeds, p=0.511) |
| **dilate_k3_i1** | 59.94 ± 3.02 | +12.59 ± 8.00 pp (+29.6% rel, 5/5 seeds, p=0.024) |
| dilate_k3_i2 | 41.49 ± 5.68 | -5.86 ± 7.20 pp (-11.0% rel, 1/5 seeds, p=0.143) |
| dilate_k5_i1 | 41.49 ± 5.68 | -5.86 ± 7.20 pp (-11.0% rel, 1/5 seeds, p=0.143) |
| gain_1.5 | 52.27 ± 4.39 | +4.93 ± 3.79 pp (+11.9% rel, 5/5 seeds, p=0.044) |
| gain_2.0 | 54.59 ± 2.80 | +7.24 ± 5.80 pp (+17.6% rel, 5/5 seeds, p=0.049) |
| dilate_k2_i1_gain_1.5 | 56.93 ± 0.96 | +9.58 ± 7.11 pp (+23.0% rel, 5/5 seeds, p=0.040) |
| aug_stroke_width | 54.63 ± 2.44 | +7.28 ± 8.84 pp (+18.4% rel, 4/5 seeds, p=0.139) |
| norm_center | 64.76 ± 3.18 | +17.42 ± 5.93 pp (+39.6% rel, 5/5 seeds, p=0.003) |
| norm_mnist | 67.22 ± 2.96 | +19.87 ± 5.65 pp (+44.8% rel, 5/5 seeds, p=0.001) |
| norm_mnist_dilate_k2_i1 | 65.72 ± 3.27 | +18.37 ± 7.91 pp (+42.2% rel, 5/5 seeds, p=0.007) |
| norm_mnist_dilate_k3_i1 | 69.46 ± 3.19 | +22.11 ± 5.24 pp (+49.5% rel, 5/5 seeds, p=0.001) |

## NumPy from-scratch CNN

### Leave-one-writer-out (5 seeds, 5 held-out writers)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 87.50 ± 1.02 | — |
| dilate_k2_i1 | 88.60 ± 1.54 | +1.10 ± 1.10 pp (+1.3% rel, 4/5 seeds, p=0.089) |
| **dilate_k3_i1** | 88.81 ± 1.24 | +1.31 ± 1.14 pp (+1.5% rel, 4/5 seeds, p=0.062) |

### MNIST → stylus (5 seeds)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 70.04 ± 4.20 | — |
| dilate_k2_i1 | 69.30 ± 2.97 | -0.74 ± 3.24 pp (-0.9% rel, 1/5 seeds, p=0.634) |
| **dilate_k3_i1** | 73.47 ± 2.56 | +3.43 ± 3.36 pp (+5.1% rel, 5/5 seeds, p=0.085) |

### Stylus → MNIST (team's regime 4) (5 seeds)

| Condition | Accuracy (%) | Δ vs baseline |
|---|---|---|
| **baseline** | 50.85 ± 5.81 | — |
| **dilate_k3_i1** | 57.37 ± 2.92 | +6.52 ± 4.85 pp (+13.6% rel, 4/5 seeds, p=0.040) |
| norm_mnist | 63.05 ± 2.80 | +12.20 ± 4.05 pp (+24.9% rel, 5/5 seeds, p=0.003) |
| norm_mnist_dilate_k3_i1 | 63.46 ± 3.59 | +12.61 ± 3.41 pp (+25.5% rel, 5/5 seeds, p=0.001) |

## Stylus → MNIST: the class-7 / class-8 failure

Per-seed metrics from each run's confusion matrix, mean ± std over seeds, in %. `pred share 8` is the fraction of all MNIST test images predicted as 8 (the true share of 8s is 9.7%).

### PyTorch CNN

| Condition | recall 7 | precision 8 | recall 8 | pred share 8 |
|---|---|---|---|---|
| baseline | 0.9 ± 0.6 | 30.3 ± 12.2 | 95.3 ± 4.1 | 35.2 ± 14.3 |
| dilate_k2_i1 | 2.0 ± 1.0 | 47.5 ± 10.3 | 83.2 ± 8.0 | 17.9 ± 4.8 |
| dilate_k2_i2 | 4.1 ± 4.1 | 47.1 ± 10.0 | 37.7 ± 22.0 | 7.3 ± 3.5 |
| dilate_k3_i1 | 22.2 ± 9.2 | 88.4 ± 5.6 | 48.0 ± 15.9 | 5.3 ± 1.9 |
| dilate_k3_i2 | 13.0 ± 3.7 | 38.9 ± 48.7 | 0.1 ± 0.2 | 0.0 ± 0.0 |
| dilate_k5_i1 | 13.0 ± 3.7 | 38.9 ± 48.7 | 0.1 ± 0.2 | 0.0 ± 0.0 |
| gain_1.5 | 2.9 ± 2.3 | 36.5 ± 14.1 | 92.3 ± 5.0 | 27.7 ± 10.2 |
| gain_2.0 | 6.9 ± 4.8 | 40.4 ± 10.2 | 91.0 ± 5.5 | 23.2 ± 6.2 |
| dilate_k2_i1_gain_1.5 | 5.4 ± 2.4 | 57.6 ± 10.4 | 70.1 ± 14.6 | 12.6 ± 4.8 |
| aug_stroke_width | 8.3 ± 9.1 | 54.2 ± 3.5 | 74.7 ± 10.2 | 13.5 ± 2.5 |
| norm_center | 71.9 ± 4.5 | 44.6 ± 4.7 | 89.1 ± 2.4 | 19.7 ± 2.3 |
| norm_mnist | 75.9 ± 5.9 | 64.5 ± 10.8 | 82.8 ± 5.9 | 13.0 ± 3.5 |
| norm_mnist_dilate_k2_i1 | 62.3 ± 6.5 | 72.3 ± 15.4 | 75.5 ± 8.2 | 10.8 ± 3.5 |
| norm_mnist_dilate_k3_i1 | 72.5 ± 6.5 | 91.3 ± 1.8 | 62.0 ± 12.7 | 6.6 ± 1.4 |

### NumPy CNN

| Condition | recall 7 | precision 8 | recall 8 | pred share 8 |
|---|---|---|---|---|
| baseline | 7.1 ± 7.8 | 31.2 ± 11.2 | 95.6 ± 2.9 | 32.8 ± 10.5 |
| dilate_k3_i1 | 21.5 ± 11.8 | 75.9 ± 9.3 | 36.1 ± 9.1 | 4.7 ± 1.4 |
| norm_mnist | 73.8 ± 7.1 | 54.6 ± 10.0 | 78.5 ± 9.6 | 14.2 ± 2.5 |
| norm_mnist_dilate_k3_i1 | 74.3 ± 7.5 | 90.6 ± 4.3 | 44.7 ± 9.3 | 4.8 ± 0.9 |

## Per held-out writer (PyTorch, leave-one-writer-out)

| Held-out writer | baseline (%) | dilate_k3_i1 (%) | Δ (pp) |
|---|---|---|---|
| Gia Huy | 98.76 ± 0.75 | 98.55 ± 1.50 | -0.21 |
| Le Khoi | 92.50 ± 1.33 | 94.51 ± 2.59 | +2.01 |
| Le Kiet | 82.73 ± 2.07 | 83.83 ± 2.05 | +1.10 |
| Minh Cao | 84.06 ± 1.68 | 82.86 ± 1.43 | -1.20 |
| Minh Ngo | 85.12 ± 2.56 | 87.46 ± 2.08 | +2.34 |

