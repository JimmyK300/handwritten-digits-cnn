# Results

Generated from `results/runs.jsonl` (510 runs) with config `configs/study.json`.

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

## Per held-out writer (PyTorch, leave-one-writer-out)

| Held-out writer | baseline (%) | dilate_k3_i1 (%) | Δ (pp) |
|---|---|---|---|
| Gia Huy | 98.76 ± 0.75 | 98.55 ± 1.50 | -0.21 |
| Le Khoi | 92.50 ± 1.33 | 94.51 ± 2.59 | +2.01 |
| Le Kiet | 82.73 ± 2.07 | 83.83 ± 2.05 | +1.10 |
| Minh Cao | 84.06 ± 1.68 | 82.86 ± 1.43 | -1.20 |
| Minh Ngo | 85.12 ± 2.56 | 87.46 ± 2.08 | +2.34 |

