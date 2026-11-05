# Regimes R1–R5

Generated from `results/regimes/runs.jsonl` (1205 runs) with `configs/regimes.json`. All values are mean ± std over 5 seeds (each seed first averaged over writer combinations), in %, except gap loss (nats). Test sets: each writer's held-out split (286 images) and the MNIST test set (10,000). Definitions are in `digitstudy/regime_report.py`.

## 1. All regimes (baseline arm, no preprocessing)

| Regime | Train images | Train acc | Seen held-out acc | Unseen-writer acc | MNIST acc | WriterBias (5 writers) | WriterBias (unseen) | Gap acc (pp) | Gap loss | Cross-writer gap (pp) | Unseen macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 60000 | 99.3 ± 0.1 | 98.8 ± 0.2 | 82.9 ± 2.5 | 98.8 ± 0.2 | 16.9 ± 4.9 | 16.9 ± 4.9 | 0.5 ± 0.1 | 0.014 ± 0.003 | — | 82.2 ± 2.7 |
| R2 | 1713 | 96.1 ± 0.4 | 94.8 ± 0.4 | 67.8 ± 2.2 | 34.4 ± 2.4 | 37.6 ± 1.8 | 22.9 ± 1.1 | 1.3 ± 0.5 | 0.065 ± 0.010 | 27.0 ± 2.1 | 67.0 ± 2.2 |
| R3 k=2 | 3426 | 96.3 ± 0.3 | 94.8 ± 0.4 | 78.6 ± 0.7 | 41.6 ± 1.2 | 26.6 ± 0.8 | 16.0 ± 1.1 | 1.6 ± 0.1 | 0.079 ± 0.005 | 16.1 ± 0.4 | 78.4 ± 0.7 |
| R3 k=3 | 5140 | 96.9 ± 0.3 | 95.3 ± 0.3 | 83.4 ± 0.8 | 44.1 ± 1.8 | 20.1 ± 0.5 | 8.8 ± 0.9 | 1.6 ± 0.1 | 0.084 ± 0.005 | 11.9 ± 0.6 | 83.1 ± 0.8 |
| R3 k=4 | 6853 | 97.1 ± 0.4 | 95.5 ± 0.7 | 85.7 ± 1.4 | 45.6 ± 2.7 | 14.8 ± 0.8 | — | 1.6 ± 0.4 | 0.080 ± 0.007 | 9.8 ± 0.7 | 85.1 ± 1.7 |
| R4 | 8566 | 97.5 ± 0.4 | 96.1 ± 0.5 | — | 48.3 ± 3.3 | 8.3 ± 1.3 | — | 1.3 ± 0.4 | 0.076 ± 0.011 | — | — |
| R3s k=2 | 1712 | 94.3 ± 0.7 | 92.5 ± 0.6 | 75.8 ± 1.6 | 39.6 ± 1.1 | 28.2 ± 2.0 | 16.7 ± 1.7 | 1.8 ± 0.5 | 0.087 ± 0.021 | 16.7 ± 1.2 | 75.6 ± 1.6 |
| R3s k=3 | 1710 | 92.7 ± 0.5 | 90.9 ± 0.5 | 78.7 ± 0.5 | 40.6 ± 3.5 | 23.8 ± 1.5 | 9.9 ± 1.6 | 1.8 ± 0.5 | 0.096 ± 0.019 | 12.2 ± 0.4 | 78.3 ± 0.6 |
| R3s k=4 | 1712 | 92.4 ± 1.2 | 90.5 ± 0.8 | 80.1 ± 1.9 | 42.6 ± 3.3 | 19.4 ± 1.9 | — | 1.8 ± 0.5 | 0.105 ± 0.023 | 10.4 ± 1.6 | 79.1 ± 2.2 |
| R5 (1 writer) | 1713 | 97.4 ± 0.2 | 97.1 ± 0.3 | 92.8 ± 0.5 | 98.5 ± 0.3 | 12.5 ± 1.1 | 12.1 ± 1.1 | 0.3 ± 0.4 | 0.016 ± 0.004 | 4.3 ± 0.6 | 92.7 ± 0.5 |
| R5 (3 writers) | 5140 | 97.8 ± 0.1 | 97.2 ± 0.1 | 94.4 ± 0.4 | 98.5 ± 0.3 | 9.1 ± 0.5 | 5.3 ± 0.5 | 0.6 ± 0.2 | 0.029 ± 0.006 | 2.8 ± 0.4 | 94.4 ± 0.4 |

R1's seen held-out accuracy is MNIST test accuracy and its unseen-writer accuracy covers all 5 writers. R4 has no unseen writer. R3s caps the total training set at one writer's size (≈1712 images) for every k.

## 2. Core hypothesis: writer diversity (baseline arm)

| Series | Writers k | Train images | Unseen-writer acc | WriterBias (5 writers) | WriterBias (unseen) | Gap acc (pp) | Cross-writer gap (pp) | MNIST acc |
|---|---|---|---|---|---|---|---|---|
| full data | 1 | 1713 | 67.8 ± 2.2 | 37.6 ± 1.8 | 22.9 ± 1.1 | 1.3 ± 0.5 | 27.0 ± 2.1 | 34.4 ± 2.4 |
| full data | 2 | 3426 | 78.6 ± 0.7 | 26.6 ± 0.8 | 16.0 ± 1.1 | 1.6 ± 0.1 | 16.1 ± 0.4 | 41.6 ± 1.2 |
| full data | 3 | 5140 | 83.4 ± 0.8 | 20.1 ± 0.5 | 8.8 ± 0.9 | 1.6 ± 0.1 | 11.9 ± 0.6 | 44.1 ± 1.8 |
| full data | 4 | 6853 | 85.7 ± 1.4 | 14.8 ± 0.8 | — | 1.6 ± 0.4 | 9.8 ± 0.7 | 45.6 ± 2.7 |
| size-matched | 1 | 1713 | 67.8 ± 2.2 | 37.6 ± 1.8 | 22.9 ± 1.1 | 1.3 ± 0.5 | 27.0 ± 2.1 | 34.4 ± 2.4 |
| size-matched | 2 | 1712 | 75.8 ± 1.6 | 28.2 ± 2.0 | 16.7 ± 1.7 | 1.8 ± 0.5 | 16.7 ± 1.2 | 39.6 ± 1.1 |
| size-matched | 3 | 1710 | 78.7 ± 0.5 | 23.8 ± 1.5 | 9.9 ± 1.6 | 1.8 ± 0.5 | 12.2 ± 0.4 | 40.6 ± 3.5 |
| size-matched | 4 | 1712 | 80.1 ± 1.9 | 19.4 ± 1.9 | — | 1.8 ± 0.5 | 10.4 ± 1.6 | 42.6 ± 3.3 |

- **full data, unseen-writer accuracy**: Spearman ρ(k, value) = +0.95 (p = 7.48e-11, n = 20 seed-level points); k=4 vs k=1: +17.9 ± 2.2 pp (paired t, p = 5e-05).
- **full data, WriterBias (5 writers)**: Spearman ρ(k, value) = -0.97 (p = 1.97e-12, n = 20 seed-level points); k=4 vs k=1: -22.8 ± 1.8 pp (paired t, p = 9.64e-06).
- **full data, gap acc**: Spearman ρ(k, value) = +0.27 (p = 0.247, n = 20 seed-level points); k=4 vs k=1: +0.3 ± 0.6 pp (paired t, p = 0.299).
- **full data, cross-writer gap**: Spearman ρ(k, value) = -0.97 (p = 1.97e-12, n = 20 seed-level points); k=4 vs k=1: -17.3 ± 2.1 pp (paired t, p = 5.38e-05).
- **size-matched, unseen-writer accuracy**: Spearman ρ(k, value) = +0.92 (p = 1.58e-08, n = 20 seed-level points); k=4 vs k=1: +12.3 ± 2.0 pp (paired t, p = 0.000171).
- **size-matched, WriterBias (5 writers)**: Spearman ρ(k, value) = -0.96 (p = 1.46e-11, n = 20 seed-level points); k=4 vs k=1: -18.3 ± 2.2 pp (paired t, p = 4.91e-05).
- **size-matched, gap acc**: Spearman ρ(k, value) = +0.44 (p = 0.051, n = 20 seed-level points); k=4 vs k=1: +0.6 ± 0.4 pp (paired t, p = 0.0298).
- **size-matched, cross-writer gap**: Spearman ρ(k, value) = -0.94 (p = 9.64e-10, n = 20 seed-level points); k=4 vs k=1: -16.6 ± 2.5 pp (paired t, p = 0.000127).

## 2b. Training-set size sweep (RQ2, baseline arm)

| Regime | Share of each writer's train split | Train images | Gap acc (pp) | Gap loss | Seen held-out acc | Unseen-writer acc | WriterBias (5 writers) | MNIST acc |
|---|---|---|---|---|---|---|---|---|
| R3 k=3 | 10% | 513 | 2.6 ± 1.2 | 0.055 ± 0.021 | 60.4 ± 4.2 | 51.6 ± 3.5 | 24.6 ± 1.3 | 25.7 ± 1.4 |
| R3 k=3 | 25% | 1284 | 1.6 ± 0.6 | 0.082 ± 0.020 | 88.5 ± 0.6 | 75.4 ± 1.8 | 24.8 ± 2.4 | 41.1 ± 2.8 |
| R3 k=3 | 50% | 2570 | 1.7 ± 0.6 | 0.090 ± 0.020 | 93.1 ± 0.5 | 80.2 ± 1.0 | 22.6 ± 1.6 | 42.4 ± 2.5 |
| R3 k=3 | 100% | 5140 | 1.6 ± 0.1 | 0.084 ± 0.005 | 95.3 ± 0.3 | 83.4 ± 0.8 | 20.1 ± 0.5 | 44.1 ± 1.8 |
| R4 | 10% | 855 | 2.4 ± 1.7 | 0.099 ± 0.019 | 81.6 ± 2.5 | — | 16.6 ± 2.7 | 40.8 ± 4.4 |
| R4 | 25% | 2140 | 2.1 ± 0.4 | 0.119 ± 0.018 | 90.3 ± 1.2 | — | 12.9 ± 2.4 | 45.1 ± 2.2 |
| R4 | 50% | 4283 | 1.5 ± 0.5 | 0.092 ± 0.025 | 93.7 ± 1.2 | — | 10.7 ± 0.8 | 45.4 ± 5.4 |
| R4 | 100% | 8566 | 1.3 ± 0.4 | 0.076 ± 0.011 | 96.1 ± 0.5 | — | 8.3 ± 1.3 | 48.3 ± 3.3 |

- **R3 k=3, gap acc**: Spearman ρ(size, value) = -0.27 (p = 0.247); 100% vs 10%: -1.09 ± 1.2 pp (paired t, p = 0.104).
- **R3 k=3, gap loss**: Spearman ρ(size, value) = +0.50 (p = 0.0234); 100% vs 10%: +0.0296 ± 0.025 nats (paired t, p = 0.0544).
- **R3 k=3, WriterBias**: Spearman ρ(size, value) = -0.77 (p = 7.73e-05); 100% vs 10%: -4.48 ± 1.1 pp (paired t, p = 0.000717).
- **R4, gap acc**: Spearman ρ(size, value) = -0.43 (p = 0.0607); 100% vs 10%: -1.03 ± 2 pp (paired t, p = 0.304).
- **R4, gap loss**: Spearman ρ(size, value) = -0.49 (p = 0.0288); 100% vs 10%: -0.0232 ± 0.02 nats (paired t, p = 0.062).
- **R4, WriterBias**: Spearman ρ(size, value) = -0.88 (p = 2.94e-07); 100% vs 10%: -8.32 ± 3.4 pp (paired t, p = 0.00565).

## 3. Preprocessing arms

Unseen-writer accuracy (MNIST accuracy for the R4 row), Δ vs baseline paired by seed (pp, p from paired t-test).

| Regime | baseline | dilate_k3_i1 | norm_mnist | norm_mnist_dilate_k3_i1 |
|---|---|---|---|---|
| R1 | 82.9 ± 2.5 | 80.4 ± 2.6 (-2.5, p=0.00) | 91.0 ± 1.5 (+8.1, p=0.00) | 86.5 ± 1.8 (+3.6, p=0.01) |
| R2 | 67.8 ± 2.2 | 67.1 ± 1.5 (-0.7, p=0.18) | 69.3 ± 1.9 (+1.5, p=0.02) | 69.7 ± 1.6 (+1.9, p=0.02) |
| R3 k=2 | 78.6 ± 0.7 | 78.7 ± 2.0 (+0.1, p=0.86) | 81.3 ± 1.0 (+2.7, p=0.00) | 81.8 ± 1.6 (+3.2, p=0.00) |
| R3 k=3 | 83.4 ± 0.8 | 84.3 ± 1.3 (+0.9, p=0.03) | 85.6 ± 1.0 (+2.2, p=0.00) | 86.1 ± 1.4 (+2.7, p=0.00) |
| R3 k=4 | 85.7 ± 1.4 | 86.7 ± 1.1 (+1.0, p=0.03) | 87.6 ± 1.0 (+1.9, p=0.01) | 88.1 ± 0.5 (+2.4, p=0.02) |
| R4 | 48.3 ± 3.3 | 59.5 ± 3.3 (+11.2, p=0.01) | 64.3 ± 3.3 (+16.0, p=0.00) | 70.7 ± 0.7 (+22.3, p=0.00) |
| R3s k=2 | 75.8 ± 1.6 | 75.9 ± 1.4 (+0.2, p=0.73) | — | — |
| R3s k=3 | 78.7 ± 0.5 | 78.0 ± 1.5 (-0.8, p=0.20) | — | — |
| R3s k=4 | 80.1 ± 1.9 | 80.5 ± 1.7 (+0.5, p=0.35) | — | — |
| R5 (1 writer) | 92.8 ± 0.5 | 91.4 ± 0.4 (-1.3, p=0.00) | — | — |
| R5 (3 writers) | 94.4 ± 0.4 | 93.6 ± 0.6 (-0.9, p=0.00) | — | — |

MNIST accuracy for models trained on stylus data, by arm:

| Regime | baseline | dilate_k3_i1 | norm_mnist | norm_mnist_dilate_k3_i1 |
|---|---|---|---|---|
| R2 | 34.4 ± 2.4 | 39.2 ± 1.7 (+4.8, p=0.02) | 42.6 ± 1.3 (+8.2, p=0.00) | 46.0 ± 1.6 (+11.7, p=0.00) |
| R3 k=3 | 44.1 ± 1.8 | 54.8 ± 2.3 (+10.7, p=0.00) | 55.8 ± 2.3 (+11.7, p=0.00) | 63.7 ± 2.8 (+19.6, p=0.00) |
| R4 | 48.3 ± 3.3 | 59.5 ± 3.3 (+11.2, p=0.01) | 64.3 ± 3.3 (+16.0, p=0.00) | 70.7 ± 0.7 (+22.3, p=0.00) |

## 4. R5: MNIST pre-training + fine-tuning (lr × 0.1)

| Arm | Data | Init | Same/seen-writer acc | Unseen-writer acc | MNIST acc | Forgetting vs R1 (pp) | Unseen acc after 1 epoch |
|---|---|---|---|---|---|---|---|
| baseline | 1 writer | scratch | 94.8 ± 0.4 | 67.8 ± 2.2 | 34.4 ± 2.4 | — | 42.8 ± 2.7 |
| baseline | 1 writer | MNIST | 97.1 ± 0.3 | 92.8 ± 0.5 | 98.5 ± 0.3 | 0.3 ± 0.2 | 90.0 ± 1.2 |
| baseline | 3 writers | scratch | 95.3 ± 0.3 | 83.4 ± 0.8 | 44.1 ± 1.8 | — | 75.0 ± 2.6 |
| baseline | 3 writers | MNIST | 97.2 ± 0.1 | 94.4 ± 0.4 | 98.5 ± 0.3 | 0.3 ± 0.2 | 93.5 ± 0.4 |
| dilate_k3_i1 | 1 writer | scratch | 95.0 ± 0.6 | 67.1 ± 1.5 | 39.2 ± 1.7 | — | 48.7 ± 2.2 |
| dilate_k3_i1 | 1 writer | MNIST | 96.6 ± 0.3 | 91.4 ± 0.4 | 98.7 ± 0.1 | 0.1 ± 0.1 | 87.9 ± 0.9 |
| dilate_k3_i1 | 3 writers | scratch | 95.7 ± 0.3 | 84.3 ± 1.3 | 54.8 ± 2.3 | — | 75.5 ± 2.1 |
| dilate_k3_i1 | 3 writers | MNIST | 96.8 ± 0.2 | 93.6 ± 0.6 | 98.6 ± 0.1 | 0.2 ± 0.1 | 92.2 ± 0.4 |

R1 reference (MNIST-trained, before fine-tuning): MNIST 98.8 ± 0.2, all-writer 82.9 ± 2.5.

- R5 (1 writer) vs R2 (baseline arm), unseen-writer acc: +25.0 ± 1.9 pp (p = 7.82e-06).
- R5 (1 writer) vs R2 (baseline arm), MNIST acc: +64.2 ± 2.4 pp (p = 4.78e-07).
- R5 (1 writer) vs R2 (baseline arm), WriterBias: -25.1 ± 2.3 pp (p = 1.73e-05).
- R5 (3 writers) vs R3 k=3 (baseline arm), unseen-writer acc: +11.0 ± 0.8 pp (p = 5.1e-06).
- R5 (3 writers) vs R3 k=3 (baseline arm), MNIST acc: +54.3 ± 2.0 pp (p = 4.51e-07).
- R5 (3 writers) vs R3 k=3 (baseline arm), WriterBias: -11.1 ± 0.4 pp (p = 2.48e-07).

## 5. Per-class precision / recall / F1 (baseline arm, confusion matrices summed over seeds and combinations)

### R1 (MNIST-trained) on all 5 writers

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 86.4 | 96.1 | 91.0 |
| 1 | 94.5 | 56.4 | 70.7 |
| 2 | 84.9 | 91.6 | 88.1 |
| 3 | 71.8 | 94.3 | 81.5 |
| 4 | 71.1 | 91.0 | 79.9 |
| 5 | 77.1 | 94.9 | 85.1 |
| 6 | 94.4 | 68.1 | 79.1 |
| 7 | 88.5 | 83.7 | 86.0 |
| 8 | 87.9 | 91.9 | 89.9 |
| 9 | 90.9 | 61.0 | 73.0 |
| **Macro avg** | 84.8 | 82.9 | 82.4 |

### R2 (single writer) on unseen writers

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 74.4 | 85.0 | 79.4 |
| 1 | 77.6 | 84.4 | 80.8 |
| 2 | 72.9 | 71.4 | 72.2 |
| 3 | 62.6 | 35.0 | 44.9 |
| 4 | 74.4 | 74.9 | 74.6 |
| 5 | 57.3 | 64.6 | 60.7 |
| 6 | 65.1 | 69.3 | 67.1 |
| 7 | 72.0 | 75.0 | 73.5 |
| 8 | 57.4 | 58.4 | 57.9 |
| 9 | 61.4 | 60.5 | 61.0 |
| **Macro avg** | 67.5 | 67.8 | 67.2 |

### R4 (all writers) on the MNIST test set

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 75.0 | 47.0 | 57.8 |
| 1 | 92.2 | 43.6 | 59.2 |
| 2 | 63.1 | 50.7 | 56.2 |
| 3 | 59.2 | 61.9 | 60.5 |
| 4 | 41.1 | 78.9 | 54.1 |
| 5 | 89.8 | 40.0 | 55.4 |
| 6 | 55.2 | 59.2 | 57.1 |
| 7 | 7.0 | 0.6 | 1.0 |
| 8 | 28.8 | 93.2 | 44.1 |
| 9 | 27.1 | 11.3 | 16.0 |
| **Macro avg** | 53.9 | 48.6 | 46.1 |

## 6. Per-writer held-out accuracy (baseline arm)

| Writer | R1 (MNIST-trained) | R2 own writer | R2 as unseen (mean over other trainers) | R4 (all writers) |
|---|---|---|---|---|
| Gia Huy | 84.5 ± 4.4 | 99.1 ± 0.2 | 83.4 ± 2.1 | 99.9 ± 0.3 |
| Le Khoi | 91.1 ± 4.2 | 99.3 ± 0.4 | 72.2 ± 3.8 | 99.3 ± 0.5 |
| Le Kiet | 80.8 ± 4.0 | 93.8 ± 1.0 | 59.1 ± 2.4 | 94.8 ± 1.1 |
| Minh Cao | 74.4 ± 2.7 | 90.1 ± 0.3 | 62.6 ± 1.6 | 91.6 ± 1.0 |
| Minh Ngo | 83.8 ± 3.8 | 91.8 ± 1.3 | 61.7 ± 1.7 | 95.0 ± 0.8 |

