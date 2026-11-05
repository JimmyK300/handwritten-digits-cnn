# Team's earlier results (reference only)

Copied from the team's Google Drive (folder "Architechture and Hyperparameter Operations / REGIME EVALUATION / Regime_4",
file `regime4_summary_20260517_185311.csv`, and the doc "Evaluation of regime 4 test 1", May 2026).
They were produced with the team's notebook pipeline, 3 runs (seeds 42-44), same architecture
(`masterConfig.json`: conv [32,32], kernels [5,3], hidden [128,32], Adam lr 1e-3, batch 64, 5 epochs).

| Test set | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Minh Cao (held-out split) | 93.01 | 91.61 | 92.66 |
| Minh Ngo | 94.41 | 94.41 | 94.41 |
| Le Kiet | 96.85 | 95.80 | 95.10 |
| Le Khoi | 100.00 | 99.30 | 100.00 |
| Gia Huy | 100.00 | 100.00 | 99.65 |
| MNIST test (10,000) | 51.43 | 57.19 | 42.77 |

Train set: 8,566 images (all 5 writers' train splits); train accuracy 96.8-97.2%.
On MNIST the team's model over-predicted 8 (precision 27.2%, recall 95.4%) and almost never got 7 right
(recall 0.45%), aggregated over the 3 runs. Their write-up proposed thin strokes as the cause and
morphological dilation as the fix, which `custom_to_mnist` in `results/RESULTS.md` tests.
