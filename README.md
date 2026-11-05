# Handwritten digits: CNNs on a team-collected stylus dataset

This repo asks whether **morphological dilation** helps a CNN read stylus-written digits from people it has never seen. It answers with a controlled study. Along the way it shows that the bigger problem is generalizing to new writers.

It extends the team project [`ProfK602170/CustomDataCNN`](https://github.com/ProfK602170/CustomDataCNN) (full commit history kept). It reuses the config-driven CNN and run-tracking structure from [`JimmyK300/autoresearch-cnn-rocm`](https://github.com/JimmyK300/autoresearch-cnn-rocm).

## Key results

The table shows PyTorch CNN accuracy, mean ± std over 5 seeds. Δ is the paired change against the baseline in **percentage points (pp)**. The 3×3 dilation (`dilate_k3_i1`) was chosen as the headline condition before any run.

| Protocol | Baseline | 3×3 dilation, 1 pass | Δ (pp) | Relative | p (paired t, 5 seeds) |
|---|---|---|---|---|---|
| Sample split (team's original) | 96.13 ± 0.52 | 96.31 ± 0.26 | +0.18 ± 0.52 | +0.2% | 0.47 |
| **Leave-one-writer-out** | 88.63 ± 1.36 | 89.44 ± 1.37 | **+0.81 ± 0.75** | +0.9% | 0.07 |
| MNIST → stylus (dilate test images only) | 82.72 ± 2.13 | 80.36 ± 1.90 | −2.36 ± 0.76 | −2.9% | 0.002 |

The from-scratch NumPy CNN shows the same pattern. Leave-one-writer-out: 87.50 ± 1.02 → 88.81 ± 1.24 (+1.31 ± 1.14 pp, p = 0.06). MNIST → stylus: 70.04 ± 4.20 → 73.47 ± 2.56 (+3.43 ± 3.36 pp, 5/5 seeds improved, p = 0.09).

What this shows:

1. **New writers are the real gap.** The same model loses **7.5 pp** (96.1% → 88.6%) when the test writer is held out of training. The team's original split mixed every writer's samples into both train and test, which hid this.
2. **Light dilation helps a little, at most.** One 3×3 pass gains under 1 pp for the PyTorch CNN on unseen writers, and 1.3 pp for the NumPy CNN. Four of five seeds improve, but neither result is significant at p < 0.05.
3. **Heavy dilation hurts.** Two 3×3 passes (equal to one 5×5 pass) lose 2.1 pp on unseen writers and 17.7 pp (−21% relative) in the MNIST → stylus test. Erosion is far worse: thin strokes vanish.
4. **No 20% improvement appears anywhere in this study.** An earlier informal note recalled a ~20% gain from dilation. No setting here reproduces it, in percentage points or relative terms.

Full tables for all 12 conditions, both models and every protocol are in [`results/RESULTS.md`](results/RESULTS.md). Raw per-run records, including confusion matrices, are in [`results/runs.jsonl`](results/runs.jsonl).

![Effect of each condition](results/figures/deltas_torch.png)

![Confusion matrices](results/figures/confusion_torch.png)

## Dataset

The team collected **10,000 stylus-written digits from 5 writers**: Gia Huy, Le Khoi, Le Kiet, Minh Cao and Minh Ngo. Each wrote 200 of each digit 0–9. The committed files for Le Khoi and Minh Ngo hold 1,998 samples each. The writers agreed to publish their samples under their names.

- Location: `mydata/<writer>/`, in MNIST IDX format (`train-*` and `t10k-*` files, 28×28 uint8, white strokes on black). The team's original split is roughly 6:1 within each writer.
- Stroke statistics compared with MNIST: stylus strokes are about 1 px wide. Ink pixels make up about 5% of the image, against 13% for MNIST, and their mean intensity is about 0.50, against 0.71 for MNIST. That gap is why dilation looked like a natural fix.
- `mnist/` holds the standard MNIST files (LeCun, Cortes & Burges) for the cross-domain test.

![Examples](results/figures/examples.png)

## Method

**Models**
- `digitstudy/torch_model.py`: a PyTorch CNN with conv 5×5 (32) → pool → conv 3×3 (32) → pool → FC 128 → FC 32 → 10. This is the team's `sampleModel/config.json`. Training uses Adam, lr 1e-3, batch 64, 5 epochs.
- `digitstudy/numpy_cnn.py`: a CNN written **from scratch in NumPy**. It has im2col convolution, max-pooling, ReLU, dense layers, softmax cross-entropy, backprop and Adam. Layers: conv 5×5 (8) → pool → conv 3×3 (16) → pool → FC 64 → 10. Its gradients are checked against finite differences in `tests/test_numpy_cnn.py`. *(The NumPy CNN and dilation code from the earlier version of the project were lost, so both were rewritten here.)*

**Protocols** (`digitstudy/study.py`)
- `sample_split`: the team's original split, with every writer in both train and test.
- `lowo`: leave-one-writer-out. Train on 4 writers, test on the 5th, for all 5 writers. The score is the macro average over held-out writers.
- `mnist_to_stylus`: train on MNIST (60k), test on all 10k stylus digits. Preprocessing is applied to the test images only.

**Conditions** (`configs/study.json`)
- Applied to train and test images, except in `mnist_to_stylus`:
  - the baseline
  - grayscale dilation with kernel {2, 3, 5} × iterations {1, 2}
  - 2×2 erosion
  - brightness gain {0.7, 1.5, 2.0}
  - 2×2 dilation combined with gain 1.5
- Train-time only: random stroke-width augmentation.

**Statistics.** Every condition runs with 5 seeds (0–4), and the PyTorch and NumPy models train under identical conditions. Results are reported as mean ± sample std. The change against the baseline is paired by seed, with a two-sided paired t-test. p-values are **not** corrected for testing 11 conditions, so only the pre-chosen 3×3 condition should be read as a confirmatory test. Confusion matrices are summed over seeds and folds.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m digitstudy reproduce                     # 450 jobs / 510 runs, ~35 min on a 20-thread CPU
```

`reproduce` runs the study and then writes `results/RESULTS.md`, `results/summary.json` and `results/figures/`. Runs are appended to `results/runs.jsonl` and skipped if already present, so an interrupted run resumes. To start over, delete `results/runs.jsonl`. Other commands:

```bash
python -m digitstudy smoke        # ~10 s end-to-end check (what CI runs)
python -m digitstudy report       # rebuild tables/figures from existing runs
pytest -q                         # unit tests, gradient checks, pipeline test
```

Edit `configs/study.json` to change seeds, conditions, protocols or worker count.

## Repository layout

| Path | Origin |
|---|---|
| `myAPI/`, `main.ipynb`, `mnist/`, `mydata/`, `sampleModel/`, `mini_mnist_data/`, `tests/test_mydata_split.py`, `tests/test_results_export.py` | Original team project (ProfK602170, kinus-is-coding, JimmyK300) |
| `digitstudy/`, `configs/`, `results/`, `tests/test_numpy_cnn.py`, `tests/test_transforms.py`, `tests/test_study_pipeline.py`, `.github/workflows/ci.yml` | Added for this study |

## Limitations

- **Only 5 writers.** Leave-one-writer-out has 5 folds, and per-writer effects range from −1.2 to +2.3 pp. A conclusion about "new writers in general" needs more writers.
- **All five writers are team members.** The test does not cover a broader population or other devices.
- **One architecture and short training** (5 epochs, no tuning per condition). A larger model, or tuning per condition, could change the small effects.
- **The effects are small and uncorrected for multiple comparisons.** The best sweep condition on unseen writers, 2×2 dilation with gain 1.5 (+0.92 pp, p = 0.03), was found after the fact. Treat it as a hypothesis.
- **Other gaps are not tested.** In the MNIST → stylus test, digit size and centering probably matter more than stroke width: stylus digits are smaller and not centred by center of mass like MNIST. This study does not test that.
- `dilate_k3_i2` and `dilate_k5_i1` give identical numbers by construction: two passes of a flat 3×3 square equal one 5×5 pass.

## Credits and license

- Data collection and the original PyTorch pipeline: the CustomDataCNN team (GitHub: ProfK602170, kinus-is-coding, JimmyK300), with the writers named above. Project lead: Cao Chí Minh (JimmyK300).
- The study code, NumPy CNN, tests and CI were added in this repo.
- MNIST: Yann LeCun, Corinna Cortes and Christopher J.C. Burges. It is redistributed here as in the team repo and is not covered by this repo's license.
- Code and the team's stylus data are released under the [MIT License](LICENSE).
