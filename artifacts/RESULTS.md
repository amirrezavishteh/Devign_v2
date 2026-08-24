# Devign reproduction — results

Reproduction of Zhou et al., *Devign: Effective Vulnerability Identification by Learning
Comprehensive Program Semantics via Graph Neural Networks*, NeurIPS 2019.

**Status: Phase 1 in progress.** Sections marked *not measured* are not yet run. No number in this
document is estimated, extrapolated, or carried over from a different configuration.

---

## 0. What can and cannot be reproduced

The paper reports four projects. **Only two were ever released.** FFmpeg and QEMU are public as
CodeXGLUE `google/code_x_glue_cc_defect_detection`; Linux Kernel and Wireshark were never
published.

| Paper column | Reproducible | Why |
|---|---|---|
| QEMU | yes | released |
| FFmpeg | yes | released |
| Linux Kernel | **no** | never released — not reproducible |
| Wireshark | **no** | never released — not reproducible |
| Combined | partial | can only pool the two released projects, so it is **not** the paper's Combined |

Any table below reporting "Combined" means QEMU + FFmpeg only. It is not comparable to the
paper's Combined column and is never presented as such.

### Corpus actually used

| | count |
|---|---|
| functions in the release | 27,318 |
| after exact-duplicate removal | 27,258 |
| positive rate | 45.58% |
| graphs successfully built | 20,657 (75.8%) |
| dropped: >500 nodes (paper's filter) | 5,212 (19.1%) |
| dropped: unparseable | 1,389 (5.1%) |
| final split (CodeXGLUE partition) | 16,536 train / 2,071 val / 2,050 test |
| training-split positive rate | 43.21% |

---

## 1. Readout behaviour at initialisation

Reproduces and extends the audit's Eq. 9 finding. Four readouts on an **identical trunk, identical
initialisation, identical batch**, plus the two real models end-to-end. Real QEMU+FFmpeg batch:
128 graphs × 171 padded nodes, 62,538 edges, 48.4% positive. Seed 42, A100.

Regenerate with `python -m scripts.measure_readout --config configs/a100_codexglue.yaml`.

| readout | logit range | prob spread | loss | trunk ‖grad‖ | vs linear control |
|---|---|---|---|---|---|
| `conv_affine_off` — **Eq. 9 as written** | 1.80e-03 | 4.51e-04 | 0.6931 | 1.57e-04 | **383.8× weaker** |
| `conv_affine_on` — Eq. 9 + repo's affine | 9.01e-02 | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| `ggrn_sum` — **Eq. 5 as written** | 3.65e+01 | 0.00e+00 | **24.7238** | 3.29e+01 | 545.6× stronger |
| `linear_head` — control | 7.52e-01 | 1.86e-01 | 0.7008 | 6.03e-02 | (reference) |
| `devign (full)` | 9.01e-02 | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| `ggrn (full)` | 4.15e+01 | 0.00e+00 | **24.6154** | 3.29e+01 | 545.7× stronger |

`ln(2) = 0.6931` is the loss of a model at chance. Far above it at initialisation means the
readout is **saturated**, not learning quickly.

### Two findings, not one

**Eq. 9 collapses.** Logits span 1.8e-03 across the whole batch, every probability lands inside
[0.4996, 0.5000], and loss is exactly ln(2) — the model emits 0.5 for everything. The gradient
reaching the GGNN trunk is **384× weaker** than through a plain linear head on the same pooled
features. This is larger than the 53× the prior audit measured; on real data with real graph
sizes the attenuation is worse. Cause is structural: Eq. 9 ends in `MLP(Z) ⊙ MLP(Y)`, both heads
start near zero, and since ∂(zy)/∂z = y and ∂(zy)/∂y = z, each branch's gradient is scaled by the
other branch's smallness — the trunk is starved through both paths at once.

**Eq. 5 saturates — this is new.** Flat summation does not merely avoid Eq. 9's problem; it fails
the opposite way. It sums an *unnormalised* per-node logit over up to 500 nodes, so logits reach
+41, every probability pins at 1.0, and loss is **24.6 against ln(2) = 0.693**. At initialisation
it calls every graph vulnerable with total confidence, on a split that is 43.2% positive. Its
large trunk gradient is a saturated sigmoid, not health.

**Consequence for the paper's Q2.** "Does the Conv module beat flat summation?" is a comparison
between two badly-conditioned readouts, not between a good one and a bad one. Any three-arm
comparison that adds a new readout to these two is measuring against two broken baselines.

Not fixed here: normalising Eq. 5's sum by node count would deviate from the equation as written,
so it is recorded rather than silently applied. See `models/devign.py::GgrnModel`.

---

## 2. Reproduction vs the paper — *not measured*

Awaiting the seed sweeps. Will report accuracy and F1 at threshold 0.5 (the only figures
comparable to the paper), F1 at the validation-tuned threshold, ROC-AUC, PR-AUC, and the
majority-class baseline, as mean ± std over ≥3 seeds.

## 3. `paper_faithful` vs `repo_default` — *not measured*

## 4. Random vs commit-disjoint leakage gap — *not measured*

## 5. Training curves — partially available

Every run writes `training_curve.csv` (train loss, val AUC/PR-AUC/F1/accuracy, probability spread,
LR, mean trunk gradient norm, OOM-skipped count). Devign seed 1, epoch 1, repo-default config:

| epoch | train loss | val AUC | val F1 | prob spread | trunk ‖grad‖ |
|---|---|---|---|---|---|
| 1 | 0.6787 | 62.47 | 6.24 | 0.440 | 5.46e-02 |

The probability spread of 0.440 at epoch 1 is the affine fix working: the same architecture with
`logit_affine: false` holds the entire batch inside a 4.5e-04 band.

---

## Reproducibility

Every `meta.json` carries a manifest: git SHA, resolved-config hash, library versions, seed,
device, and a hash of the ordered function IDs in each split. Two runs are comparable only if
their split hashes match (`training.manifest.compare_manifests`).

**Determinism gate: PASS on the A100** — two same-seed runs produced byte-identical `metrics.json`
and `model.pt` on real data. Verify with `python -m scripts.check_determinism`.

Hardware: 2× A100-SXM4-80GB, driver 570.86.10, CUDA 12.4, Python 3.10.12, torch 2.5.1.
TF32 is disabled so the A100 and the development laptop compute identical arithmetic.
