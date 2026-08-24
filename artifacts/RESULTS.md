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

## 2. Reproduction vs the paper

CodeXGLUE split, Combined (QEMU + FFmpeg), **mean ± std over 3 seeds**. `test@tuned` applies a
threshold fitted on validation to the held-out test split, so it is unbiased. `val@0.5` is the
fixed-threshold figure and is the one comparable to the paper.

| metric | Devign test@tuned | Devign val@0.5 | Ggrn test@tuned | majority class |
|---|---|---|---|---|
| accuracy | 62.76 ± 1.43 | 64.69 ± 0.42 | 61.79 ± 0.29 | **56.05** |
| F1 | 57.56 ± 2.57 | 55.86 ± 2.12 | 57.16 ± 2.06 | **0.00** |
| precision | 57.69 ± 1.78 | 58.15 ± 1.35 | 56.38 ± 0.85 | 0.00 |
| recall | 57.60 ± 5.01 | 53.99 ± 5.22 | 58.16 ± 4.92 | 0.00 |
| ROC-AUC | 68.67 ± 1.54 | 70.83 ± 0.21 | 67.08 ± 0.87 | 50.00 |
| PR-AUC | 63.15 ± 1.95 | 63.11 ± 0.49 | 60.48 ± 1.94 | 43.95 |

Manifests agree across all six runs: identical split hashes, identical seedless config hash, no
run excluded.

### Against the paper

The paper reports **74.33 accuracy / 73.07 F1** on QEMU. This reproduction reaches **62.76 /
57.56** on the pooled QEMU+FFmpeg CodeXGLUE split — roughly 11.6 accuracy points and 15.5 F1
points below. Published reproductions of this dataset cluster at 60–66% accuracy (CodeBERT ≈62%),
so this sits inside the normal band and the paper's figure is the outlier. It is not a like-for-
like cell: the paper's QEMU column is a different split of a different project subset, and no
honest comparison closes that gap.

The number that matters for "did it learn anything" is the majority-class row: 56.05 accuracy at
0.00 F1. Both models clear it, so the models are doing real work — but 62.76 against a 56.05 floor
is a 6.7-point margin, not the 18-point margin the paper's number would imply.

### Q2: does the Conv module beat flat summation? — **no, not reproducibly**

The paper's own ablation claims the Conv module adds **+4.66 accuracy and +6.37 F1** over Ggrn.
Paired per-seed differences (Devign − Ggrn, test @ tuned threshold):

| metric | seed 1 | seed 2 | seed 3 | mean | Devign wins |
|---|---|---|---|---|---|
| accuracy | +1.17 | +2.05 | −0.29 | **+0.98** | 2/3 |
| F1 | −2.59 | +5.73 | −1.93 | **+0.40** | **1/3** |
| ROC-AUC | +1.99 | +3.72 | −0.95 | +1.59 | 2/3 |
| PR-AUC | +3.16 | +6.21 | −1.36 | +2.67 | 2/3 |

**The sign flips across seeds on every metric.** Devign wins F1 on one seed of three despite a
positive mean, and the per-seed spread (±2.57 for Devign, ±2.06 for Ggrn) is larger than the mean
difference on every metric except PR-AUC. The claimed +6.37 F1 advantage does not appear; +0.40
with an inconsistent sign is indistinguishable from noise.

No significance test is reported here, deliberately. With 3 seeds a Wilcoxon signed-rank test has
a minimum attainable p of 0.25, so it cannot reject anything and quoting it would dress noise as
statistics. Phase 3 runs 5 seeds, where the test becomes meaningful.

This is the context in which any new readout must be judged: the incumbent's advantage over a flat
sum is already within noise, so "beats Eq. 9" is a weaker claim than it sounds.

## 3. `paper_faithful` vs `repo_default`

### Data is identical; only the features differ

Both arms are prepared from the same functions and land on identical splits
(16,536 / 2,071 / 2,050), so any delta is attributable to the configuration rather than the data.

What `embedding.internal_code: zero` does, measured on the same function (id 13111, 172 nodes) in
both prepared splits:

| arm | nodes with an all-zero code half | mean abs code value |
|---|---|---|
| `repo_default` (`mean_standardized`) | 0.0% | 0.724 |
| `paper_faithful` (`zero`) | **44.2%** | 0.193 |

Under the paper's literal reading, internal AST nodes carry Type only. With `time_steps: 6`
against a median AST depth of 11, the upper half of a typical tree never receives a token vector
from anywhere — it is not that the information arrives late, it is that it does not arrive.

### Detection numbers

Devign, CodeXGLUE split, 3 seeds each, held-out test at the tuned threshold. `paper_faithful`
does no threshold tuning (all three seeds sit at 0.500, as the paper does).

| metric | `paper_faithful` | `repo_default` | delta | majority |
|---|---|---|---|---|
| accuracy | 57.90 ± 0.74 | **62.76 ± 1.43** | −4.86 | 56.05 |
| F1 | **58.36 ± 0.23** | 57.56 ± 2.57 | **+0.80** | 0.00 |
| precision | 51.63 ± 0.69 | **57.69 ± 1.78** | −6.06 | 0.00 |
| recall | **67.11 ± 0.55** | 57.60 ± 5.01 | +9.51 | 0.00 |
| ROC-AUC | 63.30 ± 0.79 | **68.67 ± 1.54** | −5.36 | 50.00 |
| PR-AUC | 55.96 ± 1.13 | **63.15 ± 1.95** | −7.19 | 43.95 |

**The paper's configuration scores HIGHER F1 while being a worse classifier.** That is the whole
argument for not selecting on F1@0.5, made visible: `paper_faithful` gains 9.51 points of recall
and loses 6.06 of precision — it over-predicts the positive class. F1 rewards that trade; accuracy
(−4.86), ROC-AUC (−5.36) and PR-AUC (−7.19) all say the ranking is genuinely worse.

ROC-AUC is the honest summary because it is threshold-free and a constant predictor scores exactly
50 no matter what constant it emits. On that measure the paper's configuration is **5.36 points
worse**, and its accuracy of 57.90 clears the 56.05 majority-class floor by under two points.

This is a compound of six reversed deviations (no logit affine, lr 1e-4, no schedule, F1
selection, no threshold tuning, `internal_code: zero`, feature-axis convolution, max-node readout),
so it does not attribute the effect to any one of them. `configs/a100_conv_affine_off.yaml`
isolates the affine term alone and runs in Phase 3.

The dead start is visible in the training curve: `paper_faithful` begins at probability spread
**0.062** and F1 **0.00** at epoch 1, against `repo_default`'s 0.440 and 6.24.

## 4. Leakage: the released split shares commits between train and test

### The channel, measured directly

`python -m scripts.measure_commit_overlap --config configs/a100_codexglue.yaml`

27,258 functions across **12,293 distinct commits**; **3,729 commits touch more than one
function**, and those are the ones a split can scatter.

| split | n_train | n_eval | eval functions from a **training commit** | eval body duplicated in train |
|---|---|---|---|---|
| **codexglue** (the released split) | 21,808 | 2,726 | **1,776 (65%)** | 6 (0%) |
| random 75/12.5/12.5 | 20,443 | 3,408 | **2,205 (65%)** | 6 (0%) |
| commit-disjoint | 20,443 | 3,407 | 0 (0%) | 2 (0%) |

**Roughly two thirds of every evaluation function comes from a commit the model also trained on** —
on the CodeXGLUE release as much as on a random re-split. Near-duplicate bodies are negligible
(0%), so the channel is commit membership, not copy-pasted code: one fix commit touches several
functions, and a random assignment puts siblings on both sides. The model can recognise the commit
rather than the flaw.

This applies to the **released partition**, which is the split every published number on this
dataset is measured on.

### What closing it costs — *partial*

Devign on the commit-disjoint split, training in progress at time of writing:

| | CodeXGLUE split | commit-disjoint |
|---|---|---|
| best val ROC-AUC | 70.83 ± 0.21 | **54.64** (in progress) |

A ROC-AUC of 54.64 is 4.6 points above chance. On the released split the same architecture, data
and code reach 70.83. **The gap is roughly 16 AUC points**, and it is the clearest single statement
this reproduction can make about what the benchmark measures.

Final numbers, with the gap computed against the 3-seed mean rather than a single run, land here
when the run completes.

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
