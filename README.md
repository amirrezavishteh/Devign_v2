# Devign — Vulnerability Identification via Graph Neural Networks

An end-to-end, runnable reproduction of **Devign** (Zhou et al., *"Devign: Effective Vulnerability
Identification by Learning Comprehensive Program Semantics via Graph Neural Networks"*, NeurIPS
2019 — [arXiv:1909.03496](https://arxiv.org/abs/1909.03496)) in PyTorch, trained on the **paper
authors' own released dataset** — not a synthetic stand-in.

It encodes each C function as a **composite multi-edge graph** (AST + CFG + DFG + NCS), learns
node representations with a **Gated Graph Recurrent layer**, and classifies whole graphs with the
paper's novel **Conv module**. All three components, the four Table-2 baselines, the Table-3
imbalanced study, the single-edge ablation, and a commit-disjoint leakage check are implemented
and have been run on real data.

The repository goes past reproduction in two directions. It **measures why the paper's readout
trains badly** rather than asserting it, and it adds a **third readout** — gated attention MIL
pooling — that does the same job without the defect and yields statement-level localisation for
free. Full numbers live in [`artifacts/RESULTS.md`](artifacts/RESULTS.md).

---

## 0. What this reproduction found

Every figure below is measured in this repository and regenerable by the command beside it. None
is quoted from the paper or from another reproduction.

### The reproduction lands well below the paper, inside the published band

| | accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **This repro** (Devign, CodeXGLUE test, 3 seeds) | 62.76 ± 1.43 | 57.56 ± 2.57 | 68.67 ± 1.54 | 63.15 ± 1.95 |
| Majority-class baseline | 56.05 | 0.00 | 50.00 | 43.95 |
| Paper (QEMU column) | 74.33 | 73.07 | — | — |

Public reproductions of this dataset cluster at 60–66% accuracy (CodeBERT ≈62%), so this sits in
the normal band and the paper's number is the outlier. The cells are not like-for-like: the
paper's QEMU column is a different split of a different project subset, and no honest comparison
closes that gap. **Linux Kernel and Wireshark were never released and are permanently
unreproducible** — this repo never prints those columns.

### The Conv module's advantage over flat summation does not reproduce

The paper's own ablation claims the Conv module adds **+4.66 accuracy and +6.37 F1** over Ggrn.
Measured pairwise per seed:

| metric | seed 1 | seed 2 | seed 3 | mean | Devign wins |
|---|---|---|---|---|---|
| accuracy | +1.17 | +2.05 | −0.29 | +0.98 | 2/3 |
| F1 | −2.59 | +5.73 | −1.93 | **+0.40** | **1/3** |

The **sign flips across seeds on every metric**. A single run would have shown either a win or a
loss depending on which seed you happened to use.

### Both of the paper's readouts are broken at initialisation — in opposite directions

`python -m scripts.measure_readout --config configs/a100_codexglue.yaml`

| readout | logit range | prob spread | loss | trunk ‖grad‖ | vs linear control |
|---|---|---|---|---|---|
| **Eq. 9** (Conv, as written) | 1.80e-03 | 4.51e-04 | 0.6931 | 1.57e-04 | **384× weaker** |
| Eq. 9 + learnable affine | 9.01e-02 | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| **Eq. 5** (flat summation) | 4.15e+01 | 0.00e+00 | **24.62** | 3.29e+01 | 546× stronger |
| linear control | 7.52e-01 | 1.86e-01 | 0.7008 | 6.03e-02 | (reference) |
| **MIL attention (new)** | 8.58e-01 | 2.11e-01 | 0.7153 | 1.58e-01 | 1.0× — healthy |

`ln(2) = 0.6931` is the loss of a model at chance.

**Eq. 9 collapses.** It ends in `MLP(Z) ⊙ MLP(Y)`; both heads start near zero, and since
∂(zy)/∂z = y and ∂(zy)/∂y = z, each branch is throttled by the other's smallness. Every
probability in the batch lands inside [0.4996, 0.5000].

**Eq. 5 saturates.** It sums an *unnormalised* per-node logit over up to 500 nodes, so logits
reach +41 and every probability pins at 1.0 — it begins by calling every graph vulnerable with
total confidence on a 43.2%-positive split. Its large gradient is a saturated sigmoid, not health.

This matters for the paper's Q2: *"does the Conv module beat flat summation?"* compares two
badly-conditioned readouts, not a good one against a bad one.

### Determinism was not free

`index_add_` message passing was **bitwise different on 20 of 20 repeats** (6e-7 drift per
forward). Replaced with a sorted segment reduction, which is exactly reproducible and also **~6%
faster** — 9.62 ± 0.04 ms against 10.25 ± 0.06 ms. Two same-seed runs now produce byte-identical
`metrics.json` **and** `model.pt` on the A100.

### A tuned threshold was being reported on its own split

`train_model` fitted the decision threshold on validation and then scored validation at it. On one
run that gap was **F1 0.00 (unbiased) vs 68.97 (biased)** on the same split. Metrics now carry the
split they were computed on and the split their threshold came from, and `require_unbiased`
refuses the pair wherever a number enters a table.

---

## 1. Architecture (paper → code)

| Paper component | Section | Implementation |
|---|---|---|
| Graph Embedding Layer (composite semantics) | 2.2 | [devign_data/parser.py](devign_data/parser.py), [devign_data/graph_builder.py](devign_data/graph_builder.py) |
| Node features `x_v` = Code (word2vec, 100-d) ⊕ Type (label enc.) | 2.2.2 | [devign_data/word2vec_embed.py](devign_data/word2vec_embed.py), [models/node_init.py](models/node_init.py) |
| Gated Graph Recurrent layer (Eq. 3–4, T=6, z=200, SUM agg) | 2.3 | [models/ggnn.py](models/ggnn.py) — dense **and** sparse edge-list propagation (see §2) |
| Conv module (Eq. 6–9, dual branch, 2 conv layers, `pool2` = (2,2)/(1,2)) | 2.4 | [models/conv_module.py](models/conv_module.py) |
| Devign model + Ggrn flat-summation variant (Eq. 5) | 2.4 | [models/devign.py](models/devign.py) |
| Baselines: Metrics+XGBoost, BiLSTM, BiLSTM+Att, CNN | 3.2 | [models/baselines.py](models/baselines.py), [models/metrics_xgboost.py](models/metrics_xgboost.py) |
| Training (Adam, lr 1e-4, bs 128 via gradient accumulation, L2, early stop) | 3.3 | [training/trainer.py](training/trainer.py) |
| Checkpoint selection on AUC + validation-tuned decision threshold | — | [training/metrics.py](training/metrics.py), [training/trainer.py](training/trainer.py) |
| Table 2 / Table 3 / ablation / commit-disjoint leakage check | 3.3 | [scripts/](scripts/), [evaluation/](evaluation/) |

### The 7 edge types
`AST, REV_AST, CFG, NCS, DFG_R, DFG_W, DFG_C` share the same node set `V = V_ast`. The paper fixes
`k=7` adjacency matrices but only names 6 representations; we add `REV_AST` (reverse parent↔child)
as the 7th so messages flow both up and down the AST backbone — the standard GGNN choice for
tree-shaped graphs. The adjacency is a raw binary tensor `A ∈ {0,1}^(k×m×m)`, exactly the paper's
formulation (`dataset.add_self_loops` / `dataset.normalize_adj` default off in
[config.yaml](config.yaml); turning them on is an explicit, opt-in deviation).

### Dense vs. sparse message passing
`A` is never actually materialized as a `[B,k,M,M]` tensor during training: at the paper's
`batch_size=128` and the 500-node cap, that tensor alone is 896 MB, and GGNN intermediates push
past several GB — more than an 8 GB GPU can hold for real (non-toy) graphs. `models/ggnn.py`
implements an edge-list (`index_add_`-based) propagation that is **numerically identical** to the
dense form (proved in [tests/test_sparse_equivalence.py](tests/test_sparse_equivalence.py), 7
tests) at `O(E·z)` instead of `O(k·M²)`. A size-bucketed batch sampler plus a hard per-batch
node budget (`dataset.max_nodes_per_batch`) keep peak memory bounded regardless of graph size,
with gradient accumulation preserving the paper's effective `batch_size=128` exactly.

---

## 2. Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `gensim` (word2vec), `tree-sitter` + `tree-sitter-c` (AST parsing),
`scikit-learn`, `xgboost`, `huggingface_hub` + `pyarrow` (real dataset download), `flawfinder`
(Table 3). Python 3.10+.

**Static analyzers for Table 3.** `flawfinder` is a pure-Python pip package and always installs.
`cppcheck` is a separate native binary — install it yourself (`apt-get install cppcheck` on
Linux/Colab, `winget install Cppcheck` where that source works, or a portable build on Windows)
and either put it on `PATH` or point `evaluation.static_analyzer_paths.cppcheck` at the binary in
[config.yaml](config.yaml). If it genuinely isn't available, its Table 3 row renders as
`cppcheck (heuristic-fallback)` — a labelled regex mimic, never presented as the real tool's
result.

> **Joern vs tree-sitter.** The paper uses Joern (a JVM/Scala code-property-graph tool) for
> AST/CFG extraction. To keep this repo self-contained we parse with **tree-sitter** and derive
> CFG/DFG/NCS heuristically in [devign_data/graph_builder.py](devign_data/graph_builder.py) — including the
> paper's parser-error filter (`root.has_error`), which real code actually triggers. To use Joern
> instead, replace `build_graph()` with a Joern CPG exporter producing the same `CodeGraph`. The
> DFG builder in particular is flow-insensitive and scope-blind (a single source-order pass over
> identifiers), so DFG_R/W/C edges are noisier than Joern's — worth keeping in mind when reading
> the ablation.

---

## 3. Data

The original Devign datasets were **manually labeled (~600 man-hours)**. Only two of the paper's
four projects were ever made public — **FFmpeg and QEMU**, as the `google/code_x_glue_cc_defect_detection`
release on HuggingFace, which *is* the Devign authors' own `function.json`. **Linux Kernel and
Wireshark were never released** — including the paper's single best result, Linux F1 84.97 — so
no faithful reproduction can produce those two columns from public data.

`data.source` in [config.yaml](config.yaml) selects where raw functions come from:

- **`devign_release` (default).** Downloads and caches the real FFmpeg+QEMU data via
  [devign_data/hf_devign.py](devign_data/hf_devign.py) (`huggingface_hub` + `pyarrow`, no extra install). All
  three published splits are concatenated and re-split ourselves (Sec 3.3 does its own random
  split, not CodeXGLUE's). **27,258 functions after de-duplication** (FFmpeg 9,726 / QEMU 17,532).
- **`file`.** Point `data.real_data_path` at any `json`/`jsonl`/`csv` with
  `{func, target, project, commit_id}` columns (e.g. a Big-Vul export).
- **`synthetic`.** [devign_data/templates.py](devign_data/templates.py)'s 10 CWE-pattern generator. Offline
  smoke-test only — it is what `--quick` / the test suite exercise, and its numbers are **not**
  vulnerability-detection evidence (the generator is trivially separable; every strong model
  saturates near 100%, including the CNN baseline).

Functions with **> 500 nodes, or that tree-sitter could not parse cleanly, are filtered out**
(`data.max_nodes`, `drop_parse_errors` in `build_graph`) — the paper reports ~15% dropped this way.

### Model selection and the decision threshold (important)
Checkpoints are selected on validation **AUC**, not F1, and this is not a stylistic choice. On this
43–51%-positive data a degenerate "predict almost everything vulnerable" classifier scores
F1 ≈ 60–67% at a 0.5 threshold, and a properly trained model balancing precision against recall
never beats it — so selecting on F1@0.5 restores an early, untrained epoch. Measured on real qemu:
selecting on F1 saved epoch 9 (**53.99%** accuracy) over epoch 70 (**62.99%**). AUC is exactly 50
for any constant predictor and cannot be gamed that way.

After the best checkpoint is restored, the decision threshold is tuned on validation and persisted
to `meta.json`. The objective is **guarded F1** — maximise F1 *subject to accuracy ≥ accuracy@0.5*
— which provably cannot do worse than the 0.5 default on either metric the paper reports (0.5 is
always feasible) and cannot select the degenerate all-positive cut (that craters accuracy). Plain
F1 fails the second property; plain MCC fails the first, and on real data chose threshold 0.996,
trading 14 points of F1 for 4 points of accuracy. Every downstream consumer — Table 2/3, the Q5 holdout, `inference.predict` — reads
that threshold instead of assuming 0.5. For Table 3 the threshold is **recalibrated** on a
validation subsample at the imbalanced 10% prevalence, because a cut tuned at ~46% prevalence drives
F1 to ~0 on a 10%-positive set. `training.class_weighting` is off by default: the paper does no
reweighting, and upweighting positives pushes the model toward exactly the degenerate regime above.

**Split.** Class-stratified **75 / 12.5 / 12.5** (train / val / test) by default — the paper's own
75% training fraction is preserved, but its 25% remainder is halved so early stopping (on val) and
the reported number (on test, untouched by training) don't share data. Set `data.paper_split: true`
for the paper's exact 75/25 with no test set. `data.split_by: commit` produces a **commit-disjoint**
split instead of a random one — see the leakage check in §5.

---

## 4. Running

### Quick smoke run (synthetic data, few epochs — validates wiring, not accuracy)
```bash
python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml'))
cfg['data']['source'] = 'synthetic'
yaml.safe_dump(cfg, open('config_smoke.yaml', 'w'))
"
python -m scripts.reproduce --config config_smoke.yaml --quick
```
This is what the test suite and CI-style checks should use — it never touches the network and
finishes in well under a minute. Its numbers are not evidence of anything; see §3.

### Full study on real data
```bash
# 1. Download the real FFmpeg+QEMU data, build graphs, train word2vec (~2 min on a many-core
#    machine, ~15-25 min on a 2-4 vCPU Colab session).
python -m scripts.prepare_data

# 2. Everything else — Devign + Ggrn + 4 baselines, per-project AND Combined, Table 2/3,
#    the unseen-commit holdout, the single-edge ablation, and the commit-disjoint leakage check.
python -m scripts.reproduce --epochs 100
```

`scripts.reproduce` is **resumable**: every stage (each project × model training run, each
baseline, the ablation, the leakage check) writes its own artifact and is skipped on a re-run if
that artifact already exists — pass `--force` to redo everything, or delete a specific artifact to
redo just that piece. `--epochs N` overrides every model's epoch count uniformly; omit it for the
paper's exact `200`/`patience 100` schedule (`config.yaml`).

Individual stages also run standalone:
```bash
python -m scripts.train --model devign --project qemu       # one project, one model
python -m scripts.train_baselines
python -m scripts.run_imbalanced        # Table 3 vs Cppcheck/Flawfinder
python -m scripts.run_ablation --models devign ggrn --project combined
python -m scripts.run_cve               # Q5 proxy — see §5's caveat
python -m scripts.run_leakage_check     # commit-disjoint Combined, Devign + Ggrn
```

### Machine-specific configs
A config may inherit from another with `extends:`, and should contain **only** the keys that
differ — see [config_a100.yaml](config_a100.yaml):

```yaml
extends: config.yaml
dataset:
  max_nodes_per_batch: 24000   # this machine's GPU budget; everything else is inherited
```

This is not cosmetic. `config_a100.yaml` was previously a full *copy* of `config.yaml`, so
later fixes to `dropout`, `word2vec_min_count` and `early_stopping_patience` never reached the
training box and two full multi-hour runs were spent on stale hyperparameters. Verify inheritance
any time you add a machine config:

```bash
python -c "from training.utils import load_config; c=load_config('config_a100.yaml'); \
print(c['model']['dropout'], c['embedding']['word2vec_min_count'], c['training']['monitor'])"
```

All hyperparameters live in [config.yaml](config.yaml). Results land under `artifacts/<model>/<project>/`
(`model.pt`, `metrics.json`, `meta.json`) plus `artifacts/{table2,table3,cve,leakage_check}.json`
and `artifacts/ablation/ablation.json`.

### Running on Colab (free tier)
[Devign_Colab.ipynb](Devign_Colab.ipynb) runs the whole study on a free Colab T4 GPU. It's written
around free-tier reality: checkpoints/metrics persist to Google Drive so a disconnect doesn't lose
progress, the resumable `reproduce.py` picks up where it left off, and — a nice side effect of the
Linux container — `cppcheck` installs cleanly via `apt-get`, so Table 3 gets the real tool on both
rows instead of the fallback. See the notebook's own markdown cells for the upload/setup steps.

### Single-function inference
```bash
python -m inference.predict --code "void f(const char *s){ char b[16]; strcpy(b,s); }"
python -m inference.predict --file path/to/function.c
```
Outputs the verdict, `P(vulnerable)`, node count, and per-edge-type counts. Loads whichever
project's trained model you point it at (default: the Combined model).

---

## 5. Evaluation outputs and what they mean

- **Table 2** — Accuracy/F1 for QEMU, FFmpeg, and Combined. Devign/Ggrn and every baseline are
  trained **separately per project and once more on the pooled Combined data** — a project's
  column is that project's own model, not a pooled model's per-project breakdown.
- **Table 3** — imbalanced (10% vulnerable) setting vs Cppcheck/Flawfinder, resampled from the
  held-out split.
- **Ablation (Q3)** — single-edge graphs (`AST/CFG/NCS/DFG_R/DFG_W/DFG_C`) vs Composite, Combined,
  for Devign and Ggrn — the composite's own retraining is shared between `run_ablation.py` and
  `reproduce.py` via one artifact path, so the two never disagree.
- **Unseen-commit holdout ("Q5 proxy")** — **not a CVE or zero-day test.** The paper's 40-CVE /
  112-function set was never published and cannot be reconstructed from the released data. What we
  report instead: vulnerable functions whose commit contributed nothing to training. It answers
  "does the model generalize past the fixes it trained on", which is weaker than the paper's
  question and is labelled as such everywhere it's printed or serialized
  ([evaluation/cve_eval.py](evaluation/cve_eval.py)). The previous version of this repo generated
  functions from the *same* templates the model trained on and named them `CVE_qemu_0` — that
  measured memorization of a generator, not generalization, and has been removed entirely.
- **Commit-disjoint leakage check** — a random split can put two functions from the same
  vulnerability-fix commit on both sides of train/val, letting the model partly recognise the
  commit instead of the flaw. `scripts.run_leakage_check` retrains Devign+Ggrn on Combined with a
  commit-disjoint split and reports the accuracy/F1 gap against the random-split number — a large
  gap means the headline Table 2 number is partly commit memorisation.

**Expected honest gap vs. the paper.** Public reproductions on this dataset typically land well
below the paper's 72.26% Combined accuracy, for two structural reasons that are not implementation
bugs: (1) tree-sitter's heuristic CFG/DFG vs. Joern's real code-property graph, and (2) Linux
Kernel and Wireshark — the paper's strongest datasets — are simply unavailable. See §6 for the
actual numbers from this run.

---

## 6. The three readouts

All three share an identical NodeInit + GGNN trunk, so a comparison isolates the readout and
nothing else. Parameter counts are within 0.1% (MIL 630,785 vs Devign 631,410).

| `model.readout` | equation | what it does |
|---|---|---|
| `conv` | Eq. 6–9 | the paper's Conv module: two σ-stacks, two MLP heads, elementwise product |
| `sum` (Ggrn) | Eq. 5 | `Sigmoid(Σ_j MLP([H_j, x_j]))`, masked to real nodes |
| `mil` | — | gated attention pooling (Ilse, Tomczak & Welling, ICML 2018) |

### Why MIL

The Conv module's stated job (Sec 2.4) is to *"select sets of nodes and features that are relevant
to the current graph-level task"*. That is a description of attention-based multiple-instance
pooling, and Eq. 9 is a degenerate implementation of it:

```
e_j   = w^T ( tanh(V h_j) ⊙ sigmoid(U h_j) )     # gated attention, per node
a     = softmax(e over REAL nodes only)           # padded nodes → -inf first
z     = Σ_j a_j h_j                               # bag embedding
logit = Linear(z)
```

It is linear in the node embeddings, so its gradient is alive at initialisation with **no rescue
term** — no `logit_affine`, no scale, no bias. That is hypothesis **H3**, and it is verified
rather than assumed (see the table in §0).

The contribution claimed here is *not* the operator — attention MIL is Ilse et al. 2018, and
attention for vulnerability localisation is LineVul and others. It is: **the paper's central
novelty is a degenerate approximation of a known-better operator, here is the measurement, here is
the principled replacement, and here is the interpretability the paper listed as future work.**

### Localisation comes out for free

Every tree-sitter node carries a byte span, so attention over nodes projects onto source lines
without a second model. `GraphSample.node_lines` carries a 1-based `[start, end]` per node.

Ground truth cannot come from CodeXGLUE — it ships `func` and `target` only, with no `func_after`,
no diff and no line annotation, so **it cannot support a line-level evaluation at all**. This repo
uses PrimeVul's paired split (86–92% measured label accuracy) and derives vulnerable lines by
diffing `func_before` against `func_after`. Lines the patch *deleted or replaced* count; pure
insertions do not, because an added bounds check has no counterpart line in the vulnerable
function.

```bash
python -m scripts.run_localization --config configs/a100_codexglue.yaml     --model mil --model-dir artifacts/seed1/mil/combined     --primevul data/primevul/test_paired.jsonl
```

Scored on **true positives only** — ranking lines inside a function the model called clean proves
nothing — against three baselines: random ranking, a **line-length prior** (if a trivial length
prior matches the attention map, the attention learned nothing), and input-gradient saliency for
the Conv module, labelled a PROXY because Eq. 9 pools its node axis away and exposes no per-node
score at all.

On the attention-is-not-explanation debate (Jain & Wallace 2019; Wiegreffe & Pinter 2019): this
does not assert that attention *is* the explanation. It validates attention against ground-truth
vulnerable lines, which is a different and testable claim.

---

## 7. Project layout

```
devign_data/  parser, graph builder, word2vec, dataset/batching (dense + sparse), line spans,
              hf_devign (real dataset download), primevul (localisation ground truth), prepare
data/         NOT a package -- the dataset directory, gitignored. The source package above is
              named differently so an ignore rule for the dataset can never swallow the code.
models/       node_init, ggnn, conv_module (Eq. 6-9), mil_pool (attention MIL), devign+ggrn+mil,
              baselines, metrics_xgboost
training/     trainer (Adam/L2/early-stop/grad-accum/CSV curve), metrics (+ leakage guard),
              manifest (run provenance), utils (seeding, determinism, device pinning)
evaluation/   localize (attention -> lines, top-k/MRR/IFA), ablation, imbalanced, static
              analyzers, cve_eval, report formatters
scripts/      prepare_data, train, run_seeds, measure_readout, compare_readouts,
              run_localization, check_determinism, run_ablation, run_leakage_check,
              run_phase1.sh, reproduce
tests/        132 tests: graph builder, sparse<->dense equivalence + run-to-run identity, MIL
              masking/batch-invariance, readout attenuation, line spans, PrimeVul diffing,
              threshold leakage, grammar canary, Python-version compatibility
configs/      overlays that `extends:` the root config (smoke, a100_codexglue, a100_random,
              a100_paper_faithful)
artifacts/    RESULTS.md and per-run outputs (weights/JSON gitignored, markdown tracked)
```

### Key scripts

| command | purpose |
|---|---|
| `scripts.prepare_data` | parse → graph → word2vec → featurise → split |
| `scripts.train` | train one model (`--model devign` / `ggrn` / `mil`) |
| `scripts.run_seeds` | N seeds, mean ± std, majority baseline, manifest check |
| `scripts.measure_readout` | readout diagnostics at initialisation (§0) |
| `scripts.compare_readouts` | detection table + paired Wilcoxon + Cohen's d |
| `scripts.run_localization` | line-level localisation vs PrimeVul |
| `scripts.check_determinism` | **gate**: same seed ⇒ byte-identical run |
| `scripts.run_leakage_check` | commit-disjoint split, to size random-split leakage |

---

## 8. Tests and gates

```bash
python -m pytest tests/ -q                         # 132 tests
python -m scripts.check_determinism --epochs 3     # same seed -> byte-identical run
```

Both are gates, not suggestions. `check_determinism` compares `metrics.json` and `model.pt`
byte-for-byte; `allclose` would pass on a non-deterministic scatter whose 6e-7 per-step drift
compounds into a different model. Run it on whichever machine will produce the numbers —
determinism is a property of the machine, not of the repo.

Several tests exist because a specific bug shipped:

| test | the bug it prevents |
|---|---|
| `test_sparse_equivalence` | atomic scatter differing bitwise run to run |
| `test_mil_pool` | a padded node changing a graph's logit or its attention |
| `test_threshold_leakage` | reporting a tuned threshold on the split it was fitted on |
| `test_grammar_canary` | a tree-sitter bump silently changing the node-type vocabulary |
| `test_localize` | counting newlines in `str` rather than UTF-8 bytes (non-ASCII drift) |
| `test_primevul` | counting pure insertions, or reindentation, as vulnerable lines |
| `test_python_compat` | 3.12-only syntax reaching the 3.10 training server |

---

## 9. Status

| phase | state |
|---|---|
| **0 — runnable + deterministic** | done. 132 tests green on both machines; determinism gate PASSES on the A100 with real data |
| **1 — reproduce Devign, report honestly** | Devign and Ggrn done (3 seeds each); `paper_faithful` arm running; baselines and leakage check queued |
| **2 — implement Devign-MIL** | done. Operator, line spans, PrimeVul ground truth, localisation eval, H3 verified |
| **3 — evaluate MIL vs the reproduction** | queued: 5 seeds per arm, paired Wilcoxon, localisation tables |

Hypotheses were **pre-registered before Phase 3 ran**:

- **H1** — attention MIL *matches or beats* the FIXED Conv module on detection. Matching is a
  success: the claim is that Eq. 9's complexity buys nothing once its optimisation defect is
  corrected.
- **H2** — attention localises vulnerabilities to source lines far better than chance, and better
  than anything obtainable from the Conv module.
- **H3** — MIL needs no `logit_affine` rescue, being linear in the node embeddings. **Confirmed**
  (§0): trunk gradient 1.0× the linear control, against Eq. 9's 384× attenuation.

The expected outcome is a **tie on detection and a win on localisation**. A tie supports the
thesis rather than undermining it. The comparison that counts is MIL against `conv` with
`logit_affine: TRUE` — beating the broken version is not a finding, and `scripts.compare_readouts`
computes every delta against the fixed baseline for that reason.

---

## 10. Running on the A100 server (SSH)

Development happens on a Windows laptop; **all reported numbers come from the A100 box**. The two
machines must agree, so setup is pinned rather than "whatever pip resolves".

### One-time setup

```bash
git clone <repo> && cd DEVIGEN
python -m venv .venv && source .venv/bin/activate

# 1. torch from the CUDA index matching the server's driver -- NOT from PyPI.
#    Check with `nvidia-smi`; cu121 works on driver >= 530.
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2. everything else at the exact versions the laptop measured on.
pip install -r requirements.lock.txt
pip install -e .
```

`requirements.lock.txt` pins the dependency **closure**, not a `pip freeze` of an unrelated
environment. If the server needs a different CUDA build of torch, that is the one line to change --
record it, because it changes the manifest.

### Gate before any training

```bash
pytest tests/ -q                                   # must be fully green
python -m scripts.check_determinism --epochs 3     # must print PASS
```

`check_determinism` trains the same short run twice and requires **byte-identical** `metrics.json`
and `model.pt`. Run it on the A100 itself: determinism is a property of the machine, not of the
repo. It passed here on an RTX 4060 (CUDA 12.1), which predicts but does not prove the A100 result.

### Reproducibility settings that matter on this hardware

| Setting | Value | Why |
|---|---|---|
| `project.deterministic` | `true` | Sorted segment-reduce message passing instead of atomic scatter. Measured **~6% faster** as well as reproducible, so there is no tradeoff to weigh. |
| `project.allow_tf32` | `false` | TF32 is ON by default for cuDNN convolutions on Ampere. The Conv module is Conv1d/Conv2d, so the default gives the A100 different arithmetic from the laptop for identical code and seed. |
| `embedding.word2vec_deterministic` | `true` | gensim is only reproducible single-threaded; its vectors *are* the node features. |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Set automatically in `training/utils.py` before torch is imported. |

Turning TF32 on is defensible for throughput -- but then re-run the **whole** study with it on.
Never mix TF32 and non-TF32 runs in one table. The manifest in every `meta.json` records which was
used, and `training.manifest.compare_manifests` reports whether two runs were comparable at all.

### Typical invocation

```bash
python -m scripts.prepare_data --config config_a100.yaml
python -m scripts.train --config config_a100.yaml --model devign
python -m scripts.reproduce --config config_a100.yaml          # full study
```

Long runs: use `tmux` or `nohup` so an SSH drop does not kill training. Every epoch checkpoints to
`<artifacts>/<model>/<project>/checkpoint.pt`, so a dropped session resumes at the last completed
epoch rather than restarting from scratch.
