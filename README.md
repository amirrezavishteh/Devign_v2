# Devign — reproduction, audit, and a replacement readout

A from-scratch PyTorch reproduction of **Devign** (Zhou et al., *"Devign: Effective Vulnerability
Identification by Learning Comprehensive Program Semantics via Graph Neural Networks"*, NeurIPS
2019 — [arXiv:1909.03496](https://arxiv.org/abs/1909.03496)), trained on the paper authors' own
released dataset.

The work has three parts:

1. **Reproduce** the paper honestly and say where it lands.
2. **Audit** the architecture — measure *why* the paper's readout trains badly, rather than
   asserting it.
3. **Replace** it with gated attention MIL pooling, which does the same job without the defect and
   yields statement-level localisation for free.

Everything below is measured in this repository and regenerable by the command beside it. Nothing
is quoted from the paper or from another reproduction. Full tables:
[`artifacts/RESULTS.md`](artifacts/RESULTS.md) and [`artifacts/IDEA_EVAL.md`](artifacts/IDEA_EVAL.md).

---

## Headline results

### 1. The reproduction lands ~11 points below the paper, inside the published band

| | accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **This repro** (Devign, CodeXGLUE test, 5 seeds) | 63.50 ± 0.83 | 56.27 ± 4.57 | 68.58 ± 0.88 | 63.32 ± 1.20 |
| Majority-class baseline | 56.05 | 0.00 | 50.00 | 43.95 |
| Paper (QEMU column) | 74.33 | 73.07 | — | — |

Public reproductions of this dataset cluster at 60–66% accuracy (CodeBERT ≈62%), so this sits in
the normal band and the paper's figure is the outlier. The cells are **not like-for-like**: the
paper's QEMU column is a different split of a different project subset. **Linux Kernel and
Wireshark were never released and are permanently unreproducible** — those columns are never
printed here.

### 2. Two thirds of the benchmark is commit leakage

`python -m scripts.measure_commit_overlap --config configs/a100_codexglue.yaml`

| split | eval functions from a **training commit** | eval body duplicated in train |
|---|---|---|
| **codexglue** (the released split) | **1,776 / 2,726 = 65%** | 6 (0%) |
| random 75/12.5/12.5 | 2,205 / 3,408 = 65% | 6 (0%) |
| commit-disjoint | 0 (0%) | 2 (0%) |

One fix commit usually touches several functions; 3,729 of the 12,293 commits touch more than one.
Any split that does not group by commit scatters siblings across both sides, and the model can
recognise the commit rather than the flaw. Near-duplicate bodies are ~0%, so the channel is
commit membership, not copy-pasted code.

**What closing it costs** — same architecture, data, code and hyperparameters:

| metric | commit-disjoint | CodeXGLUE (3 seeds) | gap |
|---|---|---|---|
| accuracy | **50.73** | 64.69 ± 0.42 | +13.96 |
| F1 | **18.26** | 55.86 ± 2.12 | +37.60 |
| ROC-AUC | **54.64** | 70.83 ± 0.21 | +16.19 |

That split is exactly 50.00% positive, so accuracy **50.73 beats the majority baseline by 0.73
points** and AUC sits **4.64 above chance**. Once no fix commit spans both sides, the model is
close to guessing.

This does not make the paper wrong about its protocol — Sec 3.3 splits randomly and says so. It
means the benchmark rewards commit recognition, and a reproduction that omits this is reporting a
number it does not understand.

*Caveats:* the commit-disjoint figure is one seed against a 3-seed mean (the gap is ~80× the seed
spread, so not noise, but it is not itself a mean); and it trains on 20,443 functions against
21,808, so a small part of the gap is less data rather than less leakage.

### 3. Both of the paper's readouts are broken at initialisation — in opposite directions

`python -m scripts.measure_readout --config configs/a100_codexglue.yaml`

Real 128-graph batch, identical trunk and seed across arms:

| readout | logit range | prob spread | loss | trunk ‖grad‖ | vs linear control |
|---|---|---|---|---|---|
| **Eq. 9** (Conv, as written) | 1.80e-03 | 4.51e-04 | 0.6931 | 1.57e-04 | **384× weaker** |
| Eq. 9 + learnable affine | 9.01e-02 | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| **Eq. 5** (flat summation) | 4.15e+01 | 0.00e+00 | **24.62** | 3.29e+01 | 546× stronger |
| **MIL attention** (new) | 8.58e-01 | 2.11e-01 | 0.7153 | 1.58e-01 | **1.0× — healthy** |
| linear control | 7.52e-01 | 1.86e-01 | 0.7008 | 6.03e-02 | (reference) |

`ln(2) = 0.6931` is the loss of a model at chance.

**Eq. 9 collapses.** It ends in `MLP(Z) ⊙ MLP(Y)`; both heads start near zero, and since
∂(zy)/∂z = y and ∂(zy)/∂y = z, each branch is throttled by the other's smallness. Every
probability in the batch lands inside [0.4996, 0.5000]. A prior audit measured 53× attenuation; on
real graph sizes it is **384×**.

**Eq. 5 saturates.** It sums an *unnormalised* per-node logit over up to 500 nodes, so logits reach
+41, every probability pins at 1.0, and loss starts at 24.6 on a 43.2%-positive split. Its large
gradient is a saturated sigmoid, not health.

**Consequence for the paper's Q2** — *"does the Conv module beat flat summation?"* compares two
badly-conditioned readouts, not a good one against a bad one.

### 4. The Conv module's claimed advantage does not reproduce

The paper's ablation claims the Conv module adds **+4.66 accuracy and +6.37 F1** over Ggrn.
Measured pairwise per seed:

| metric | seed 1 | seed 2 | seed 3 | mean | Devign wins |
|---|---|---|---|---|---|
| accuracy | +1.17 | +2.05 | −0.29 | +0.98 | 2/3 |
| F1 | −2.59 | +5.73 | −1.93 | **+0.40** | **1/3** |

The sign flips across seeds on every metric.

### 5. The paper's own configuration scores higher F1 while being a worse classifier

| metric | `paper_faithful` | `repo_default` | delta |
|---|---|---|---|
| accuracy | 57.90 ± 0.74 | **62.76 ± 1.43** | −4.86 |
| **F1** | **58.36 ± 0.23** | 57.56 ± 2.57 | **+0.80** |
| recall | **67.11 ± 0.55** | 57.60 ± 5.01 | +9.51 |
| precision | 51.63 ± 0.69 | **57.69 ± 1.78** | −6.06 |
| ROC-AUC | 63.30 ± 0.79 | **68.67 ± 1.54** | −5.36 |

It buys 9.51 points of recall for 6.06 of precision — over-predicting the positive class, which is
exactly the trade F1 rewards. Every threshold-free measure says the ranking is worse. This is the
case against selecting on F1@0.5, measured rather than asserted.

### 6. Devign-MIL: ties on ranking, loses accuracy, far more stable

5 seeds per arm, one trunk, verified identical config and split hashes:

| readout | accuracy | F1 | ROC-AUC | params |
|---|---|---|---|---|
| conv, `logit_affine` TRUE | **63.50 ± 0.83** | 56.27 ± **4.57** | 68.58 ± 0.88 | 631,410 |
| **mil (k=1)** | 61.61 ± 0.70 | **58.47 ± 0.72** | 68.08 ± 0.59 | 630,785 |

| metric | delta (MIL − Conv) | MIL wins | Cohen's d | reading |
|---|---|---|---|---|
| ROC-AUC | −0.50 | 1/5 | −1.23 | tie |
| PR-AUC | −0.74 | 2/5 | −0.44 | tie |
| F1 | +2.20 | 3/5 | +0.45 | tie (sign flips) |
| **accuracy** | **−1.89** | **0/5** | **−1.91** | **Conv wins, consistently** |

**H1 is not supported as stated.** It predicted "matches or beats". On threshold-free ranking it
matches; **on accuracy it loses 1.89 points on every one of five seeds**. The two readouts order
the test set about equally well and differ in where their operating point lands.

**MIL is 6.3× more stable**, which was not hypothesised and is reported as an observation:

```
conv:  52.22  59.79  62.25  52.07  55.01     range 10.18,  std 4.57
mil:   58.21  57.35  58.80  59.20  58.81     range  1.85,  std 0.72
```

A single-seed comparison of these two arms could have reported anything from MIL +6.6 to MIL −4.0
on F1 by seed choice alone.

### 7. Determinism was not free — and costs nothing

`index_add_` message passing was **bitwise different on 20 of 20 repeats** (6e-7 drift per
forward). Replaced with a sorted segment reduction: exactly reproducible **and ~6% faster**
(9.62 ± 0.04 ms vs 10.25 ± 0.06 ms). Two same-seed runs now produce byte-identical `metrics.json`
**and** `model.pt` on the A100.

### 8. A tuned threshold was being reported on its own split

`train_model` fitted the decision threshold on validation, then scored validation at it. On one run
that gap was **F1 0.00 (unbiased) vs 68.97 (biased)** on the same split. Metrics now carry the
split they were computed on and the split their threshold came from; `require_unbiased` refuses
the pair wherever a number enters a table.

---

## Pre-registered hypotheses

Fixed before Phase 3 ran ([`artifacts/IDEA_EVAL.md`](artifacts/IDEA_EVAL.md) history).

| | hypothesis | status |
|---|---|---|
| **H1** | MIL **matches or beats** the FIXED Conv module on detection | **partially confirmed** — ties on AUC/PR-AUC/F1, **loses accuracy 5/5 seeds** |
| **H2** | Attention localises to source lines far better than chance and than the Conv module | *blocked on PrimeVul* |
| **H3** | MIL needs no `logit_affine` rescue, being linear in the node embeddings | **confirmed** — trunk gradient 1.0× a linear control |

The expected outcome was a tie on detection and a win on localisation. The detection tie holds on
ranking but not on accuracy, and that is reported as a failure of H1 rather than reframed.

---

## Architecture (paper → code)

| Paper component | § | Implementation |
|---|---|---|
| Composite multi-edge graph | 2.2 | [devign_data/parser.py](devign_data/parser.py), [devign_data/graph_builder.py](devign_data/graph_builder.py) |
| `x_v` = Code (word2vec 100-d) ⊕ Type | 2.2.2 | [devign_data/word2vec_embed.py](devign_data/word2vec_embed.py), [models/node_init.py](models/node_init.py) |
| Gated Graph Recurrent layer (Eq. 3–4) | 2.3 | [models/ggnn.py](models/ggnn.py) |
| Conv module (Eq. 6–9) | 2.4 | [models/conv_module.py](models/conv_module.py) |
| Ggrn flat summation (Eq. 5) | 2.4 | [models/devign.py](models/devign.py) |
| **Attention MIL pooling** (new) | — | [models/mil_pool.py](models/mil_pool.py) |

### The 7 edge types

`AST, REV_AST, CFG, NCS, DFG_R, DFG_W, DFG_C`. The paper fixes k=7 without naming the seventh;
`REV_AST` (reverse parent←child) is the standard choice for tree backbones, letting messages flow
up as well as down.

### The three readouts

All share an identical NodeInit + GGNN trunk, so a comparison isolates the readout. Parameter
counts are within 0.1%.

```
e_j   = w^T ( tanh(V h_j) ⊙ sigmoid(U h_j) )     # gated attention, per node
a     = softmax(e over REAL nodes only)           # padded nodes → -inf first
z     = Σ_j a_j h_j                               # bag embedding
logit = Linear(z)
```

The Conv module's stated job (Sec 2.4) — *"select sets of nodes and features that are relevant to
the current graph-level task"* — is a description of attention-based multiple-instance pooling.
Eq. 9 is a degenerate implementation of it.

**The contribution claimed is narrow.** Attention MIL is Ilse, Tomczak & Welling (ICML 2018);
attention for vulnerability localisation is LineVul and others. The claim is: *the paper's central
novelty is a degenerate approximation of a known-better operator, here is the measurement, here is
the principled replacement, and here is the interpretability the paper listed as future work.*

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.lock.txt
pip install -e .
```

`requirements.lock.txt` pins the dependency **closure**, not a `pip freeze` of an unrelated
environment. Verified on two environments: Python 3.12 / RTX 4060 / CUDA 12.1, and Python 3.10 /
2× A100-SXM4-80GB / CUDA 12.4.

The tree-sitter **grammar** is pinned exactly (node types are half of every node feature) while the
runtime is not — `tests/test_grammar_canary.py` hashes the grammar's output over a fixed canary
function, which is a stricter guard than a version string and passed identically on 0.25.2 and
0.26.0.

### Data

```bash
mkdir -p data/codexglue && cd data/codexglue
for f in train validation test; do
  curl -L -O "https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection/resolve/main/data/${f}-00000-of-00001.parquet"
done
```

27,318 functions → 27,258 after dedupe (45.58% positive) → **20,657 graphs built** (75.8%; 19.1%
dropped by the paper's >500-node filter, 5.1% unparseable) → 16,536 / 2,071 / 2,050.

For localisation ground truth, run `python -m scripts.fetch_primevul` — CodeXGLUE ships `func` and
`target` only, with no `func_after` and no line annotation, so it **cannot** support a line-level
evaluation at all.

---

## Running

```bash
# gates — both must pass before any number is trusted
pytest tests/ -q                                    # 132 tests
python -m scripts.check_determinism --epochs 3      # same seed ⇒ byte-identical run

# reproduction
python -m scripts.prepare_data --config configs/a100_codexglue.yaml
python -m scripts.run_seeds --config configs/a100_codexglue.yaml --model devign --seeds 1 2 3 4 5

# audit
python -m scripts.measure_readout        --config configs/a100_codexglue.yaml
python -m scripts.measure_commit_overlap --config configs/a100_codexglue.yaml
python -m scripts.run_leakage_check      --config configs/a100_codexglue.yaml --models devign

# the three-arm comparison
bash scripts/run_phase3.sh
python -m scripts.compare_readouts --dir artifacts/seeds --baseline devign

# localisation (needs PrimeVul)
bash scripts/run_phase3_localization.sh
```

| script | purpose |
|---|---|
| `prepare_data` | parse → graph → word2vec → featurise → split |
| `train` | one model (`--model devign` / `ggrn` / `mil`) |
| `run_seeds` | N seeds, mean ± std, majority baseline, manifest check, `--set k=v` overrides |
| `measure_readout` | readout diagnostics at initialisation |
| `measure_commit_overlap` | the leakage channel, directly |
| `compare_readouts` | detection table + paired Wilcoxon + Cohen's d |
| `run_localization` | line-level localisation vs PrimeVul |
| `render_attention` | HTML heat overlay beside the fix diff |
| `check_determinism` | **gate**: same seed ⇒ byte-identical run |

Long runs are queued by `scripts/run_phase1.sh`, `run_phase3.sh` and `run_t_sweep.sh` — sequential
(the GPU is shared), resumable (each stage marks itself done), and isolated (one stage failing
does not stop the others).

---

## Honesty machinery

These are enforced in code, not by discipline:

| guard | what it prevents |
|---|---|
| `require_unbiased` | reporting a tuned threshold on the split it was fitted on |
| `compare_manifests` | pooling runs that trained on different data or config |
| arm-collision guard | two files silently mapping to one row of a results table |
| `--set` unknown-key refusal | a typo training the default and reporting it as the swept value |
| majority-class row | every model number printed beside the class balance |
| `p_floor` beside every p-value | "p = 0.25 on 3 seeds" read as weak evidence when it is the floor |
| PrimeVul unpaired-split detection | a file that parses cleanly and yields zero labels |
| format-version check | evaluating on graphs that predate a feature the evaluation needs |

Every run writes a manifest: git SHA, resolved-config hash, library versions, seed, device, and a
hash of the ordered function IDs in each split.

### Statistical honesty

With 5 seeds the smallest attainable two-sided Wilcoxon p is **0.0625** — nothing can reach
p < 0.05 however consistent it looks. With 3 seeds the floor is 0.25. That floor is printed beside
every p-value. Observed seed spread is ±0.72 to ±4.57 F1, so **differences under ~2 points are
reported as ties, not as a rank order**.

---

## Tests

```bash
python -m pytest tests/ -q      # 132 tests
```

Several exist because a specific bug shipped:

| test | the bug it prevents |
|---|---|
| `test_sparse_equivalence` | atomic scatter differing bitwise run to run |
| `test_mil_pool` | a padded node changing a graph's logit or attention |
| `test_threshold_leakage` | reporting a tuned threshold on its own split |
| `test_grammar_canary` | a tree-sitter bump silently changing the node-type vocabulary |
| `test_localize` | counting newlines in `str` rather than UTF-8 bytes |
| `test_primevul` | counting pure insertions or reindentation as vulnerable lines |
| `test_python_compat` | 3.12-only syntax reaching the 3.10 training server |

---

## Project layout

```
devign_data/  parser, graph builder, word2vec, dataset/batching, line spans,
              hf_devign (dataset download), primevul (localisation ground truth), prepare
models/       node_init, ggnn, conv_module (Eq. 6-9), mil_pool (attention MIL),
              devign+ggrn+mil, baselines, metrics_xgboost
training/     trainer (Adam/L2/early-stop/grad-accum/CSV curve), metrics (+ leakage guard),
              manifest (run provenance), utils (seeding, determinism, device pinning)
evaluation/   localize (attention -> lines, top-k/MRR/IFA), ablation, imbalanced,
              static analyzers, cve_eval, report formatters
scripts/      see the table above, plus the queue scripts
configs/      overlays that `extends:` the root config
artifacts/    RESULTS.md, IDEA_EVAL.md (weights/JSON gitignored, markdown tracked)
```

---

## Status

| phase | state |
|---|---|
| **0 — runnable + deterministic** | done. 132 tests green on both machines; determinism gate PASSES on the A100 with real data |
| **1 — reproduce and report** | done. All five deliverables |
| **2 — implement Devign-MIL** | done. Operator, line spans, PrimeVul loader, localisation eval, H3 verified |
| **3 — evaluate** | detection: MIL + Conv complete (5 seeds), Ggrn + `affine FALSE` running; localisation blocked on PrimeVul |

### Open items

- **Eq. 5 saturates at initialisation**, so the three-arm comparison currently measures MIL
  against *two* degenerate baselines. A node-count-normalised variant would be the fair
  comparison; it deviates from the equation as written, so it is recorded rather than silently
  applied. **Open decision.**
- **PrimeVul not yet placed on the server** — blocks H2.
- The T ∈ {4, 6, 8, 12} over-squashing sweep is implemented but not yet run. Median AST depth here
  is 11 against the paper's T = 6, so the upper half of a typical tree is unreachable from its
  leaves.

---

## On attention as explanation

Jain & Wallace (2019) and Wiegreffe & Pinter (2019) established that attention weights are not
automatically explanations. This work does not claim otherwise. It validates attention against
**ground-truth vulnerable lines derived from the actual fix commit**, which is a different and
falsifiable claim — and it reports a line-length prior alongside, because if a trivial length
heuristic matches the attention map then the attention has learned nothing.

---

## Running on the A100 server

```bash
python -m scripts.prepare_data --config configs/a100_codexglue.yaml
bash scripts/run_phase3.sh
```

`config_a100.yaml` pins `project.cuda_device: 1` (card 0 is routinely held at ~80 of 82 GB by other
users) and `project.allow_tf32: false`. TF32 is on by default for cuDNN convolutions on Ampere, and
the Conv module is Conv1d/Conv2d — leaving it at the default would give the A100 and a development
laptop different arithmetic for identical code and seed. Turning it on is defensible for
throughput, but then re-run the **whole** study: never mix TF32 and non-TF32 runs in one table.

Use `tmux` or `setsid nohup` — every epoch checkpoints, so a dropped SSH session costs minutes
rather than hours.
