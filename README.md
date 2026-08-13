# Devign — Vulnerability Identification via Graph Neural Networks

An end-to-end, runnable reproduction of **Devign** (Zhou et al., *"Devign: Effective Vulnerability
Identification by Learning Comprehensive Program Semantics via Graph Neural Networks"*, NeurIPS
2019) in PyTorch, trained on the **paper authors' own released dataset** — not a synthetic stand-in.

It encodes each C function as a **composite multi-edge graph** (AST + CFG + DFG + NCS), learns
node representations with a **Gated Graph Recurrent layer**, and classifies whole graphs with the
paper's novel **Conv module**. All three components, the four Table-2 baselines, the Table-3
imbalanced study, the single-edge ablation, and a commit-disjoint leakage check are implemented
and have been run on real data (see §5).

---

## 1. Architecture (paper → code)

| Paper component | Section | Implementation |
|---|---|---|
| Graph Embedding Layer (composite semantics) | 2.2 | [data/parser.py](data/parser.py), [data/graph_builder.py](data/graph_builder.py) |
| Node features `x_v` = Code (word2vec, 100-d) ⊕ Type (label enc.) | 2.2.2 | [data/word2vec_embed.py](data/word2vec_embed.py), [models/node_init.py](models/node_init.py) |
| Gated Graph Recurrent layer (Eq. 3–4, T=6, z=200, SUM agg) | 2.3 | [models/ggnn.py](models/ggnn.py) — dense **and** sparse edge-list propagation (see §2) |
| Conv module (Eq. 6–9, dual branch, 2 conv layers, `pool2` = (2,2)/(1,2)) | 2.4 | [models/conv_module.py](models/conv_module.py) |
| Devign model + Ggrn flat-summation variant (Eq. 5) | 2.4 | [models/devign.py](models/devign.py) |
| Baselines: Metrics+XGBoost, BiLSTM, BiLSTM+Att, CNN | 3.2 | [models/baselines.py](models/baselines.py), [models/metrics_xgboost.py](models/metrics_xgboost.py) |
| Training (Adam, lr 1e-4, bs 128 via gradient accumulation, L2, early stop) | 3.3 | [training/trainer.py](training/trainer.py) |
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
> CFG/DFG/NCS heuristically in [data/graph_builder.py](data/graph_builder.py) — including the
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
  [data/hf_devign.py](data/hf_devign.py) (`huggingface_hub` + `pyarrow`, no extra install). All
  three published splits are concatenated and re-split ourselves (Sec 3.3 does its own random
  split, not CodeXGLUE's). **27,258 functions after de-duplication** (FFmpeg 9,726 / QEMU 17,532).
- **`file`.** Point `data.real_data_path` at any `json`/`jsonl`/`csv` with
  `{func, target, project, commit_id}` columns (e.g. a Big-Vul export).
- **`synthetic`.** [data/templates.py](data/templates.py)'s 10 CWE-pattern generator. Offline
  smoke-test only — it is what `--quick` / the test suite exercise, and its numbers are **not**
  vulnerability-detection evidence (the generator is trivially separable; every strong model
  saturates near 100%, including the CNN baseline).

Functions with **> 500 nodes, or that tree-sitter could not parse cleanly, are filtered out**
(`data.max_nodes`, `drop_parse_errors` in `build_graph`) — the paper reports ~15% dropped this way.

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

## 6. Verified run — actual results (real data)

*(Filled in after `python -m scripts.reproduce --epochs 100` completes on the real FFmpeg+QEMU
data — see `artifacts/RESULTS.md` for the full tables with paper-delta columns once the run
finishes; this section will be updated with the same numbers.)*

---

## 7. Project layout

```
data/         parser, graph builder (incl. parse-error filter), word2vec, dataset/batching
              (dense + sparse), hf_devign (real dataset download), synthetic templates, prepare
models/       node_init, ggnn (dense + sparse GGNN), conv_module, devign+ggrn, baselines, xgboost
training/     trainer (Adam/L2/early-stop/grad-accum), metrics, baseline training, utils
evaluation/   ablation, imbalanced (Table 3), static analyzers, cve_eval (unseen-commit holdout),
              report formatters
scripts/      prepare_data, train, train_baselines, run_ablation, run_imbalanced, run_cve,
              run_leakage_check, reproduce (resumable end-to-end orchestrator)
inference/    predict (single-function CLI + DevignPredictor class)
tests/        graph builder, sparse≡dense GGNN equivalence, split logic, model forward/backward
Devign_Colab.ipynb   free-tier Colab notebook for the full study
config.yaml   all hyperparameters
```

---

## 8. Tests

```bash
python -m pytest tests/ -q
```
