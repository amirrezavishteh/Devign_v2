# Method and results

What Devign does, what we changed, and what the numbers came out as.

This is the technical narrative. For setup, commands and repository layout see
[`README.md`](README.md); for the full tables see [`artifacts/RESULTS.md`](artifacts/RESULTS.md)
and [`artifacts/IDEA_EVAL.md`](artifacts/IDEA_EVAL.md).

---

## 1. The task

Given the source of one C function, predict whether it contains a vulnerability. Binary
classification over ~20,000 functions from FFmpeg and QEMU, labelled by whether a
vulnerability-fixing commit touched them.

The dataset is 45.6% positive, so **the number to beat is not 50%** — a classifier that always
answers "not vulnerable" scores **56.05% accuracy** on our test split. Every table in this project
prints that row.

---

## 2. Devign's pipeline

```
C source
   │  tree-sitter parse
   ▼
AST  ──────────────────────────────────► composite graph g(V, X, A)
   │  + 6 derived edge families               V = every AST node
   │                                          X = node features
   ▼                                          A ∈ {0,1}^(k×m×m), k = 7
node features x_v = [ Code ‖ Type ]
   │  Code : word2vec over source tokens (100-d)
   │  Type : learned embedding of the tree-sitter node type (100-d)
   ▼
Gated Graph Recurrent layer, T = 6 steps        (Eq. 3–4)
   │  h⁽¹⁾ = [x, 0];  a = Σ_p A_pᵀ(W_p H);  h ← GRU(h, a)
   ▼
H ∈ ℝ^(m×200)  — one 200-d vector per node
   │
   ▼  READOUT  ← this is where the work is
P(vulnerable)
```

### The 7 edge types

| edge | meaning |
|---|---|
| `AST` | parent → child |
| `REV_AST` | child → parent |
| `CFG` | approximate control flow (if/while/for/switch/break/goto/return) |
| `NCS` | natural code sequence — AST leaves left-to-right |
| `DFG_R` / `DFG_W` / `DFG_C` | LastRead / LastWrite / ComputedFrom over identifiers |

The paper fixes k = 7 without naming the seventh. `REV_AST` is the standard choice for a tree
backbone: without it messages only flow downward and a leaf's token never reaches the root.

### Corpus accounting

| | count |
|---|---|
| released functions | 27,318 |
| after exact-duplicate removal | 27,258 (45.58% positive) |
| graphs successfully built | **20,657 (75.8%)** |
| dropped: > 500 nodes (the paper's filter) | 5,212 (19.1%) |
| dropped: unparseable | 1,389 (5.1%) |
| final split | 16,536 / 2,071 / 2,050 |

**Only 2 of the paper's 4 projects were ever released.** Linux Kernel and Wireshark are
permanently unreproducible and are never printed here.

---

## 3. The readout — where Devign's novelty lives

Everything above the readout is standard GGNN. The paper's contribution is the **Conv module**
(Eq. 6–9), which turns `H` into one probability:

```
Z⁽ˡ⁾ = σ([H, x])        Y⁽ˡ⁾ = σ(H)          σ(·) = MAXPOOL(ReLU(CONV(·)))
ŷ = Sigmoid( AVG( MLP(Z) ⊙ MLP(Y) ) )                            (Eq. 9)
```

Its stated job (Sec 2.4) is to *"select sets of nodes and features that are relevant to the
current graph-level task"*.

### What we measured about it

`python -m scripts.measure_readout --config configs/a100_codexglue.yaml`

One real 128-graph batch, identical trunk and seed across arms, at initialisation:

| readout | prob spread | loss | trunk ‖grad‖ | vs linear control |
|---|---|---|---|---|
| **Eq. 9 as written** | 4.51e-04 | 0.6931 | 1.57e-04 | **384× weaker** |
| Eq. 9 + learnable affine | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| **Eq. 5, flat summation** | 0.00e+00 | **24.62** | 3.29e+01 | 546× stronger |
| **MIL attention** | 2.11e-01 | 0.7153 | 1.58e-01 | **1.0× — healthy** |
| plain linear head (control) | 1.86e-01 | 0.7008 | 6.03e-02 | (reference) |

`ln(2) = 0.6931` is the loss of a model at chance.

**Eq. 9 collapses.** It ends in a product of two MLP heads. Both start near zero, and since
∂(zy)/∂z = y and ∂(zy)/∂y = z, each branch's gradient is scaled by the *other* branch's
smallness — the trunk is starved through both paths at once. Every probability in the batch lands
inside [0.4996, 0.5000], and the first ~8 epochs score F1 = 0.00.

**Eq. 5 saturates** — the opposite failure. It sums an *unnormalised* per-node logit over up to
500 nodes, so logits reach +41, every probability pins at 1.0, and loss starts at 24.6 on a
43.2%-positive split. Its large gradient is a saturated sigmoid, not health.

So the paper's own Q2 — *"does the Conv module beat flat summation?"* — compares **two
badly-conditioned readouts**, not a good one against a bad one.

---

## 4. What we propose: Devign-MIL

The Conv module's stated job *is* attention-based multiple-instance pooling. Eq. 9 is a degenerate
implementation of it. So we replace the readout and keep everything else:

```
e_j   = wᵀ ( tanh(V h_j) ⊙ sigmoid(U h_j) )      gated attention, per node
a     = softmax(e  over REAL nodes only)          padded nodes → −∞ first
z     = Σ_j a_j h_j                               bag embedding
logit = Linear(z)
```

Following Ilse, Tomczak & Welling, *Attention-based Deep Multiple Instance Learning* (ICML 2018).
Input `h_j = [H_j ‖ x_j]` — the same tensor Eq. 9's Z branch consumes, so the comparison isolates
the readout.

**Three properties follow:**

1. **Live gradient at initialisation.** It is linear in the node embeddings, so no rescue term is
   needed — no affine, no scale, no bias. Measured at 1.0× the linear control (table above).
2. **Parameter parity.** 630,785 against Devign's 631,410 — 0.1% apart, identical trunk. Neither
   arm can win on capacity.
3. **Localisation for free.** `a` is a distribution over nodes; every node carries a source span;
   so a per-line score is a *projection* of `a`, not a second model.

Masking is mandatory and asserted in tests: padded positions go to −∞ before the softmax, so a
graph's prediction cannot depend on which graphs it was batched with.

### The contribution, stated narrowly

Attention MIL is not new (Ilse et al. 2018). Attention for vulnerability localisation is not new
(LineVul and others). The claim is:

> *the paper's central novelty is a degenerate approximation of a known-better operator, here is
> the measurement that shows it, here is the principled replacement, and here is the
> interpretability the paper listed as future work.*

---

## 5. Results

### 5.1 Reproduction vs the paper

| | accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **This reproduction** (5 seeds) | 63.50 ± 0.83 | 56.27 ± 4.57 | 68.58 ± 0.88 |
| Majority-class baseline | 56.05 | 0.00 | 50.00 |
| Paper (QEMU column) | 74.33 | 73.07 | — |

Public reproductions of this dataset cluster at 60–66%, so this is inside the normal band and the
paper's figure is the outlier. The cells are **not like-for-like** — the paper's QEMU column is a
different split of a different project subset.

### 5.2 Table 2 — the sequence baselines win

Combined (QEMU + FFmpeg), baselines one run each:

| model | accuracy | F1 |
|---|---|---|
| **CNN** | **67.70** | 56.25 |
| 3-layer BiLSTM | 66.06 | 55.87 |
| **BiLSTM + Attention** | 64.08 | **63.67** |
| Devign (this repro, 5 seeds) | 63.50 ± 0.83 | 56.27 ± 4.57 |
| Metrics + XGBoost | 60.36 | 18.15 |
| majority class | 56.05 | 0.00 |

**CNN beats Devign by 4.20 accuracy; BiLSTM + Attention by 7.40 F1.** In the paper Devign beats
all four. A CNN over token sequences has no AST, no CFG, no data-flow and no message passing — and
composite program structure is Devign's entire premise.

*Caveat:* baselines are single-run. The accuracy gaps exceed Devign's ±0.83 spread; the F1 gap is
comparable to its ±4.57 and is the weaker claim.

### 5.3 The four readouts, one trunk, 5 seeds each

| readout | accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| sum (Eq. 5) | 61.89 ± 0.32 | 57.74 ± 1.66 | 67.03 ± 0.67 | 60.79 ± 1.50 |
| conv, affine FALSE (Eq. 9 as written) | 62.27 ± 1.08 | 56.47 ± 2.07 | 68.16 ± 0.60 | 62.65 ± 0.55 |
| **conv, affine TRUE** (honest baseline) | **63.50 ± 0.83** | 56.27 ± 4.57 | **68.58 ± 0.88** | **63.13 ± 1.09** |
| **mil (k=1)** | 61.61 ± 0.70 | **58.47 ± 0.72** | 68.08 ± 0.59 | 62.39 ± 1.47 |

**MIL vs the fixed Conv module**, paired per seed:

| metric | delta | MIL wins | Cohen's d | reading |
|---|---|---|---|---|
| ROC-AUC | −0.50 | 1/5 | −1.23 | tie |
| PR-AUC | −0.74 | 2/5 | −0.44 | tie |
| F1 | +2.20 | 3/5 | +0.45 | tie (sign flips) |
| **accuracy** | **−1.89** | **0/5** | **−1.91** | **Conv wins** |

**H1 is not supported as stated.** It predicted "matches or beats". MIL matches on threshold-free
ranking and **loses accuracy on every one of five seeds**. The two readouts order the test set
about equally well; they differ in where the operating point lands.

**MIL is 6.3× more stable**, which was not hypothesised:

```
conv:  52.22  59.79  62.25  52.07  55.01     range 10.18,  std 4.57
mil:   58.21  57.35  58.80  59.20  58.81     range  1.85,  std 0.72
```

A single-seed comparison of these arms could have reported anything from MIL +6.6 to MIL −4.0.

### 5.4 The affine fix moves the boundary, not the model

| metric | TRUE − FALSE | TRUE wins |
|---|---|---|
| accuracy | +1.23 | **5/5** |
| ROC-AUC | +0.42 | 3/5 |
| PR-AUC | +0.47 | 3/5 |

The Eq. 9 defect is real at initialisation. But by early-stopping convergence the model **escapes
it unaided** — ranking quality with and without the affine is a tie. The affine buys a
better-placed threshold, not a better model, and it does not explain the gap to the paper.

### 5.5 The benchmark is mostly commit memorisation

| split | eval functions sharing a commit with training |
|---|---|
| **codexglue (the released split)** | **65%** |
| random 75/12.5/12.5 | 65% |
| commit-disjoint | 0% |

Closing that channel:

| metric | commit-disjoint | CodeXGLUE | gap |
|---|---|---|---|
| accuracy | **50.73** | 64.69 | +13.96 |
| F1 | **18.26** | 55.86 | +37.60 |
| ROC-AUC | **54.64** | 70.83 | +16.19 |

That split is exactly 50.00% positive — so **accuracy 50.73 beats the majority baseline by 0.73
points**, and AUC sits 4.64 above chance. Once no fix commit spans both sides, the model is close
to guessing.

This does not make the paper wrong about its protocol; Sec 3.3 splits randomly and says so. It
means the benchmark rewards commit recognition.

---

## 6. Verdict

**On the reproduction.** Devign reaches 63.50 accuracy against a 56.05 floor — real signal, about
7 points of it, and ~11 points below the paper. That is inside the published band for this dataset.

**On the architecture.** Both of the paper's readouts are badly conditioned at initialisation, in
opposite directions, and neither defect survives to convergence. The Conv module does beat flat
summation, at a third the claimed size — and loses on the paper's own headline metric.

**On the graph premise.** Three sequence baselines with no program structure at all match or beat
the graph model on this data. That is the most uncomfortable result here and the one most worth
following up.

**On Devign-MIL.** It ties on ranking, loses ~1.9 accuracy consistently, needs no rescue term, and
is 6× more stable across seeds. As a *detector* it is not better. Its case rests on localisation,
which is **not yet measured** — PrimeVul ground truth is not in place — and until it is, the
honest summary is: a simpler readout with equal ranking quality, worse accuracy, and an untested
interpretability claim.

**The strongest argument against this work** is that the operator is borrowed, the detection
result is a tie-or-loss, and the localisation claim is unmeasured. What would settle it is a
line-level result that clears the line-length prior by more than seed variance, on ground truth
with known label quality.

---

## 7. What is not yet measured

| item | why it matters |
|---|---|
| **Localisation (H2)** | The only claim that would *distinguish* this work. Blocked on PrimeVul |
| Seeded baselines | Table 2's baselines are single-run; the F1 gap needs error bars |
| Per-project QEMU / FFmpeg graph columns | Currently training |
| Three arms on commit-disjoint | Whether the readout ranking survives leakage removal |
| T ∈ {4, 6, 8, 12} | Median AST depth is 11 against the paper's T = 6, so the upper half of a typical tree is unreachable from its leaves |

---

## 8. Reproducibility

Every number here is regenerable. Two gates must pass first:

```bash
pytest tests/ -q                                 # 132 tests
python -m scripts.check_determinism --epochs 3   # same seed ⇒ byte-identical run
```

`check_determinism` compares `metrics.json` **and** `model.pt` byte-for-byte. Getting there
required replacing atomic scatter in the GGNN — it was bitwise different on 20 of 20 repeats — with
a sorted segment reduction, which is exactly reproducible and ~6% faster.

Every run writes a manifest: git SHA, resolved-config hash, library versions, seed, device, and a
hash of the ordered function IDs in each split. Two runs are comparable only if their split hashes
match, and `training.manifest.compare_manifests` checks rather than assumes.
