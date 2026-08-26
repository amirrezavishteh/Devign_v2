# Critique of Devign, with a proposed solution

A review of the disadvantages, limitations and shortcomings of

> Zhou, Liu, Siow, Du, Liu. **"Devign: Effective Vulnerability Identification by Learning
> Comprehensive Program Semantics via Graph Neural Networks."** NeurIPS 2019.
> [arXiv:1909.03496](https://arxiv.org/abs/1909.03496)

followed by a proposed scientific solution in §4.

Wherever a criticism can be backed by a measurement, the measurement is from **this
repository's own reproduction** and the command that produces it is given. Where a criticism is a
judgement rather than a measurement, it is labelled as such. Where the paper is being judged by
standards that post-date it, that is said plainly rather than used as a rhetorical advantage.

---

## 1. What the paper claims

| claim | where |
|---|---|
| A composite graph (AST + CFG + DFG + NCS) captures program semantics that token-sequence models miss | Sec 2.2 |
| A gated graph recurrent layer learns useful node representations over that graph | Sec 2.3 |
| A novel **Conv module** (Eq. 6–9) selects task-relevant nodes and features for graph-level prediction | Sec 2.4 |
| Devign beats four baselines by **10.51% accuracy / 8.68% F1** on average | Table 2 |
| The Conv module alone adds **4.66% accuracy / 6.37% F1** over flat summation | Sec 3.4, Q2 |
| Devign generalises to unseen CVEs | Sec 3.4, Q5 |

---

## 2. Shortcomings

### 2.1 Half the dataset was never released — the headline numbers are unverifiable

The paper reports four projects: Linux Kernel, QEMU, Wireshark, FFmpeg. **Only QEMU and FFmpeg
were ever published** (as CodeXGLUE `google/code_x_glue_cc_defect_detection`). Linux Kernel and
Wireshark — the two largest, together 36,604 of the paper's 58,965 functions — were not.

Consequences:

- The **Combined** column, the paper's headline, cannot be reproduced by anyone.
- The two largest projects, which dominate that average, cannot be checked at all.
- No independent party can confirm the 10.51% / 8.68% margin.

This is not a minor artifact gap. It means **the central quantitative claim of the paper is, as of
today, permanently unfalsifiable**. Our reproduction prints those columns as
`never released — not reproducible` rather than inventing them.

*Severity: high. This is the single largest obstacle to assessing the work.*

### 2.2 The evaluation protocol leaks — two thirds of the test set shares a commit with training

The paper splits randomly (Sec 3.3). A vulnerability-fixing commit typically touches **several
functions**, so a random assignment scatters siblings from the same commit across train and test.

Measured on the released data (`python -m scripts.measure_commit_overlap`):

| split | eval functions from a commit also in training |
|---|---|
| **the released CodeXGLUE split** | **1,776 / 2,726 = 65%** |
| random 75/12.5/12.5 | 65% |
| commit-disjoint | 0% |

3,729 of the 12,293 commits touch more than one function. Near-duplicate function bodies account
for ~0%, so the channel is **commit membership**, not copy-pasted code.

What closing it costs, same architecture and code:

| metric | commit-disjoint | released split | gap |
|---|---|---|---|
| accuracy | **50.73** | 64.69 | 13.96 |
| F1 | **18.26** | 55.86 | 37.60 |
| ROC-AUC | **54.64** | 70.83 | 16.19 |

The commit-disjoint split is exactly 50.00% positive, so **accuracy 50.73 exceeds the
majority-class baseline by 0.73 points** and ROC-AUC sits **4.64 above chance**.

**Once no fix commit spans both sides, the model is close to guessing.** A large share of what the
benchmark measures is the ability to recognise a commit, not a flaw.

*Fairness note:* the paper does not hide this — Sec 3.3 states the split is random, and
commit-disjoint splitting was not standard practice in 2019. The criticism is that the number
means less than it appears to, not that the authors concealed anything. *Severity: high.*

### 2.3 The central architectural novelty is degenerate at initialisation

Eq. 9 is

```
ŷ = Sigmoid( AVG( MLP(Z⁽ˡ⁾) ⊙ MLP(Y⁽ˡ⁾) ) )
```

an **elementwise product of two MLP heads**. Both are randomly initialised near zero, so their
product starts near zero *squared*. Worse, since ∂(zy)/∂z = y and ∂(zy)/∂y = z, each branch's
gradient is scaled by the *other* branch's smallness — the trunk is starved through both paths
simultaneously.

Measured on a real 128-graph batch at initialisation, identical trunk and seed across arms
(`python -m scripts.measure_readout`):

| readout | prob spread | loss | gradient reaching the GGNN trunk |
|---|---|---|---|
| **Eq. 9 as written** | 4.51e-04 | 0.6931 | **384× weaker** than a linear head |
| Eq. 5 (flat summation) | 0.00e+00 | **24.62** | 546× *stronger* (saturated) |
| plain linear head | 1.86e-01 | 0.7008 | (reference) |

Every probability in the batch lands inside **[0.4996, 0.5000]**, loss is exactly `ln 2`, and the
first ~8 epochs score **F1 = 0.00**. The same pathology is present in the original authors'
released code.

**And Eq. 5, the paper's own ablation baseline, fails the opposite way.** It sums an
*unnormalised* per-node logit over up to 500 nodes, so logits reach +41, every probability pins at
1.0, and loss starts at 24.6 on a 43%-positive split.

So the paper's Q2 — *"does the Conv module beat flat summation?"* — **compares two
badly-conditioned readouts**, not a good one against a bad one.

*Qualification found by our own experiment:* the defect costs the first several epochs but does
**not** survive to convergence. Adding a learnable affine to Eq. 9 changes accuracy by +1.23 (5/5
seeds) but ROC-AUC by only +0.42 (3/5) — it relocates the decision boundary rather than producing
a better model. The defect is real, and it explains less than it first appears to.
*Severity: medium.*

### 2.4 Every reported number is a single run, with no variance and no significance test

The paper reports point estimates. No standard deviations, no seed counts, no significance tests.

Our 5-seed measurements show why that is not acceptable at these effect sizes:

```
Devign, per-seed test F1:   52.22  59.79  62.25  52.07  55.01     std 4.57, range 10.18
Devign, per-seed recall:    (std 10.38)
```

A **single run** of Devign could report F1 anywhere in a 10-point band. The paper's Q2 claim of
+6.37 F1 over flat summation is smaller than this spread. We measured that difference at
**−1.47 F1** over 5 seeds with the sign flipping — i.e. the direction of the paper's headline
ablation, on its own headline metric, is not stable.

We were caught by this ourselves: at 3 seeds we recorded "the Conv module's advantage does not
reproduce" because the accuracy sign flipped 2/3; at 5 seeds it does not flip at all. Small-sample
artifacts cut both ways.

*Fairness note:* single-run reporting was common in 2019 ML papers. It is nonetheless a
shortcoming, and one the field has since largely corrected. *Severity: high for the ablation
claims, medium for the headline.*

### 2.5 F1@0.5 is the wrong selection metric for this data, and the paper leans on it

On a 43–51% positive corpus, a classifier that predicts "vulnerable" almost always scores F1
≈ 60–67% while being useless. A properly trained model that balances precision against recall
cannot beat it at a fixed 0.5 threshold.

Demonstrated directly: we trained Devign under the paper's own configuration (F1 selection,
lr 1e-4, no schedule, no threshold tuning) against our default:

| metric | paper's configuration | our default | delta |
|---|---|---|---|
| **F1** | **58.36 ± 0.23** | 57.56 ± 2.57 | **+0.80** |
| accuracy | 57.90 ± 0.74 | **62.76 ± 1.43** | −4.86 |
| recall | **67.11 ± 0.55** | 57.60 ± 5.01 | +9.51 |
| precision | 51.63 ± 0.69 | **57.69 ± 1.78** | −6.06 |
| ROC-AUC | 63.30 ± 0.79 | **68.67 ± 1.54** | −5.36 |

**The paper's configuration scores higher F1 while being a worse classifier on every
threshold-free measure.** It buys 9.51 points of recall for 6.06 of precision — over-predicting
the positive class, exactly the trade F1 rewards. Reporting ROC-AUC or PR-AUC alongside would have
made this visible. *Severity: medium.*

### 2.6 The baseline comparison does not hold up

The paper's Table 2 has Devign beating all four baselines. We ran all four on the released data:

| model | Combined accuracy | Combined F1 |
|---|---|---|
| **CNN** | **67.70** | 56.25 |
| 3-layer BiLSTM | 66.06 | 55.87 |
| **BiLSTM + Attention** | 64.08 | **63.67** |
| Devign (5 seeds) | 63.50 ± 0.83 | 56.27 ± 4.57 |
| Metrics + XGBoost | 60.36 | 18.15 |
| majority class | 56.05 | 0.00 |

**CNN beats Devign by 4.20 accuracy; BiLSTM+Attention by 7.40 F1.** Three of four baselines match
or beat the graph model.

A CNN over token sequences has no AST, no control-flow graph, no data-flow edges and no message
passing. Composite program structure is Devign's entire premise; on this data it does not show up
as an advantage.

*Caveat, stated honestly:* our baselines are single-run and carry no error bars, and the F1 gap is
comparable to Devign's own ±4.57 spread. The accuracy gaps exceed Devign's ±0.83 and are the
sounder claim. This is evidence that the premise is unsupported *on the released half of the
data*, not proof that it is false. *Severity: high if it replicates with seeded baselines.*

### 2.7 Critical hyperparameters are unspecified

The paper does not state: dropout rate, L2 weight, type-embedding dimension, Conv channel count,
MLP hidden width, or what the `AVG` in Eq. 9 averages over. It also does not specify how non-leaf
AST nodes obtain a Code feature — Fig. 2 shows only a source span.

That last one is not cosmetic. Under the literal reading (internal nodes carry Type only),
**44.2% of a representative function's nodes end up with an entirely blank code half** — measured
on the same function under both interpretations. With T = 6 against a median AST depth of 11,
those nodes never receive a token vector from anywhere.

A reproducer must guess, and different guesses give materially different models.
*Severity: medium.*

### 2.8 The CVE generalisation claim (Q5) cannot be checked

Sec 3.4 reports results on 40 CVEs / 112 functions. **That set was never released.** We
substitute an unseen-commit holdout and label it as a proxy, never as a CVE result. As published,
the claim is unverifiable. *Severity: medium.*

---

## 3. Limitations (inherent to the formulation, not errors)

These are not mistakes. They are boundaries of what the approach can do.

### 3.1 Function-level granularity is not actionable

Devign answers *"does this function contain a vulnerability?"* A reviewer receiving that verdict on
a 300-line function still has to find the flaw. The paper acknowledges interpretability as future
work, but the architecture makes it hard: the Conv module **pools the node axis away** before its
MLP heads, so it exposes no per-node quantity at all. The best localisation obtainable from it is
input-gradient saliency, which is a post-hoc proxy rather than something the model computes.

### 3.2 Labels are commit-derived, therefore noisy

A function is "vulnerable" if a fix commit touched it. Fix commits also contain refactoring,
renaming and unrelated cleanup, so some positives contain no vulnerability. Negatives are
"not yet known to be vulnerable", not "safe". This ceilings any model trained on it, and
independent audits of similar corpora put label accuracy well below 100%.

### 3.3 T = 6 cannot reach the top of a typical tree

Message passing runs for T = 6 steps. Median AST depth in this corpus is **11** (Q3 14, p90 18).
A token at a leaf therefore **cannot** influence the function root — not "arrives late", but never
arrives. This is a structural bound on what the composite graph can express, and the paper neither
measures nor discusses it.

### 3.4 The composite graph is only as good as its extractor

The paper uses Joern. CFG and DFG edges are heuristic static approximations; pointer aliasing,
indirect calls and macros degrade them. A "data-flow edge" is a guess, and errors propagate
silently into the representation.

### 3.5 One benchmark, one language, four projects

C only, four large open-source C projects, one labelling methodology. Nothing in the paper
establishes that the result transfers to other languages, smaller codebases, or differently
labelled data.

---

## 4. Proposed solution

> The brief asks for **at least one** proposed scientific solution. §4.1 is the main proposal,
> developed in detail. §4.5 lists three smaller ones.

### 4.1 Main proposal — statement-level weak supervision under commit-disjoint evaluation

**The two most serious problems compound.** §2.2 shows the benchmark largely rewards commit
recognition; §3.1 shows the output is not actionable even when correct. Both have the same root:
**the label is attached to the wrong object.** A vulnerability lives in a *statement* or in a
*relation between statements*, but supervision is attached to the whole function, and the split is
drawn over functions rather than over the commits that generated the labels.

**Proposal.** Treat the function as a **bag of statements** and the function-level label as **weak
supervision over that bag**, then evaluate under a commit-disjoint split.

Formally, for a function with statement-level node groups `S₁ … S_m`:

```
e_j    = wᵀ ( tanh(V h_j) ⊙ sigmoid(U h_j) )        gated attention over statements
a      = softmax(e)                                  a distribution, not a heatmap
z      = Σ_j a_j h_j                                 bag embedding
ŷ      = Sigmoid( Linear(z) )                        function-level prediction
rank(S_j) = a_j                                      statement-level output, for free
```

This is Ilse, Tomczak & Welling's attention-MIL (ICML 2018) applied so that the **standard MIL
assumption** — *a bag is positive iff at least one instance is positive* — matches the semantics of
the task exactly: a function is vulnerable iff at least one statement is.

**Why this should work, and why it is not just "add attention":**

1. **It matches the label-generating process.** The label comes from a diff over statements. MIL
   is the estimator designed for precisely this situation — bag labels, instance predictions, no
   instance supervision.
2. **It fixes the optimisation defect for free.** The pooling is linear in the node embeddings, so
   the gradient is alive at initialisation with no rescue term. Measured: 1.0× a linear control
   against Eq. 9's 384× attenuation.
3. **It makes the output actionable.** `a` ranks statements. A reviewer gets "look at line 47",
   not "something is wrong somewhere in these 300 lines".
4. **It makes the leakage measurable rather than hidden.** Under a commit-disjoint split, a model
   that memorised commits has nothing to fall back on, so the statement ranking is the only thing
   that can carry the score.

**Experimental protocol** — this is what makes it a scientific proposal rather than an idea:

| step | detail |
|---|---|
| Data | PrimeVul `*_paired.jsonl` (86–92% measured label accuracy) — vulnerable lines derived by diffing `func_before` against `func_after`. **Not** Big-Vul (25–54% label accuracy) |
| Split | commit-disjoint, mandatory |
| Detection metrics | accuracy, F1, ROC-AUC, PR-AUC, majority baseline, ≥5 seeds, paired Wilcoxon |
| Localisation metrics | Top-1, Top-5, MRR, normalised rank, **IFA** (Initial False Alarms — clean lines read before the first true one) |
| Scored on | **true positives only** — ranking inside a function called clean measures nothing |
| Baselines | random line ranking; **a line-length / cyclomatic prior**; input-gradient saliency from the Conv module (labelled a proxy) |

**Falsification criteria, fixed in advance:**

- If attention **cannot beat the line-length prior** by more than seed variance, the attention has
  learned nothing about vulnerability and the proposal fails. Long lines are disproportionately
  complex expressions, so length alone is real signal.
- If **attention entropy is near-uniform**, MIL has collapsed to mean pooling and any localisation
  win is an artifact of the projection, not the model.
- If detection under commit-disjoint splitting stays near chance for *every* arm, the problem is
  the benchmark rather than the readout, and no architecture change will rescue it.

**Predicted failure mode, stated before measuring:** the MIL assumption fits **single-site** flaws
(CWE-787 out-of-bounds write, CWE-190 integer overflow) and should fit **relational** flaws poorly
(CWE-416 use-after-free, CWE-362 race), because there the vulnerability lives in a *relation
between* statements, not in any one of them. A per-CWE breakdown either confirms this or refutes
it; both are findings.

**Cost.** The readout is ~60 lines. The trunk, data pipeline and training loop are unchanged. The
expensive part is the evaluation, not the model.

**Honest status.** We implemented the detector half of this and it **did not win on detection** —
it ties on ROC-AUC and PR-AUC and loses ~1.9 accuracy points on 5 of 5 seeds. The localisation
half, which is the part the proposal actually rests on, is **not yet measured**. So this is a
proposal with a partially negative preliminary result, presented as such.

### 4.2 Why not simply fix Eq. 9?

We tried. Adding a learnable scale and bias to the graph-level logit removes the dead start —
epoch-1 F1 goes 0.00 → 52.36 and probability spread 0.001 → 0.636. But over five seeds it changes
ROC-AUC by only **+0.42 (3/5 seeds)**. It relocates the decision boundary; it does not produce a
better model. Patching the symptom is not worth a paper.

### 4.3 Why not simply normalise Eq. 5?

Flat summation saturates because it sums an unnormalised per-node logit over up to 500 nodes.
Dividing by node count would fix that. But Eq. 5 is already within 1.55 ROC-AUC of the fixed Conv
module, so the ceiling on this fix is small — and it deviates from the paper's equation as written,
so it belongs in an ablation, not in a reproduction.

### 4.4 The uncomfortable alternative

Given §2.6 — three sequence baselines matching or beating the graph model — a fair reading is that
**the composite-graph premise may not be what carries the signal on this dataset**. If seeded
baselines confirm it, the productive research question changes from *"which readout?"* to
*"does program structure help at all here, and if not, is that the data or the method?"* That is a
more interesting question than either paper asks, and this reproduction is not in a position to
answer it.

### 4.5 Smaller proposals

1. **Report ROC-AUC and PR-AUC alongside accuracy/F1.** Costs nothing; would have exposed §2.5
   immediately.
2. **Raise T, or add a virtual global node.** Median AST depth 11 against T = 6 means the upper
   half of a typical tree is unreachable (§3.3). A single global node connected to all others
   bounds the path length at 2 for O(m) extra edges.
3. **Release the data, or scope the claims to what was released.** §2.1 would then not exist.

---

## 5. Summary

| # | issue | type | severity | evidence |
|---|---|---|---|---|
| 2.1 | Half the dataset never released | shortcoming | **high** | 2 of 4 projects public |
| 2.2 | 65% commit leakage; AUC 70.83 → 54.64 | shortcoming | **high** | measured |
| 2.3 | Eq. 9 degenerate at init (384×); Eq. 5 saturated | shortcoming | medium | measured |
| 2.4 | Single-run reporting; F1 spread ±4.57 | shortcoming | high | measured |
| 2.5 | F1@0.5 selection rewards over-prediction | shortcoming | medium | measured |
| 2.6 | Three sequence baselines beat the graph model | shortcoming | high* | measured, single-run |
| 2.7 | Key hyperparameters unspecified; 44.2% blank features | shortcoming | medium | measured |
| 2.8 | CVE set never released | shortcoming | medium | — |
| 3.1 | Function-level output is not actionable | limitation | — | structural |
| 3.2 | Commit-derived labels are noisy | limitation | — | structural |
| 3.3 | T = 6 vs median AST depth 11 | limitation | — | measured |
| 3.4 | Heuristic CFG/DFG extraction | limitation | — | structural |
| 3.5 | One language, four projects | limitation | — | structural |

\* pending seeded baselines.

**The fairest overall assessment.** Devign was a well-motivated 2019 paper that introduced a
sensible representation and was reported to the standards of its time. What has not aged well is
the *evidence*: the headline number rests on data nobody can obtain, the benchmark rewards commit
recognition to a degree nobody measured, the central architectural novelty is degenerate at
initialisation, and on the released half of the data simpler models do at least as well. None of
that makes the composite-graph idea wrong. It makes the paper's specific quantitative claims
unsupportable today, and it points at a better-posed version of the same problem: **statement-level
prediction, weak supervision, commit-disjoint evaluation.**

---

## References

- Zhou et al. *Devign.* NeurIPS 2019. [arXiv:1909.03496](https://arxiv.org/abs/1909.03496)
- Ilse, Tomczak, Welling. *Attention-based Deep Multiple Instance Learning.* ICML 2018.
- Chakraborty et al. *Deep Learning based Vulnerability Detection: Are We There Yet?* TSE 2022.
- Ding et al. *PrimeVul: Vulnerability Detection with Code Language Models.* 2024.
- Fu & Tantithamthavorn. *LineVul: Transformer-based Line-Level Vulnerability Prediction.* MSR 2022.
- Jain & Wallace. *Attention is not Explanation.* NAACL 2019.
- Wiegreffe & Pinter. *Attention is not not Explanation.* EMNLP 2019.
- Alon et al. *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR 2021.
