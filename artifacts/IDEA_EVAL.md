# Devign-MIL — evaluation against the reproduction

Evaluates gated attention MIL pooling as a replacement for the Devign Conv module (Eq. 6–9),
measured against the Phase 1 reproduction in [`RESULTS.md`](RESULTS.md).

**Status: Phase 3 running.** Sections marked *not measured* are not yet run. Nothing here is
estimated, extrapolated, or carried across from a different configuration.

---

## Pre-registered hypotheses

Stated before Phase 3 was run, and not revised afterwards. Committed in
`README.md §9` and in this file's history.

| | hypothesis | status |
|---|---|---|
| **H1** | Attention-MIL pooling **matches or beats** the FIXED Conv module on detection. *Matching is a success*: the claim is that Eq. 9's complexity buys nothing once its optimisation defect is corrected. | *not measured* |
| **H2** | Attention weights localise the vulnerability to source lines far better than chance, and better than any localisation obtainable from the Conv module. | *not measured* |
| **H3** | MIL needs no `logit_affine` rescue, because it is linear in the node embeddings and therefore has a live gradient at initialisation. | **confirmed** |

### H3 — confirmed

Measured on a real 128-graph batch at initialisation, identical trunk and seed across arms
(`python -m scripts.measure_readout --config configs/a100_codexglue.yaml`):

| readout | logit range | prob spread | loss | trunk ‖grad‖ | vs linear control |
|---|---|---|---|---|---|
| conv, `logit_affine` **false** (paper) | 1.80e-03 | 4.51e-04 | 0.6931 | 1.57e-04 | **384× weaker** |
| conv, `logit_affine` **true** | 9.01e-02 | 2.25e-02 | 0.6935 | 7.79e-03 | 7.7× weaker |
| sum (Eq. 5, Ggrn) | 4.15e+01 | 0.00e+00 | **24.62** | 3.29e+01 | 546× stronger |
| **mil (k=1)** | 8.58e-01 | 2.11e-01 | 0.7153 | 1.58e-01 | **1.0× — matches** |
| linear control | 7.52e-01 | 1.86e-01 | 0.7008 | 6.03e-02 | (reference) |

MIL passes gradient to the trunk indistinguishably from a plain linear head, with no affine, no
scale and no bias. `ln(2) = 0.6931` is the loss of a model at chance.

**Both of the paper's readouts are badly conditioned, in opposite directions.** Eq. 9 collapses —
every probability inside [0.4996, 0.5000]. Eq. 5 saturates — it sums an unnormalised per-node
logit over up to 500 nodes, so logits reach +41 and every probability pins at 1.0, starting at
loss 24.6 on a 43.2%-positive split. Its large gradient is a saturated sigmoid, not health.

> **Threat to the Phase 3 conclusion, stated up front.** Because Eq. 5 saturates, the three-arm
> comparison currently measures MIL against *two* degenerate baselines. "MIL beats flat summation"
> would then be a claim about a broken summation. A node-count-normalised Eq. 5 would be the fair
> comparison; it deviates from the equation as written, so it is recorded here rather than
> silently applied. **This is an open decision, not a finished result.**

---

## 3.1 Detection — *not measured*

One trunk, five seeds per arm, CodeXGLUE split and commit-disjoint split.

| readout | Acc | F1@0.5 | F1@tuned | ROC-AUC | PR-AUC | params |
|---|---|---|---|---|---|---|
| sum (Eq. 5, Ggrn) | | | | | | 553,331 |
| conv, `logit_affine` FALSE ← paper as written | | | | | | 631,410 |
| **conv, `logit_affine` TRUE ← the honest baseline** | | | | | | 631,410 |
| mil (k=1) | | | | | | 630,785 |
| mil (k=4) | | | | | | |
| majority class | | | | | | — |

Parameter counts are already measured: MIL is within **0.1%** of Devign, so neither arm can win on
capacity alone.

**The comparison that counts is MIL vs `conv, logit_affine TRUE`.** Beating Eq. 9 as written is
not a finding — that arm attenuates the trunk gradient 384×, and any reviewer will say so in one
line. `scripts/compare_readouts.py` computes every delta against the fixed baseline for that
reason, and refuses to build a table from two files naming the same arm.

Reported with a paired Wilcoxon signed-rank test across seeds and paired Cohen's d. **With 5 seeds
the smallest attainable two-sided p is 0.0625**, so no comparison here can reach p < 0.05 however
consistent it looks; the floor is printed beside every p-value so it cannot be misread. Phase 1
measured seed-to-seed F1 spread at ±2.06 to ±2.57, so **differences under ~2 points are noise** and
are reported as ties rather than as a rank order.

### Context from Phase 1

The incumbent's own advantage is already within noise. The paper claims the Conv module adds
+4.66 accuracy / +6.37 F1 over flat summation; measured pairwise over 3 seeds it is **+0.98 /
+0.40 with the sign flipping across seeds** (Devign won F1 on 1 of 3). So "MIL beats Eq. 9" is a
weaker claim than it sounds, and "MIL ties Eq. 9" is the outcome the thesis predicts.

## 3.2 Localisation — *not measured*

Primary claim. PrimeVul paired split, vulnerable lines derived by diff. **True positives only** —
ranking lines inside a function the model called clean measures nothing — with the true-positive
count printed beside every mean.

| method | Top-1 | Top-5 | MRR | norm. rank | IFA |
|---|---|---|---|---|---|
| random line ranking | | | | | |
| line-length / complexity prior | | | | | |
| conv module, best available (saliency PROXY) | | | | | |
| mil attention | | | | | |

The **line-length prior row is not optional**. Long lines are disproportionately complex
expressions, so length alone is real signal; if attention cannot beat it, the attention has
learned nothing about vulnerability and the localisation claim collapses.

The Conv row uses input-gradient saliency and is labelled a PROXY everywhere. That is a property
of Eq. 9, not a handicap imposed here: it pools its node axis away before the MLP heads and
exposes no per-node score at all.

**IFA** (Initial False Alarms) is the operational metric — clean lines a reviewer reads before the
first true one. An IFA of 40 makes a tool unusable however good its MRR looks.

## 3.3 Diagnostics — *not measured*

- **Attention entropy per graph.** Near-uniform means MIL has collapsed toward mean pooling and
  any localisation win is an artifact of the projection. Reported as normalised entropy in [0, 1];
  `run_localization` warns above 0.9.
- **Per-CWE breakdown.** *Prediction, stated before measuring:* MIL's core assumption — a bag is
  positive iff ≥1 instance is positive — fits single-site flaws (CWE-787 out-of-bounds write,
  CWE-190 integer overflow) and fits **relational** flaws poorly (CWE-416 use-after-free, CWE-362
  race), because there the vulnerability lives in a relation *between* statements, not in one.
  Whether this holds is a finding either way.
- **Sensitivity to T ∈ {4, 6, 8, 12}** (`scripts/run_t_sweep.sh`). Median AST depth here is 11
  (Q3 14, p90 18) against the paper's T = 6, so the upper half of a typical tree is unreachable
  from its leaves. If results improve with T, over-squashing is bounding them.

## 3.4 Qualitative evidence — *not measured*

`scripts/render_attention.py` emits an HTML page: attention heat overlay beside the actual fix
diff, for true positives across distinct CWEs, **always including at least one failure** with a
note on what the attention pointed at instead.

## 3.5 Verdict — *not measured*

To be answered in plain language whatever the answers turn out to be:

**(a) Does MIL beat the FIXED Conv module on detection, and is the difference larger than
seed-to-seed variance?**

**(b) Does MIL localise better than a trivial prior, and on which CWE classes does it fail?**

**(c) Does MIL avoid the Eq. 9 initialisation pathology without a rescue term?** — *Already
answered: yes.* Trunk gradient 1.0× a linear control against Eq. 9's 384× attenuation, with no
affine, scale or bias.

**(d) What is the strongest argument AGAINST this idea, and what would settle it?**

> Answering (d) does not wait for results. The strongest argument against is that **attention MIL
> pooling is not new** (Ilse et al. 2018), **attention for vulnerability localisation is not new**
> (LineVul and others), and a tie on detection means the operator swap buys nothing measurable on
> the task the paper was evaluated on. The contribution therefore has to be stated narrowly and
> honestly: *the paper's central novelty is a degenerate approximation of a known-better operator,
> here is the measurement that shows it, here is the principled replacement, and here is the
> interpretability the paper listed as future work.* What would settle it is a localisation result
> that clears the line-length prior by a margin exceeding seed variance, on ground truth with
> known label quality — which is why PrimeVul (86–92% label accuracy) is used rather than Big-Vul
> (25–54%).

On the attention-is-not-explanation debate (Jain & Wallace 2019; Wiegreffe & Pinter 2019): this
work does not assert that attention *is* the explanation. It validates attention against
ground-truth vulnerable lines, which is a different and falsifiable claim.

---

## Reproducing

```bash
python -m scripts.run_phase3.sh                    # 5 seeds x 4 arms, then the table
python -m scripts.compare_readouts --dir artifacts/seeds --baseline devign
python -m scripts.run_localization --model mil --model-dir artifacts/seed1/mil/combined \
    --primevul data/primevul/test_paired.jsonl
python -m scripts.render_attention --model mil --model-dir artifacts/seed1/mil/combined \
    --primevul data/primevul/test_paired.jsonl
bash scripts/run_t_sweep.sh
```

Every run writes a manifest (git SHA, resolved-config hash, library versions, seed, device,
per-split hashes). Two runs are comparable only if their split hashes match —
`training.manifest.compare_manifests` checks rather than assumes.
