"""Q5: how does the model do on vulnerable functions it has not seen the context of?

What the paper did
------------------
Zhou et al. scraped the latest 10 CVEs per project (40 CVEs, 112 vulnerable functions), fed them
to the trained model and reported 74.11% accuracy. Every label is 1, so accuracy == recall here.

What we can actually do
-----------------------
That 112-function set was never published, and neither was the CVE-to-function mapping. The
released Devign data (FFmpeg + QEMU) carries `commit_id` but no CVE ids and no dates, so the
paper's experiment cannot be reproduced -- not approximately, not with a substitute corpus.

So this module reports a strictly weaker, honestly-named proxy: **vulnerable functions drawn from
commits that contributed nothing to training**. It answers "does the model generalise to
vulnerabilities from unseen fixes?", which is the useful half of Q5, and it does NOT answer
"can the model find zero-days?". The result must never be presented as a CVE or zero-day number.

The previous implementation generated functions from the same 10 CWE templates the model trained
on, named them `CVE_<project>_<i>`, and reported 100% accuracy. That measured memorisation of a
generator, and is gone.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from data.dataset import DevignDataset, build_samples
from data.graph_builder import EDGE_TYPES
from data.word2vec_embed import NodeFeaturizer
from training.trainer import evaluate


def build_unseen_commit_holdout(splits: dict, per_project: int, seed: int = 42):
    """Vulnerable functions whose commit contributed NO function to training.

    Drawn from the held-out test split when one exists (so they are unseen twice over: unseen
    commit and unseen split), otherwise from validation.
    """
    import random

    train_commits = {fn.commit_id for fn in splits["train"] if fn.commit_id}
    pool = splits.get("test") or splits.get("val") or []
    candidates = [fn for fn in pool
                  if int(fn.target) == 1 and fn.commit_id and fn.commit_id not in train_commits]

    rng = random.Random(seed)
    by_project: dict[str, list] = {}
    for fn in candidates:
        by_project.setdefault(fn.project, []).append(fn)

    out = []
    for project in sorted(by_project):
        items = by_project[project]
        rng.shuffle(items)
        out.extend(items[:per_project])
    return out


def evaluate_holdout(cfg, model, device, per_project: int | None = None,
                     threshold: float = 0.5):
    """Run the model over the unseen-commit vulnerable holdout. Returns metrics + provenance."""
    proc = cfg["data"]["processed_dir"]
    featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
    with open(os.path.join(proc, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    per_project = per_project or cfg["evaluation"].get("holdout_per_project", 56)
    funcs = build_unseen_commit_holdout(splits, per_project, seed=cfg["project"]["seed"])
    if not funcs:
        return {
            "kind": "unseen-commit vulnerable holdout",
            "num_functions": 0,
            "note": "no vulnerable functions with a training-disjoint commit were available",
        }

    samples = build_samples(funcs, featurizer, EDGE_TYPES, cfg["data"]["max_nodes"])
    if not samples:
        return {"kind": "unseen-commit vulnerable holdout", "num_functions": 0,
                "note": "all holdout functions were dropped by the node/parse filters"}

    # Bucketed + node-budget-capped, like every other real-data loader: with up to
    # holdout_per_project * len(projects) functions each up to the 500-node cap, a single
    # batch_size=128 batch (no bucketing) could exceed the configured memory budget.
    from scripts.train import _loader, make_collate_from_cfg

    ds = DevignDataset(samples, len(featurizer.type_vocab))
    loader = _loader(ds, cfg, make_collate_from_cfg(cfg, EDGE_TYPES), shuffle=False)

    metrics, probs, labels, projects = evaluate(model, loader, device, threshold=threshold)
    projects = np.array(projects)
    preds = (probs >= threshold).astype(int)
    per_proj = {}
    for proj in sorted(set(projects.tolist())):
        m = projects == proj
        per_proj[proj] = float((preds[m] == labels[m]).mean() * 100.0)

    return {
        "kind": "unseen-commit vulnerable holdout",
        "not_a_cve_test": ("The paper's 40-CVE / 112-function set was never released; this is a "
                           "commit-disjoint proxy and is not comparable to its 74.11%."),
        "num_functions": len(samples),
        "num_commits": len({fn.commit_id for fn in funcs}),
        "threshold": threshold,
        "overall_accuracy": metrics["accuracy"],
        "per_project_accuracy": per_proj,
    }


# Back-compat alias for older entry points.
evaluate_cves = evaluate_holdout
