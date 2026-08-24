"""CLI: commit-disjoint split leakage check (Combined, Devign + Ggrn).

The paper splits randomly (Sec 3.3), which is what the headline Table 2 uses. A random split can
put two functions extracted from the *same* vulnerability-fix commit on both sides -- the model
can then partly recognise the commit rather than the flaw, inflating the reported score. This
script reruns data prep with a commit-disjoint split (`data.split_by=commit`) into a separate
processed directory, trains Devign + Ggrn on Combined, and reports the gap against the random-
split numbers already sitting in artifacts/devign and artifacts/ggrn.

A large gap here means Table 2's headline number is partly commit memorisation; a small gap means
the random-split number is trustworthy.

Usage:
    python -m scripts.run_leakage_check --epochs 100
"""
from __future__ import annotations

import argparse
import copy
import json
import os

from devign_data.dataset import positive_rate
from devign_data.graph_builder import EDGE_TYPES
from devign_data.prepare import prepare
from models.devign import build_model
from scripts.train import artifact_dir, load_graph_loaders
from training.trainer import evaluate, make_train_config, train_model
from training.utils import ensure_dir, load_config, resolve_device_from_config, seed_from_config


def _leakage_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["data"]["split_by"] = "commit"
    cfg["data"]["processed_dir"] = cfg["data"]["processed_dir"].rstrip("/\\") + "_commit_disjoint"
    return cfg


def _load_random_split_metrics(cfg: dict, model_name: str,
                               baseline_json: str | None = None) -> dict | None:
    """The number the leakage gap is measured against.

    Two sources, in order of preference:

    `baseline_json` -- a summary written by scripts.run_seeds, i.e. a MEAN OVER SEEDS. This is the
    right comparison: the gap being measured is a few points, and Phase 1 measured seed-to-seed F1
    spread at +/-2.06 to +/-2.57, so comparing a commit-disjoint run against a single random-split
    run could report seed noise as leakage.

    Otherwise the single-run artifact from scripts.train / scripts.reproduce, which is what exists
    when no sweep has been run. Returns None if neither is present, and the caller reports the
    commit-disjoint number alone rather than inventing a gap.
    """
    if baseline_json and os.path.exists(baseline_json):
        with open(baseline_json) as f:
            summary = json.load(f)
        block = summary.get("val_at_0.5") or {}
        out = {k: v.get("mean") for k, v in block.items() if isinstance(v, dict)}
        out["_source"] = f"{baseline_json} (mean over seeds {summary.get('seeds')})"
        return out if out.get("accuracy") is not None else None
    path = os.path.join(artifact_dir(cfg, model_name, "combined"), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        out = dict(json.load(f)["best_val"])
    out["_source"] = f"{path} (single run)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["devign", "ggrn"], choices=["devign", "ggrn", "mil"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--baseline-json", default=None,
                    help="a scripts.run_seeds summary to measure the gap against; a mean over "
                         "seeds, so the gap is not read off a single noisy run")
    ap.add_argument("--force-prepare", action="store_true",
                    help="rebuild the commit-disjoint processed_dir even if it already exists")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    cfg = _leakage_config(base_cfg)
    seed_from_config(cfg)
    device = resolve_device_from_config(cfg)

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.pkl")
    if args.force_prepare or not os.path.exists(train_path):
        print(f"[leakage] preparing commit-disjoint split -> {cfg['data']['processed_dir']}")
        info = prepare(cfg, verbose=True)
        print("[leakage] prepared:", info)
    else:
        print(f"[leakage] reusing already-built {cfg['data']['processed_dir']}")

    results = {}
    for model_name in args.models:
        train_ds, val_ds, train_loader, val_loader = load_graph_loaders(cfg, EDGE_TYPES)
        model = build_model(model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                            type_vocab_size=train_ds.type_vocab_size,
                            num_edge_types=len(EDGE_TYPES),
                            pos_rate=positive_rate(train_ds))
        tcfg = make_train_config(cfg, device, [s.label for s in train_ds.samples], args.epochs)
        print(f"[leakage] training {model_name} on commit-disjoint Combined "
              f"({len(train_ds)} train / {len(val_ds)} val)")
        _, best = train_model(model, train_loader, val_loader, tcfg, verbose=True)

        random_metrics = _load_random_split_metrics(base_cfg, model_name, args.baseline_json)
        entry = {"commit_disjoint": best, "random_split": random_metrics}
        if random_metrics:
            entry["baseline_source"] = random_metrics.get("_source")
            entry["gap"] = {k: round(random_metrics[k] - best[k], 2)
                            for k in ("accuracy", "f1", "auc", "pr_auc")
                            if random_metrics.get(k) is not None and k in best}
        results[model_name] = entry
        gap = entry.get("gap")
        print(f"[leakage] {model_name}: commit-disjoint acc {best['accuracy']:.2f} "
              f"f1 {best['f1']:.2f}" + (f"  (gap vs random split: {gap})" if gap else ""))

    out_dir = ensure_dir(base_cfg["project"]["artifacts_dir"])
    with open(os.path.join(out_dir, "leakage_check.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
