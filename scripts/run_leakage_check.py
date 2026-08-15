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

from data.dataset import positive_rate
from data.graph_builder import EDGE_TYPES
from data.prepare import prepare
from models.devign import build_model
from scripts.train import artifact_dir, load_graph_loaders
from training.trainer import evaluate, make_train_config, train_model
from training.utils import ensure_dir, load_config, resolve_device, set_seed


def _leakage_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["data"]["split_by"] = "commit"
    cfg["data"]["processed_dir"] = cfg["data"]["processed_dir"].rstrip("/\\") + "_commit_disjoint"
    return cfg


def _load_random_split_metrics(cfg: dict, model_name: str) -> dict | None:
    """The Combined-project random-split run, trained by scripts.reproduce / scripts.train
    --project combined -- this is the number the leakage check's gap is measured against."""
    path = os.path.join(artifact_dir(cfg, model_name, "combined"), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["best_val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["devign", "ggrn"], choices=["devign", "ggrn"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force-prepare", action="store_true",
                    help="rebuild the commit-disjoint processed_dir even if it already exists")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    cfg = _leakage_config(base_cfg)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])

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

        random_metrics = _load_random_split_metrics(base_cfg, model_name)
        entry = {"commit_disjoint": best, "random_split": random_metrics}
        if random_metrics:
            entry["gap"] = {k: round(random_metrics[k] - best[k], 2)
                            for k in ("accuracy", "f1") if k in best and k in random_metrics}
        results[model_name] = entry
        gap = entry.get("gap")
        print(f"[leakage] {model_name}: commit-disjoint acc {best['accuracy']:.2f} "
              f"f1 {best['f1']:.2f}" + (f"  (gap vs random split: {gap})" if gap else ""))

    out_dir = ensure_dir(base_cfg["project"]["artifacts_dir"])
    with open(os.path.join(out_dir, "leakage_check.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
