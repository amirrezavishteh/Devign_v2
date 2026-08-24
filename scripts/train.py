"""CLI: train a graph model (Devign, Ggrn or Mil) on the prepared dataset.

Usage:
    python -m scripts.train --model devign --config config.yaml
    python -m scripts.train --model ggrn
    python -m scripts.train --model mil     # attention MIL readout

The trained model + metadata are saved under <artifacts_dir>/<model>/.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from devign_data.dataset import (BucketBySizeSampler, DevignDataset, make_collate_fn,
                          positive_rate)
from devign_data.graph_builder import EDGE_TYPES
from models.devign import build_model
from training.manifest import build_manifest
from training.metrics import require_unbiased
from training.trainer import evaluate, make_train_config, train_model
from training.utils import (ensure_dir, load_config, loader_generator,
                            resolve_device_from_config, seed_from_config, seed_worker)


def make_collate_from_cfg(cfg, edge_types):
    ds_cfg = cfg.get("dataset", {})
    return make_collate_fn(
        edge_types,
        add_self_loops=ds_cfg.get("add_self_loops", False),
        normalize_adj=ds_cfg.get("normalize_adj", False),
        sparse=ds_cfg.get("sparse", True),
    )


def _subset(ds: DevignDataset, project: str | None) -> DevignDataset:
    """Restrict a split to one project (the paper reports per-dataset columns)."""
    if not project or project == "combined":
        return ds
    keep = [s for s in ds.samples if s.project == project]
    return DevignDataset(keep, ds.type_vocab_size, edge_types=ds.edge_types)


def _loader(ds, cfg, collate, shuffle: bool):
    """Bucketed, node-budget-capped loader. Applied to eval splits too: the budget is what keeps
    a batch of large graphs from exhausting VRAM, and that applies regardless of shuffling."""
    ds_cfg = cfg.get("dataset", {})
    bs = cfg["training"]["batch_size"]
    seed = cfg["project"]["seed"]
    # Explicit generator + per-worker seeding. Both are no-ops at today's num_workers=0 default,
    # and both are what silently breaks reproducibility the moment someone raises it: workers
    # otherwise inherit one RNG state, and shuffling otherwise consumes the global torch stream,
    # coupling batch order to how many random numbers the model happened to draw first.
    common = dict(collate_fn=collate, worker_init_fn=seed_worker,
                  generator=loader_generator(seed))
    if ds_cfg.get("bucket_by_size", True) and len(ds) > 0:
        sampler = BucketBySizeSampler(
            [s.num_nodes for s in ds.samples], bs,
            max_nodes_per_batch=ds_cfg.get("max_nodes_per_batch"),
            shuffle=shuffle, seed=seed)
        return DataLoader(ds, batch_sampler=sampler, **common)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, **common)


def load_graph_loaders(cfg, edge_types, project: str | None = None):
    proc = cfg["data"]["processed_dir"]
    train_ds = _subset(DevignDataset.load(os.path.join(proc, "train.pkl")), project)
    val_ds = _subset(DevignDataset.load(os.path.join(proc, "val.pkl")), project)
    collate = make_collate_from_cfg(cfg, edge_types)
    return (train_ds, val_ds,
            _loader(train_ds, cfg, collate, shuffle=True),
            _loader(val_ds, cfg, collate, shuffle=False))


def load_test_loader(cfg, edge_types, project: str | None = None):
    """The held-out split, untouched by training and early stopping. None if not prepared."""
    path = os.path.join(cfg["data"]["processed_dir"], "test.pkl")
    if not os.path.exists(path):
        return None, None
    test_ds = _subset(DevignDataset.load(path), project)
    if len(test_ds) == 0:
        return None, None
    collate = make_collate_from_cfg(cfg, edge_types)
    return test_ds, _loader(test_ds, cfg, collate, shuffle=False)


def artifact_dir(cfg, model_name: str, project: str | None) -> str:
    project = project or "combined"
    return os.path.join(cfg["project"]["artifacts_dir"], model_name, project)


def load_threshold(cfg, model_name: str, project: str | None, default: float = 0.5) -> float:
    """The operating point tuned on validation when this model was trained (meta.json).

    Downstream evaluation must use it rather than assuming 0.5, or it reports a different
    classifier than the one that was selected.
    """
    meta_path = os.path.join(artifact_dir(cfg, model_name, project), "meta.json")
    if not os.path.exists(meta_path):
        return default
    with open(meta_path) as f:
        return float(json.load(f).get("threshold", default))


def train_graph_model(cfg, model_name: str, device, epochs=None, project: str | None = None,
                      verbose: bool = True):
    """Train one graph model on one project's split (or the pooled 'combined' split).

    Returns (model, best_val_metrics, test_metrics_or_None). Also used directly by
    scripts.reproduce so the two entry points share one code path.
    """
    train_ds, val_ds, train_loader, val_loader = load_graph_loaders(cfg, EDGE_TYPES, project)
    if verbose:
        label = project or "combined"
        print(f"[train] {model_name}/{label}: {len(train_ds)} train / {len(val_ds)} val graphs")

    model = build_model(
        model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
        type_vocab_size=train_ds.type_vocab_size, num_edge_types=len(EDGE_TYPES),
        pos_rate=positive_rate(train_ds))

    # Checkpoint beside the model so an interrupted run resumes at the last completed epoch.
    out_dir = artifact_dir(cfg, model_name, project)
    tcfg = make_train_config(
        cfg, device, [s.label for s in train_ds.samples], epochs,
        checkpoint_path=os.path.join(out_dir, "checkpoint.pt"),
        csv_log_path=os.path.join(out_dir, "training_curve.csv"))
    model, best = train_model(model, train_loader, val_loader, tcfg, verbose=verbose)

    # The threshold is a hyperparameter chosen on validation, so applying it to the held-out TEST
    # split stays unbiased -- that split had no say in choosing it.
    threshold = best.get("threshold", 0.5)
    # Validation scored at that same threshold is biased by construction, and is tagged as such so
    # `metrics.require_unbiased` refuses it if anything tries to table it. Kept because it is a
    # useful diagnostic (it upper-bounds what the operating point can do), not a result.
    final_metrics, _, _, _ = evaluate(model, val_loader, device, threshold=threshold,
                                      split="val", fitted_on_split="val")

    test_metrics = None
    test_ds, test_loader = load_test_loader(cfg, EDGE_TYPES, project)
    if test_loader is not None:
        test_metrics, _, _, _ = evaluate(model, test_loader, device, threshold=threshold,
                                         split="test", fitted_on_split="val")

    # Provenance for this run. Carried inside the metrics dict so neither caller's signature has
    # to change; save_graph_model lifts it out into meta.json.
    splits = {"train": train_ds.samples, "val": val_ds.samples}
    if test_ds is not None:
        splits["test"] = test_ds.samples
    manifest = build_manifest(cfg, cfg["project"]["seed"], device, splits)

    return (model,
            {"best_val": best, "final_val": final_metrics, "test": test_metrics,
             "threshold": threshold, "manifest": manifest},
            train_ds)


def save_graph_model(cfg, model, model_name: str, project: str | None, metrics: dict,
                     type_vocab_size: int) -> str:
    out_dir = ensure_dir(artifact_dir(cfg, model_name, project))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    # Provenance belongs in meta.json, so metrics.json stays a file of numbers only.
    metrics = dict(metrics)
    manifest = metrics.pop("manifest", None)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    meta = {
        "model": model_name, "project": project or "combined",
        "code_dim": cfg["embedding"]["word2vec_dim"], "type_vocab_size": type_vocab_size,
        "num_edge_types": len(EDGE_TYPES), "edge_types": EDGE_TYPES,
        # The model's operating point, tuned on validation. Downstream evaluation (Table 3, the
        # Q5 holdout, inference) must use this rather than assuming 0.5.
        "threshold": metrics.get("threshold", 0.5),
        # Git SHA, resolved-config hash, library versions, seed, device and per-split hashes.
        # Two runs are comparable only if their split hashes match -- see training/manifest.py.
        "manifest": manifest,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["devign", "ggrn", "mil"], default="devign")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--project", default=None,
                    help="restrict to one project's data (default: pooled/combined)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_from_config(cfg)
    device = resolve_device_from_config(cfg)
    print(f"[train] model={args.model} project={args.project or 'combined'} device={device}")

    model, metrics, train_ds = train_graph_model(
        cfg, args.model, device, epochs=args.epochs, project=args.project)

    # `best_val` is scored at the fixed 0.5 threshold, so it is unbiased and comparable to the
    # paper. require_unbiased is called rather than assumed: it is the guard, not a comment.
    best = require_unbiased(metrics["best_val"], "scripts.train best_val")
    print(f"[train] best val @0.5: acc {best['accuracy']:.2f} f1 {best['f1']:.2f} "
          f"auc {best.get('auc', float('nan')):.2f} pr-auc {best.get('pr_auc', float('nan')):.2f}")
    print(f"[train] tuned threshold: {metrics['threshold']:.3f}")

    if metrics["test"]:
        t = require_unbiased(metrics["test"], "scripts.train test")
        print(f"[train] held-out test @tuned: acc {t['accuracy']:.2f} f1 {t['f1']:.2f} "
              f"auc {t.get('auc', float('nan')):.2f} pr-auc {t.get('pr_auc', float('nan')):.2f}")
    elif cfg["training"].get("tune_threshold", True):
        # The paper_split path: 75/25 train/val and no test set, so there is no split the tuned
        # threshold has not already seen. Say so instead of printing a number that looks unbiased.
        print("[train] NO TEST SPLIT (data.paper_split: true). The tuned threshold cannot be "
              "reported: every split it could be applied to is the one it was fitted on. Report "
              "the val@0.5 row above, or set data.paper_split: false to hold out a test split.")

    out_dir = save_graph_model(cfg, model, args.model, args.project, metrics,
                               train_ds.type_vocab_size)
    print(f"[train] saved to {out_dir}")


if __name__ == "__main__":
    main()
