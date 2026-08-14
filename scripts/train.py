"""CLI: train a graph model (Devign or Ggrn) on the prepared dataset.

Usage:
    python -m scripts.train --model devign --config config.yaml
    python -m scripts.train --model ggrn

The trained model + metadata are saved under <artifacts_dir>/<model>/.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data.dataset import BucketBySizeSampler, DevignDataset, make_collate_fn
from data.graph_builder import EDGE_TYPES
from models.devign import build_model
from training.trainer import evaluate, make_train_config, train_model
from training.utils import ensure_dir, load_config, resolve_device, set_seed


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
    if ds_cfg.get("bucket_by_size", True) and len(ds) > 0:
        sampler = BucketBySizeSampler(
            [s.num_nodes for s in ds.samples], bs,
            max_nodes_per_batch=ds_cfg.get("max_nodes_per_batch"),
            shuffle=shuffle, seed=cfg["project"]["seed"])
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate)


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
        type_vocab_size=train_ds.type_vocab_size, num_edge_types=len(EDGE_TYPES))

    tcfg = make_train_config(cfg, device, [s.label for s in train_ds.samples], epochs)
    model, best = train_model(model, train_loader, val_loader, tcfg, verbose=verbose)

    # Apply the validation-tuned operating point to the held-out test split -- the threshold is a
    # hyperparameter chosen on val, so using it on test stays unbiased.
    threshold = best.get("threshold", 0.5)
    final_metrics, _, _, _ = evaluate(model, val_loader, device, threshold=threshold)

    test_metrics = None
    _, test_loader = load_test_loader(cfg, EDGE_TYPES, project)
    if test_loader is not None:
        test_metrics, _, _, _ = evaluate(model, test_loader, device, threshold=threshold)

    return (model,
            {"best_val": best, "final_val": final_metrics, "test": test_metrics,
             "threshold": threshold},
            train_ds)


def save_graph_model(cfg, model, model_name: str, project: str | None, metrics: dict,
                     type_vocab_size: int) -> str:
    out_dir = ensure_dir(artifact_dir(cfg, model_name, project))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    meta = {
        "model": model_name, "project": project or "combined",
        "code_dim": cfg["embedding"]["word2vec_dim"], "type_vocab_size": type_vocab_size,
        "num_edge_types": len(EDGE_TYPES), "edge_types": EDGE_TYPES,
        # The model's operating point, tuned on validation. Downstream evaluation (Table 3, the
        # Q5 holdout, inference) must use this rather than assuming 0.5.
        "threshold": metrics.get("threshold", 0.5),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["devign", "ggrn"], default="devign")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--project", default=None,
                    help="restrict to one project's data (default: pooled/combined)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])
    print(f"[train] model={args.model} project={args.project or 'combined'} device={device}")

    model, metrics, train_ds = train_graph_model(
        cfg, args.model, device, epochs=args.epochs, project=args.project)
    best = metrics["best_val"]
    print(f"[train] best val: acc {best['accuracy']:.2f} f1 {best['f1']:.2f} "
          f"auc {best.get('auc', float('nan')):.2f} @ threshold {metrics['threshold']:.3f}")
    if metrics["test"]:
        t = metrics["test"]
        print(f"[train] held-out test: acc {t['accuracy']:.2f} f1 {t['f1']:.2f} "
              f"auc {t.get('auc', float('nan')):.2f}")

    out_dir = save_graph_model(cfg, model, args.model, args.project, metrics,
                               train_ds.type_vocab_size)
    print(f"[train] saved to {out_dir}")


if __name__ == "__main__":
    main()
