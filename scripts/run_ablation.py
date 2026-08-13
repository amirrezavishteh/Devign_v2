"""CLI: single-edge ablation study (Q3). Trains Devign (and optionally Ggrn) per edge type.

Usage:
    python -m scripts.run_ablation --models devign ggrn --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os

from data.graph_builder import EDGE_TYPES
from evaluation.ablation import run_single_edge
from evaluation.report import format_ablation
from models.devign import build_model
from scripts.train import load_graph_loaders
from training.trainer import TrainConfig, train_model
from training.utils import ensure_dir, load_config, resolve_device, set_seed

# Single artifact path for this study, also used by scripts.reproduce so the two entry points
# never disagree on where the composite/ablation results live.
ARTIFACT_PATH = os.path.join("ablation", "ablation.json")


def run_composite(cfg, model_name, device, epochs, project=None):
    """The all-7-edge-types model, trained fresh -- NOT reused from scripts.train's artifact,
    so this study is self-contained and reproducible from `prepare_data` alone."""
    train_ds, val_ds, train_loader, val_loader = load_graph_loaders(cfg, EDGE_TYPES, project)
    model = build_model(model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=train_ds.type_vocab_size, num_edge_types=len(EDGE_TYPES))
    tr = cfg["training"]
    n_pos = sum(s.label for s in train_ds.samples)
    pw = ((len(train_ds) - n_pos) / n_pos) if n_pos > 0 else 1.0
    tcfg = TrainConfig(lr=tr["lr"], batch_size=tr["batch_size"], epochs=epochs or tr["epochs"],
                       patience=tr["early_stopping_patience"], l2_weight=tr["l2_weight"],
                       grad_clip=tr["grad_clip"], device=device, pos_weight=pw)
    _, best = train_model(model, train_loader, val_loader, tcfg, verbose=False)
    return best


def run_ablation(cfg, models, device, epochs=None, project=None, verbose=True):
    """Returns {model_name: {edge_type: metrics, ..., 'Composite': metrics}}."""
    ablation_edges = cfg["evaluation"]["ablation_edge_types"]
    results = {}
    for model_name in models:
        results[model_name] = {}
        for edge in ablation_edges:
            if verbose:
                print(f"[ablation] {model_name} / {edge}")
            best = run_single_edge(cfg, model_name, edge, device, epochs=epochs, project=project)
            results[model_name][edge] = best
        if verbose:
            print(f"[ablation] {model_name} / Composite")
        results[model_name]["Composite"] = run_composite(cfg, model_name, device, epochs, project)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["devign"], choices=["devign", "ggrn"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--project", default=None, help="restrict to one project (default: Combined)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])

    results = run_ablation(cfg, args.models, device, epochs=args.epochs, project=args.project)

    print(format_ablation(results))
    out_path = os.path.join(cfg["project"]["artifacts_dir"], ARTIFACT_PATH)
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
