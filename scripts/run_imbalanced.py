"""CLI: Q4 / Table 3 imbalanced evaluation vs static analyzers.

Loads the trained Devign model, builds a 10%-vulnerable test set from the val split, and compares
Devign against Cppcheck / Flawfinder (and reports their fallback mode if the tools are absent).

Usage:
    python -m scripts.run_imbalanced --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import torch
from torch.utils.data import DataLoader

from data.dataset import DevignDataset, build_samples
from data.graph_builder import EDGE_TYPES
from data.word2vec_embed import NodeFeaturizer
from evaluation.imbalanced import evaluate_static_analyzers, make_imbalanced
from evaluation.report import format_table3, per_project_eval
from models.devign import build_model
from scripts.train import _loader, artifact_dir, make_collate_from_cfg
from training.trainer import evaluate
from training.utils import load_config, resolve_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--project", default="combined",
                    help="which trained Devign model to evaluate (default: combined)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])
    proc = cfg["data"]["processed_dir"]

    with open(os.path.join(proc, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    # Prefer the held-out test split (untouched by training/early stopping); fall back to val
    # when running with data.paper_split (no test set).
    source_split = splits.get("test") or splits["val"]
    imb = make_imbalanced(source_split, cfg["data"]["imbalanced_vuln_ratio"],
                          cfg["project"]["seed"])
    n_vuln = sum(f.target for f in imb)
    print(f"[imbalanced] {len(imb)} functions, {n_vuln} vulnerable "
          f"({100 * n_vuln / max(1, len(imb)):.1f}%)")

    results = {}

    # Static analyzers. Real tools if resolvable (config paths or PATH); otherwise a clearly
    # labelled regex fallback that is never presented as the real tool's result.
    sa = evaluate_static_analyzers(imb, cfg["evaluation"].get("static_analyzer_paths"))
    for label, r in sa.items():
        results[label] = r["metrics"]

    # Devign.
    meta_dir = artifact_dir(cfg, "devign", args.project)
    if os.path.exists(os.path.join(meta_dir, "model.pt")):
        featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
        samples = build_samples(imb, featurizer, EDGE_TYPES, cfg["data"]["max_nodes"])
        ds = DevignDataset(samples, len(featurizer.type_vocab))
        loader = _loader(ds, cfg, make_collate_from_cfg(cfg, EDGE_TYPES), shuffle=False)
        model = build_model("devign", cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                            type_vocab_size=len(featurizer.type_vocab),
                            num_edge_types=len(EDGE_TYPES))
        model.load_state_dict(torch.load(os.path.join(meta_dir, "model.pt"), map_location=device,
                                         weights_only=True))
        model.to(device)
        _, probs, labels, projects = evaluate(model, loader, device)
        results["Devign (Composite)"] = per_project_eval(probs, labels, projects)
    else:
        print("[imbalanced] no trained Devign model found; run scripts.train first.")

    print(format_table3(results, cfg["data"]["projects"]))
    out_dir = cfg["project"]["artifacts_dir"]
    with open(os.path.join(out_dir, "table3.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
