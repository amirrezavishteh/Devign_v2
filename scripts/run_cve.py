"""CLI: Q5 -- unseen-commit vulnerable-function holdout (see evaluation/cve_eval.py).

NOT a CVE or zero-day test: the paper's 40-CVE / 112-function set was never released. This
evaluates the trained Devign model on vulnerable functions whose commit contributed nothing to
training, which is the closest honest proxy available from the released data.

Usage:
    python -m scripts.run_cve --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from data.graph_builder import EDGE_TYPES
from data.word2vec_embed import NodeFeaturizer
from evaluation.cve_eval import evaluate_holdout
from models.devign import build_model
from scripts.train import artifact_dir
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

    featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
    model = build_model("devign", cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=len(featurizer.type_vocab),
                        num_edge_types=len(EDGE_TYPES))
    model_path = os.path.join(artifact_dir(cfg, "devign", args.project), "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    res = evaluate_holdout(cfg, model, device)
    print(f"[q5] {res['kind']} -- {res['num_functions']} functions")
    if res["num_functions"] == 0:
        print(f"[q5] {res.get('note', 'no functions available')}")
    else:
        print(f"[q5] NOTE: {res['not_a_cve_test']}")
        print(f"[q5] overall accuracy: {res['overall_accuracy']:.2f}%  "
              f"({res['num_commits']} distinct commits)")
        for proj, acc in res["per_project_accuracy"].items():
            print(f"      {proj:<16} {acc:.2f}%")
    with open(os.path.join(cfg["project"]["artifacts_dir"], "cve.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
