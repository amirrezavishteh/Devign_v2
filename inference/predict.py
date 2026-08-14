"""Single-function vulnerability inference.

Loads a trained Devign model + the saved featurizer, builds the composite graph for one C
function, and returns P(vulnerable). Usable as a library (predict_function) or CLI.

Usage:
    python -m inference.predict --file path/to/func.c
    python -m inference.predict --code "int f(){...}"
    echo "int f(){...}" | python -m inference.predict
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from data.dataset import sample_from_graph
from data.download import RawFunction
from data.graph_builder import EDGE_TYPES, build_graph
from data.word2vec_embed import NodeFeaturizer
from models.devign import build_model
from scripts.train import artifact_dir, load_threshold, make_collate_from_cfg
from training.utils import load_config, resolve_device


class DevignPredictor:
    def __init__(self, config_path="config.yaml", model_name="devign", device=None,
                 project="combined"):
        self.cfg = load_config(config_path)
        self.device = device or resolve_device(self.cfg["project"]["device"])
        proc = self.cfg["data"]["processed_dir"]
        self.featurizer = NodeFeaturizer.load(os.path.join(proc, "featurizer"))
        self.edge_types = EDGE_TYPES
        # Honour the config's adjacency/sparsity settings so inference batches are built exactly
        # the way the model's training batches were.
        self.collate = make_collate_from_cfg(self.cfg, self.edge_types)
        self.model = build_model(
            model_name, self.cfg, code_dim=self.cfg["embedding"]["word2vec_dim"],
            type_vocab_size=len(self.featurizer.type_vocab), num_edge_types=len(self.edge_types))
        model_path = os.path.join(artifact_dir(self.cfg, model_name, project), "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Train one first, e.g.\n"
                f"    python -m scripts.train --model {model_name} --project {project}\n"
                f"or pass --project with one of the projects you did train.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device,
                                              weights_only=True))
        self.model.to(self.device).eval()
        # The operating point chosen on validation during training, not an assumed 0.5.
        self.threshold = load_threshold(self.cfg, model_name, project)

    @torch.no_grad()
    def predict(self, source: str, threshold: float | None = None) -> dict:
        threshold = self.threshold if threshold is None else threshold
        graph = build_graph(source, max_nodes=self.cfg["data"]["max_nodes"])
        if graph is None or graph.num_nodes == 0:
            return {"error": "could not parse function, it has a syntax error, or it exceeds "
                             "max_nodes", "vulnerable": None, "probability": None}
        # Reuse the exact featurization the training pipeline uses -- notably `dedupe_edges`,
        # without which inference would feed a differently-weighted edge set than the model was
        # trained on (the sparse path accumulates duplicates; training data has none).
        sample = sample_from_graph(
            RawFunction(func=source, target=0, project="inference"),
            graph, self.featurizer, self.edge_types)
        batch = self.collate([sample]).to(self.device)
        prob = torch.sigmoid(self.model(batch)).item()
        return {
            "vulnerable": bool(prob >= threshold),
            "probability": prob,
            "threshold": threshold,
            "num_nodes": graph.num_nodes,
            "edge_counts": {et: len(graph.edges.get(et, [])) for et in self.edge_types},
        }


def predict_function(source: str, config_path="config.yaml", model_name="devign",
                     project="combined") -> dict:
    return DevignPredictor(config_path, model_name, project=project).predict(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="devign")
    ap.add_argument("--project", default="combined",
                    help="which trained model to load (default: the pooled Combined model)")
    ap.add_argument("--file", default=None)
    ap.add_argument("--code", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the model's validation-tuned threshold")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()
    elif args.code:
        source = args.code
    else:
        source = sys.stdin.read()

    if not source.strip():
        print("No source provided.")
        sys.exit(1)

    predictor = DevignPredictor(args.config, args.model, project=args.project)
    result = predictor.predict(source, threshold=args.threshold)
    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(2)
    verdict = "VULNERABLE" if result["vulnerable"] else "NON-VULNERABLE"
    print(f"Prediction : {verdict}")
    print(f"P(vuln)    : {result['probability']:.4f}  (threshold {result['threshold']:.3f})")
    print(f"Nodes      : {result['num_nodes']}")
    print(f"Edges      : {result['edge_counts']}")


if __name__ == "__main__":
    main()
