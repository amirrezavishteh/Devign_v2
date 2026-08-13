"""CLI: train the Table-2 baselines and report per-project metrics.

Usage:
    python -m scripts.train_baselines --config config.yaml
    python -m scripts.train_baselines --project qemu        # one project only

Artifacts land in <artifacts_dir>/baselines/<project>/<name>.json, the same layout
scripts.reproduce uses, so the two entry points share results instead of overwriting each other.
"""
from __future__ import annotations

import argparse
import json
import os

from training.train_baselines import (train_bilstm, train_cnn,
                                      train_metrics_xgboost)
from training.utils import ensure_dir, load_config, resolve_device, set_seed

BASELINES = {
    "bilstm": lambda cfg, dev, proj: train_bilstm(cfg, dev, attention=False, project=proj)[1],
    "bilstm_att": lambda cfg, dev, proj: train_bilstm(cfg, dev, attention=True, project=proj)[1],
    "cnn": lambda cfg, dev, proj: train_cnn(cfg, dev, project=proj)[1],
    "xgboost": lambda cfg, dev, proj: train_metrics_xgboost(cfg, project=proj)[1],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--project", default=None,
                    help="one project (default: every project plus pooled 'combined')")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["project"]["device"])

    projects = [args.project] if args.project else list(cfg["data"]["projects"]) + ["combined"]

    for project in projects:
        out_dir = ensure_dir(os.path.join(cfg["project"]["artifacts_dir"], "baselines", project))
        for name, fn in BASELINES.items():
            print(f"[baselines] {name}/{project}")
            metrics = fn(cfg, device, project)
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  acc {metrics['accuracy']:.2f}  f1 {metrics['f1']:.2f}")
        print(f"[baselines] saved metrics to {out_dir}")


if __name__ == "__main__":
    main()
