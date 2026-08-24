"""CLI: build the processed dataset (graphs + word2vec + features) from raw/synthetic data.

Usage:
    python -m scripts.prepare_data --config config.yaml
"""
from __future__ import annotations

import argparse

from devign_data.prepare import prepare
from training.utils import load_config, seed_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_from_config(cfg)
    info = prepare(cfg, verbose=True)
    print("[prepare_data] summary:", info)


if __name__ == "__main__":
    main()
