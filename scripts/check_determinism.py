"""Phase 0 gate: prove that same seed means same model.

Runs the same short training twice from a clean state and requires the two `metrics.json` files
to be byte-identical. Not `allclose` -- byte-identical. A tolerance-level agreement is exactly
what a non-deterministic scatter produces, and the drift it hides (measured at 6e-7 per forward
on this repo's atomic path) compounds over thousands of steps into a genuinely different model.

Usage:
    python -m scripts.check_determinism --epochs 3
    python -m scripts.check_determinism --epochs 3 --model ggrn
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile

from devign_data.graph_builder import EDGE_TYPES
from scripts.train import save_graph_model, train_graph_model
from training.manifest import compare_manifests
from training.utils import load_config, resolve_device_from_config, seed_from_config


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _one_run(cfg: dict, model_name: str, device: str, epochs: int, out_root: str) -> str:
    """Train once into an isolated artifacts root; return that root."""
    run_cfg = json.loads(json.dumps(cfg))          # deep copy, so runs cannot share mutated state
    run_cfg["project"]["artifacts_dir"] = out_root
    # Re-seed immediately before the run: the point is that the seed alone determines everything,
    # so run 2 must not inherit any RNG state from run 1.
    seed_from_config(run_cfg)
    model, metrics, train_ds = train_graph_model(
        run_cfg, model_name, device, epochs=epochs, verbose=False)
    save_graph_model(run_cfg, model, model_name, None, metrics, train_ds.type_vocab_size)
    return out_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["devign", "ggrn", "mil"], default="devign")
    ap.add_argument("--epochs", type=int, default=3,
                    help="short by design; drift shows up in the first epoch if it exists")
    ap.add_argument("--keep", action="store_true", help="keep the two run directories")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device_from_config(cfg)
    print(f"[determinism] model={args.model} device={device} epochs={args.epochs} "
          f"seed={cfg['project']['seed']}")

    tmp_root = tempfile.mkdtemp(prefix="devign-determinism-")
    try:
        roots = []
        for i in (1, 2):
            root = os.path.join(tmp_root, f"run{i}")
            print(f"[determinism] run {i}/2 ...")
            roots.append(_one_run(cfg, args.model, device, args.epochs, root))

        a = os.path.join(roots[0], args.model, "combined", "metrics.json")
        b = os.path.join(roots[1], args.model, "combined", "metrics.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            print("[determinism] FAIL: a run produced no metrics.json")
            return 2

        identical = filecmp.cmp(a, b, shallow=False)
        print(f"[determinism] run 1 metrics.json sha256 {_sha256(a)}")
        print(f"[determinism] run 2 metrics.json sha256 {_sha256(b)}")

        # Weights too: identical metrics with different weights would mean the metric is simply
        # too coarse to see the drift.
        wa = os.path.join(roots[0], args.model, "combined", "model.pt")
        wb = os.path.join(roots[1], args.model, "combined", "model.pt")
        weights_same = os.path.exists(wa) and os.path.exists(wb) and _sha256(wa) == _sha256(wb)
        print(f"[determinism] model.pt identical: {weights_same}")

        # And the manifests must agree that the two runs were even comparable.
        with open(os.path.join(roots[0], args.model, "combined", "meta.json")) as f:
            ma = json.load(f).get("manifest") or {}
        with open(os.path.join(roots[1], args.model, "combined", "meta.json")) as f:
            mb = json.load(f).get("manifest") or {}
        problems = [p for p in compare_manifests(ma, mb) if "dirty" not in p]
        if problems:
            print("[determinism] manifests disagree: " + "; ".join(problems))

        if identical and weights_same and not problems:
            print("[determinism] PASS: byte-identical metrics.json and model.pt")
            return 0

        if not identical:
            with open(a) as f1, open(b) as f2:
                print("--- run 1 ---"); print(f1.read())
                print("--- run 2 ---"); print(f2.read())
        print("[determinism] FAIL: same seed did not produce the same run")
        return 1
    finally:
        if args.keep:
            print(f"[determinism] runs kept under {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
