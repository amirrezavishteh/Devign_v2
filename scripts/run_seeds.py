"""Train one configuration across several seeds and report mean +/- std.

A single run is not a measurement. Two existing runs of this repo differ by 1.25 accuracy points
and by eight epochs of training behaviour, so any comparison resting on one run per arm cannot
tell a real effect from seed noise. Every number that reaches a results table comes from here.

What this does NOT do is pool runs blindly: each run's manifest is compared against the first, and
any run whose split hash, resolved config or commit differs is reported and excluded rather than
averaged in. Averaging over runs that trained on different data is worse than not averaging.

Usage:
    python -m scripts.run_seeds --config configs/a100_codexglue.yaml --model devign --seeds 1 2 3
    python -m scripts.run_seeds --config configs/a100_codexglue.yaml --model ggrn --project qemu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from devign_data.dataset import DevignDataset
from devign_data.graph_builder import EDGE_TYPES
from scripts.train import (artifact_dir, load_test_loader, save_graph_model,
                           train_graph_model)
from training.manifest import compare_manifests
from training.metrics import majority_baseline, require_unbiased
from training.utils import load_config, resolve_device_from_config, seed_from_config

# Reported for every cell. accuracy/f1 are the paper's two columns; auc/pr_auc are threshold-free
# and are what model selection actually keys on.
REPORTED = ["accuracy", "f1", "precision", "recall", "auc", "pr_auc", "mcc"]


def _agg(values: list[float]) -> dict:
    """mean/std over seeds. std is the sample std (n-1); with <2 seeds it is undefined, not 0."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "values": values,
    }


def summarise(per_seed: list[dict]) -> dict:
    return {m: _agg([r[m] for r in per_seed if r.get(m) is not None]) for m in REPORTED}


def _fmt(agg: dict) -> str:
    if agg["mean"] is None:
        return "not measured"
    if agg["std"] is None:
        return f"{agg['mean']:.2f} (1 seed, no std)"
    return f"{agg['mean']:.2f} +/- {agg['std']:.2f}"


def run(cfg_path: str, model_name: str, seeds: list[int], project: str | None,
        epochs: int | None, out_path: str | None) -> dict:
    base_cfg = load_config(cfg_path)
    device = resolve_device_from_config(base_cfg)

    val_rows, test_rows, manifests, thresholds = [], [], [], []
    for seed in seeds:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["project"]["seed"] = seed
        # Each seed writes to its own directory, so runs cannot overwrite one another's
        # checkpoints, curves or weights.
        cfg["project"]["artifacts_dir"] = os.path.join(
            base_cfg["project"]["artifacts_dir"], f"seed{seed}")
        seed_from_config(cfg)

        print(f"\n===== {model_name}/{project or 'combined'} seed {seed} =====")
        model, metrics, train_ds = train_graph_model(
            cfg, model_name, device, epochs=epochs, project=project, verbose=True)
        save_graph_model(cfg, model, model_name, project, metrics, train_ds.type_vocab_size)

        val_rows.append(require_unbiased(metrics["best_val"], f"run_seeds val seed={seed}"))
        if metrics["test"]:
            test_rows.append(require_unbiased(metrics["test"], f"run_seeds test seed={seed}"))
        thresholds.append(metrics["threshold"])
        manifests.append(metrics["manifest"])

    # Comparability. A difference between models trained on different splits is not a result.
    incomparable = []
    for seed, m in zip(seeds[1:], manifests[1:]):
        problems = [p for p in compare_manifests(manifests[0], m, allow_seed_difference=True)
                    if "dirty" not in p]
        if problems:
            incomparable.append({"seed": seed, "problems": problems})
    if incomparable:
        print("\n[run_seeds] WARNING: runs are not mutually comparable:")
        for item in incomparable:
            print(f"  seed {item['seed']}: {'; '.join(item['problems'])}")
        print("  These are different experiments. Do not pool them into one mean.")

    # Majority-class baseline on whichever split the headline number is reported on.
    proc = base_cfg["data"]["processed_dir"]
    baseline_split = "test" if test_rows else "val"
    labels = []
    path = os.path.join(proc, f"{'test' if test_rows else 'val'}.pkl")
    if os.path.exists(path):
        ds = DevignDataset.load(path)
        labels = [s.label for s in ds.samples
                  if not project or project == "combined" or s.project == project]
    baseline = majority_baseline(labels, split=baseline_split) if labels else None

    result = {
        "model": model_name,
        "project": project or "combined",
        "config": cfg_path,
        "split_protocol": base_cfg["data"].get("split_by"),
        "seeds": seeds,
        "epochs_override": epochs,
        "threshold_per_seed": thresholds,
        "val_at_0.5": summarise(val_rows),
        "test_at_tuned": summarise(test_rows) if test_rows else None,
        "majority_baseline": baseline,
        "incomparable_runs": incomparable,
        "manifest_first_run": manifests[0] if manifests else None,
    }

    print(f"\n===== {model_name}/{project or 'combined'} "
          f"({base_cfg['data'].get('split_by')} split, {len(seeds)} seeds) =====")
    print(f"{'metric':<12} {'val@0.5':<24} {'test@tuned':<24} majority")
    for m in REPORTED:
        v = _fmt(result["val_at_0.5"][m])
        t = _fmt(result["test_at_tuned"][m]) if result["test_at_tuned"] else "no test split"
        b = f"{baseline[m]:.2f}" if baseline and m in baseline else "-"
        print(f"{m:<12} {v:<24} {t:<24} {b}")
    if baseline:
        print(f"(majority class predicts {baseline['predicts']} on the {baseline_split} split)")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[run_seeds] wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["devign", "ggrn", "mil"], default="devign")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="at least 3; a single run is not a measurement")
    ap.add_argument("--project", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    if len(args.seeds) < 3:
        print(f"[run_seeds] NOTE: {len(args.seeds)} seed(s) given. Fewer than 3 cannot support a "
              f"std, and a difference under ~1 point will not be distinguishable from noise.")

    out = args.out or os.path.join(
        "artifacts", "seeds",
        f"{args.model}_{args.project or 'combined'}_{os.path.basename(args.config)}.json")
    run(args.config, args.model, args.seeds, args.project, args.epochs, out)


if __name__ == "__main__":
    main()
