"""Quantify how much a random split leaks through shared fix commits.

The leakage check trains a second model on a commit-disjoint split and compares scores. This
measures the *mechanism* directly and costs seconds rather than hours: how many evaluation
functions come from a commit that also appears in training, and how many near-duplicate function
bodies straddle the split.

Why both. A shared commit is the channel the paper's random split leaves open -- one
vulnerability-fixing commit usually touches several functions, so a random assignment scatters
siblings across train and test and the model can recognise the commit rather than the flaw. Exact
duplicates are already removed at load time (`_dedupe_functions`), but near-duplicates are not, and
they leak the same way while looking like distinct functions.

Usage:
    python -m scripts.measure_commit_overlap --config configs/a100_codexglue.yaml
    python -m scripts.measure_commit_overlap --config configs/a100_codexglue.yaml --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter

from devign_data.download import acquire_dataset
from devign_data.prepare import split_functions
from training.utils import load_config, seed_from_config

_WS = re.compile(r"\s+")


def normalised_body_hash(source: str) -> str:
    """Hash of the function with all whitespace collapsed.

    Catches the near-duplicates exact-match dedupe misses: the same helper reindented, or rewrapped
    by a formatter, is the same function for leakage purposes.
    """
    return hashlib.sha1(_WS.sub(" ", source).strip().encode("utf-8")).hexdigest()


def analyse(train, evaluation, label: str) -> dict:
    train_commits = {f.commit_id for f in train if f.commit_id}
    train_bodies = {normalised_body_hash(f.func) for f in train}

    shared = [f for f in evaluation if f.commit_id and f.commit_id in train_commits]
    dup_bodies = [f for f in evaluation if normalised_body_hash(f.func) in train_bodies]

    n = max(1, len(evaluation))
    shared_pos = sum(1 for f in shared if int(f.target) == 1)
    return {
        "split": label,
        "n_eval": len(evaluation),
        "n_train": len(train),
        "eval_from_a_training_commit": len(shared),
        "eval_from_a_training_commit_pct": round(100.0 * len(shared) / n, 2),
        "of_those_vulnerable": shared_pos,
        "eval_body_duplicated_in_train": len(dup_bodies),
        "eval_body_duplicated_in_train_pct": round(100.0 * len(dup_bodies) / n, 2),
        "distinct_train_commits": len(train_commits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = seed_from_config(cfg)
    functions = acquire_dataset(cfg)
    print(f"[overlap] {len(functions)} functions, "
          f"{len({f.commit_id for f in functions if f.commit_id})} distinct commits")

    sizes = Counter(Counter(f.commit_id for f in functions if f.commit_id).values())
    multi = sum(c for n, c in sizes.items() if n > 1)
    print(f"[overlap] commits touching >1 function: {multi} "
          f"(these are the ones a random split can scatter)")

    results = []
    for split_by in ("codexglue", "random", "commit"):
        cfg_i = json.loads(json.dumps(cfg))
        cfg_i["data"]["split_by"] = split_by
        try:
            train, val, test = split_functions(functions, cfg_i["data"], seed)
        except Exception as exc:
            print(f"[overlap] {split_by}: unavailable ({exc})")
            continue
        results.append(analyse(train, test or val, split_by))

    print()
    print("%-12s%9s%9s%14s%16s" % ("split", "n_train", "n_eval", "eval from a",
                                   "eval body dup"))
    print("%-12s%9s%9s%14s%16s" % ("", "", "", "train commit", "in train"))
    print("-" * 62)
    for r in results:
        print("%-12s%9d%9d%9d (%3.0f%%)%9d (%3.0f%%)" % (
            r["split"], r["n_train"], r["n_eval"],
            r["eval_from_a_training_commit"], r["eval_from_a_training_commit_pct"],
            r["eval_body_duplicated_in_train"], r["eval_body_duplicated_in_train_pct"]))
    print()
    print("`commit` must show 0% by construction -- a non-zero value there is a bug in the split,")
    print("not a property of the data. The other rows are the channel it closes.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"seed": seed, "splits": results}, fh, indent=2)
        print(f"[overlap] wrote {args.json}")


if __name__ == "__main__":
    main()
