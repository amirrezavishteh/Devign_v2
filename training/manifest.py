"""Run provenance: what code, what config, what data produced a given number.

Every `meta.json` carries one of these. The motivating failure is mundane and expensive: two runs
of this repo differ by 1.25 accuracy points and eight epochs of training behaviour, and with no
record of git SHA, config or split there is no way after the fact to tell whether that was seed
noise, a config edit between the runs, or a different data split. A results table whose rows cannot
be attributed to a specific state of the world is not evidence.

The split hash is the load-bearing field. Two runs are comparable ONLY if their split hashes
match: an accuracy difference between a model trained on split A and one trained on split B says
nothing about the models. `compare_manifests` exists so that check is a function call rather than
a habit.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Recorded because each one can move a number without any code change.
_TRACKED_PACKAGES = [
    "torch", "numpy", "gensim", "scipy", "scikit-learn", "xgboost",
    "tree-sitter", "tree-sitter-c", "pandas", "pyarrow",
]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", _REPO_ROOT, *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_git_state() -> dict:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


# Sampled ONCE, when this module is first imported, and reused for every manifest the process
# writes. Python does not reload modules mid-run, so a long job executes the code it started with
# no matter what the working tree does afterwards -- and sampling `git rev-parse` again per seed
# recorded whatever HEAD happened to be at write time instead.
#
# That is not hypothetical. A 3-seed sweep ran 12:50 to 20:39 while three commits landed; seeds 1
# and 2 recorded one SHA, seed 3 another, and compare_manifests duly declared them different
# experiments. Config hash, split hash, GPU and TF32 all matched, because the code really was
# identical. A warning that fires on runs which are in fact comparable is worse than no warning:
# it is how a genuinely mismatched run gets waved through later.
_GIT_STATE = _read_git_state()


def git_state() -> dict:
    """Commit SHA and dirty flag, as of the moment this process started.

    `dirty` matters as much as the SHA: a run from a modified working tree is not reproducible
    from that commit, and silently attributing it to the commit is worse than admitting it.
    """
    return dict(_GIT_STATE)


# Keys that say WHERE output goes, not WHAT is computed. Two runs that differ only in these are
# the same experiment, so hashing them would make every comparison spuriously fail -- as it did
# for scripts/check_determinism.py, which by construction points its two runs at different
# artifact roots.
_NON_SEMANTIC_KEYS = {
    ("project", "artifacts_dir"),
    ("project", "device"),          # cuda vs cpu changes speed, not the experiment's definition
    ("data", "hf_cache_dir"),
    ("data", "raw_dir"),
}


def config_hash(cfg: dict, ignore_seed: bool = False) -> str:
    """Digest of the FULLY RESOLVED config, i.e. after `extends:` merging.

    Hashing the file on disk would miss the thing that actually bit this project before: a
    machine-specific config that inherits from a parent whose values changed underneath it.

    Output paths are excluded (see `_NON_SEMANTIC_KEYS`): the question this hash answers is
    "were these two runs the same experiment", and where the files landed is not part of that.
    """
    trimmed = json.loads(json.dumps(cfg, default=str))
    for section, key in _NON_SEMANTIC_KEYS:
        if isinstance(trimmed.get(section), dict):
            trimmed[section].pop(key, None)
    if ignore_seed and isinstance(trimmed.get("project"), dict):
        trimmed["project"].pop("seed", None)
    canonical = json.dumps(trimmed, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_hash(samples) -> str:
    """Digest of the ORDERED identities of a split's members.

    Order is included deliberately: the same functions in a different order are a different
    training curriculum, and shuffling is seeded, so a changed order means something upstream
    changed too.
    """
    h = hashlib.sha256()
    for s in samples:
        name = getattr(s, "name", "") or ""
        project = getattr(s, "project", "") or ""
        label = int(getattr(s, "label", -1))
        h.update(f"{project}\x1f{name}\x1f{label}\x1e".encode("utf-8"))
    return h.hexdigest()


def library_versions() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(cfg: dict, seed: int, device: str, splits: dict | None = None) -> dict:
    """`splits` maps split name -> iterable of GraphSample (or anything with name/project/label)."""
    return {
        "git": git_state(),
        "config_sha256": config_hash(cfg),
        # The same config with the seed factored out. A seed sweep varies `seed` deliberately, so
        # comparing full hashes would flag every sweep as "different experiments" and train people
        # to ignore the warning -- which is how a genuinely mismatched run gets pooled in later.
        "config_sha256_seedless": config_hash(cfg, ignore_seed=True),
        "seed": seed,
        "device": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            # TF32 does not break run-to-run determinism, but it does change results BETWEEN
            # GPU generations, so a cross-machine comparison needs to know.
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "splits": {name: {"n": len(list(s)), "hash": split_hash(s)}
                   for name, s in (splits or {}).items()},
    }


def compare_manifests(a: dict, b: dict, allow_seed_difference: bool = False) -> list[str]:
    """Reasons two runs are NOT comparable. Empty list means they are.

    `allow_seed_difference` is for seed sweeps, where the seed is the one thing intended to vary.
    Everything else -- splits, config, commit -- must still match, so a sweep that accidentally
    changed the data or a hyperparameter is still caught.
    """
    problems = []
    for name, spec in a.get("splits", {}).items():
        other = b.get("splits", {}).get(name)
        if other is None:
            problems.append(f"split {name!r} present in one run, absent in the other")
        elif other["hash"] != spec["hash"]:
            problems.append(f"split {name!r} differs: {spec['hash'][:12]} vs {other['hash'][:12]}")
    if allow_seed_difference:
        key = "config_sha256_seedless"
        if a.get(key) and b.get(key) and a[key] != b[key]:
            problems.append("resolved config differs (beyond the seed)")
    elif a.get("config_sha256") != b.get("config_sha256"):
        problems.append("resolved config differs")
    if a.get("git", {}).get("sha") != b.get("git", {}).get("sha"):
        problems.append("git commit differs")
    if a.get("git", {}).get("dirty") or b.get("git", {}).get("dirty"):
        problems.append("at least one run came from a dirty working tree")
    return problems
