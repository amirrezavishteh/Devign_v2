"""Shared helpers: config loading, seeding, device resolution."""
from __future__ import annotations

import os
import random

# Set BEFORE torch is imported. cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it initialises its
# handle, and without a fixed workspace size several reduction kernels pick their split based on
# available scratch memory -- which makes the same matmul give bitwise-different results depending
# on what else the GPU was doing. torch.use_deterministic_algorithms() raises unless this is set.
# `setdefault` so an operator can still override it from the environment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402  (must follow the env var above)
import torch  # noqa: E402
import yaml  # noqa: E402


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_config_path(path: str, relative_to: str | None = None) -> str:
    """Resolve a config path against its referrer's directory, then the repo root, then cwd."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidates = []
    if relative_to:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(relative_to)), path))
    candidates.append(os.path.join(_REPO_ROOT, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) is replaced wholesale, so an
    override can redefine e.g. `data.projects` without inheriting stale entries.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str = "config.yaml", _seen: tuple = ()) -> dict:
    """Load a config, applying `extends:` inheritance.

    A machine-specific config should contain ONLY the keys that differ, e.g.

        extends: config.yaml
        dataset:
          max_nodes_per_batch: 24000

    and is deep-merged over its parent. This exists because full-copy configs silently go stale:
    an earlier `config_a100.yaml` was a frozen copy of `config.yaml`, so later fixes to dropout,
    word2vec min_count and early-stopping patience never reached the GPU and two full training
    runs were wasted on pre-fix hyperparameters.
    """
    # A bare "config.yaml" resolves relative to the repo root, not the process's cwd, so tests
    # and scripts behave the same whether invoked from the repo root or elsewhere.
    path = _resolve_config_path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_ref = cfg.pop("extends", None)
    if not parent_ref:
        return cfg

    real = os.path.abspath(path)
    if real in _seen:
        chain = " -> ".join(_seen + (real,))
        raise ValueError(f"circular config `extends` chain: {chain}")

    parent_path = _resolve_config_path(str(parent_ref), relative_to=path)
    parent = load_config(parent_path, _seen=_seen + (real,))
    return deep_merge(parent, cfg)


def set_seed(seed: int, deterministic: bool = True, allow_tf32: bool = False) -> None:
    """Seed every RNG the pipeline draws from, and pin the backends that would otherwise vary.

    Seeding alone does NOT make a CUDA run reproducible. Three further sources of drift have to be
    closed, or runs diverge and no result is comparable to another:

      * cuDNN's autotuner (`benchmark`) picks a convolution algorithm by timing candidates at the
        first call, so the algorithm -- and therefore the floating-point rounding -- depends on
        transient machine load. `benchmark=False` + `deterministic=True` pins it.
      * Scatter-style reductions built on atomic float adds accumulate in completion order, and
        float addition is not associative. `use_deterministic_algorithms` makes torch either pick
        a deterministic kernel or complain. See `models/ggnn.py::_propagate_sparse`, which was
        rewritten to a sorted segment reduction for exactly this reason.
      * TF32. This one does NOT affect run-to-run determinism on one machine, and it is the one
        that bites when development and training happen on different GPUs -- here, an RTX 4060
        laptop and an A100 server. TF32 keeps 10 mantissa bits instead of 23, and torch enables it
        for cuDNN convolutions by default on every Ampere-or-newer card. The Conv module is built
        from Conv1d/Conv2d, so leaving it at the default silently gives the A100 and the laptop
        different arithmetic for the same code and seed. Pinned OFF by default so the two agree;
        set `project.allow_tf32: true` to trade that for A100 throughput, and note that runs on
        either side of the switch are not comparable.

    `warn_only=True` keeps a run alive if some op has no deterministic implementation: the warning
    names it, rather than a multi-hour job dying at hour three. Pass `deterministic=False` for the
    `--fast` path, which trades reproducibility for throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def seed_from_config(cfg: dict, seed: int | None = None) -> int:
    """Apply `project.*` reproducibility settings. Returns the seed actually used."""
    project = cfg.get("project", {})
    seed = project["seed"] if seed is None else seed
    set_seed(seed,
             deterministic=project.get("deterministic", True),
             allow_tf32=project.get("allow_tf32", False))
    return seed


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for DataLoader: give each worker a seed derived from torch's own.

    Without this, every worker process inherits the parent's numpy/random state, so any dataset
    that draws a random number produces the SAME stream in each worker -- and which worker serves
    which sample depends on OS scheduling. Harmless at num_workers=0 (today's default), and the
    thing that silently breaks reproducibility the moment someone raises it.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int) -> torch.Generator:
    """An explicit RNG for DataLoader shuffling, so it does not consume the global torch stream."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
