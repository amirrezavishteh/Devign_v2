"""Dataset acquisition / synthesis for the 4 Devign projects.

The original Devign datasets were manually labeled (~600 man-hours) and only two of the four
projects (QEMU, FFmpeg -> the "function" CodeXGLUE/Devign release) were ever made partly public;
there is no canonical, stable download for all four with per-function labels. To keep this repo
runnable out-of-the-box we support two modes:

  1. Synthetic mode (default): generate paired vulnerable/safe C functions from CWE templates
     (data/templates.py), with per-project counts and vulnerable ratios chosen to mirror Table 1.

  2. Real mode: load an existing json/jsonl/csv with at least {func, target, project} columns.
     The widely-used CodeXGLUE "Devign" release (Zhou et al.) uses exactly `func`/`target`, so a
     downloaded `function.json` works directly. Set data.real_data_path in config.yaml.
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

from data.templates import TEMPLATES


@dataclass
class RawFunction:
    func: str
    target: int          # 1 vulnerable, 0 non-vulnerable
    project: str
    name: str = ""
    cwe: str = ""
    # Commit the function was extracted from. Carried through so the split can be made
    # commit-disjoint (leakage check) and so the Q5 holdout can require unseen commits.
    commit_id: str = ""


# ----------------------------------------------------------------------------------------------
# Synthetic generation
# ----------------------------------------------------------------------------------------------

def generate_synthetic(projects: list[str], samples_per_project: int,
                       vuln_ratio: dict[str, float], seed: int = 42) -> list[RawFunction]:
    rng = random.Random(seed)
    out: list[RawFunction] = []
    for project in projects:
        ratio = vuln_ratio.get(project, 0.5)
        n_vuln = int(round(samples_per_project * ratio))
        n_safe = samples_per_project - n_vuln
        # Generate vulnerable functions.
        for _ in range(n_vuln):
            gen = rng.choice(TEMPLATES)
            vuln_src, _safe_src, name = gen(rng)
            out.append(RawFunction(func=vuln_src.strip(), target=1, project=project,
                                   name=name, cwe=gen.__name__.replace("gen_", "")))
        # Generate non-vulnerable (fixed) functions.
        for _ in range(n_safe):
            gen = rng.choice(TEMPLATES)
            _vuln_src, safe_src, name = gen(rng)
            out.append(RawFunction(func=safe_src.strip(), target=0, project=project,
                                   name=name, cwe=gen.__name__.replace("gen_", "")))
    rng.shuffle(out)
    return out


def generate_cve_set(projects: list[str], per_project: int, seed: int = 1234) -> list[RawFunction]:
    """Q5: a held-out set emulating 'latest CVEs' -- vulnerable functions only, unseen seed."""
    rng = random.Random(seed)
    out: list[RawFunction] = []
    for project in projects:
        for i in range(per_project):
            gen = rng.choice(TEMPLATES)
            vuln_src, _safe, name = gen(rng)
            out.append(RawFunction(func=vuln_src.strip(), target=1, project=project,
                                   name=f"CVE_{project}_{i}", cwe=gen.__name__.replace("gen_", "")))
    return out


# ----------------------------------------------------------------------------------------------
# Real data loading
# ----------------------------------------------------------------------------------------------

def _load_records(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported real_data_path extension: {ext}")


def load_real(path: str) -> list[RawFunction]:
    records = _load_records(path)
    out: list[RawFunction] = []
    for r in records:
        func = r.get("func") or r.get("function") or r.get("code")
        if func is None:
            continue
        target = int(r.get("target", r.get("label", 0)))
        project = r.get("project") or r.get("repo") or "unknown"
        out.append(RawFunction(func=func, target=target, project=str(project),
                               name=str(r.get("name", "")), cwe=str(r.get("cwe", "")),
                               commit_id=str(r.get("commit_id", ""))))
    return out


# ----------------------------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------------------------

def save_raw(functions: list[RawFunction], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for fn in functions:
            f.write(json.dumps(asdict(fn)) + "\n")


def load_raw(path: str) -> list[RawFunction]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(RawFunction(**json.loads(line)))
    return out


def _from_devign_release(data_cfg: dict) -> list[RawFunction]:
    """The paper's own released data (FFmpeg + QEMU). See data/hf_devign.py."""
    from data.hf_devign import fetch_devign_release

    records = fetch_devign_release(cache_dir=data_cfg.get("hf_cache_dir"))
    functions = [RawFunction(**r) for r in records]
    keep = set(data_cfg.get("projects") or [])
    if keep:
        functions = [f for f in functions if f.project in keep]
    return functions


def acquire_dataset(config: dict, out_path: Optional[str] = None) -> list[RawFunction]:
    """Load raw functions according to `data.source`.

    devign_release -- download the paper's released FFmpeg+QEMU data (default)
    file           -- read data.real_data_path (json/jsonl/csv with func/target/project)
    synthetic      -- CWE templates; kept only so the pipeline runs offline as a smoke test.
                      Numbers produced from it are NOT vulnerability-detection results.
    """
    data_cfg = config["data"]
    source = data_cfg.get("source")
    if source is None:
        # Back-compat with the pre-`source` config shape.
        source = "file" if data_cfg.get("real_data_path") else "synthetic"

    if source == "devign_release":
        functions = _from_devign_release(data_cfg)
    elif source == "file":
        real_path = data_cfg.get("real_data_path")
        if not real_path:
            raise ValueError("data.source='file' requires data.real_data_path to be set")
        functions = load_real(real_path)
    elif source == "synthetic":
        syn = data_cfg["synthetic"]
        functions = generate_synthetic(
            projects=data_cfg["projects"],
            samples_per_project=syn["samples_per_project"],
            vuln_ratio=syn["vuln_ratio"],
            seed=config["project"]["seed"],
        )
    else:
        raise ValueError(f"Unknown data.source: {source!r} "
                         "(expected one of: devign_release, file, synthetic)")

    if out_path:
        save_raw(functions, out_path)
    return functions
