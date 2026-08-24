"""Fetch the Devign authors' released dataset from the HuggingFace hub.

This is the real data behind the paper, not a re-labelling: the CodeXGLUE "defect detection"
release *is* Zhou et al.'s `function.json`, carrying `project` ('FFmpeg' | 'qemu'), `commit_id`
and the manual `target` label. Only 2 of the paper's 4 projects were ever published -- Linux
Kernel and Wireshark (the paper's two strongest datasets, including its best result of F1 84.97)
were never released, so no faithful reproduction can produce those columns.

We download all three published splits and concatenate them, because the paper does its own
random 75/25 split rather than using CodeXGLUE's; re-splitting here keeps Sec 3.3 faithful.

Requires only `huggingface_hub` + `pyarrow`, both of which the repo already depends on
transitively -- no `datasets` install needed.
"""
from __future__ import annotations

import os

_REPO_ID = "google/code_x_glue_cc_defect_detection"
# split name -> path within the HF repo. Same three files you get by downloading them by hand;
# data.download.load_devign_parquet_dir reads exactly these from a local directory.
_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# The release spells the projects 'FFmpeg' and 'qemu'; the repo uses lowercase keys throughout.
_PROJECT_NORMALISE = {"ffmpeg": "ffmpeg", "qemu": "qemu"}


def normalise_project(raw: str) -> str:
    return _PROJECT_NORMALISE.get(str(raw).strip().lower(), str(raw).strip().lower())


def fetch_devign_release(cache_dir: str | None = None) -> list[dict]:
    """Download + parse the released Devign dataset. Returns raw-schema dicts.

    Each dict is {func, target, project, name, cwe, commit_id}, matching `RawFunction`'s fields
    so it can be handed straight to `load_real`-style construction. `cwe` is always "" -- the
    Devign release carries no CWE annotation (unlike Big-Vul).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared in requirements
        raise RuntimeError(
            "huggingface_hub is required to fetch the Devign release; "
            "pip install huggingface_hub, or set data.source=file with a local copy."
        ) from exc
    import pyarrow.parquet as pq

    records: list[dict] = []
    for split, rel in _FILES.items():
        path = hf_hub_download(repo_id=_REPO_ID, repo_type="dataset", filename=rel,
                               cache_dir=cache_dir)
        table = pq.read_table(path)
        cols = {name: table.column(name).to_pylist() for name in table.schema.names}
        n = table.num_rows
        for i in range(n):
            func = cols["func"][i]
            if not func or not func.strip():
                continue
            records.append({
                "func": func,
                # `target` is a bool in the parquet schema; the pipeline wants 0/1.
                "target": int(bool(cols["target"][i])),
                "project": normalise_project(cols["project"][i]),
                "name": str(cols.get("id", [""] * n)[i]),
                "cwe": "",
                "commit_id": str(cols.get("commit_id", [""] * n)[i]),
                # Preserved so `data.split_by: codexglue` can reproduce the published split.
                # Discarded by the default random split, which is what Sec 3.3 does.
                "split": split,
            })
    # Deduping is data.download._dedupe_functions' job, so the local-parquet and HF paths cannot
    # drift apart on which duplicate survives.
    return records


def summarise(records: list[dict]) -> str:
    """Human-readable composition, mirroring the paper's Table 1 layout."""
    by_project: dict[str, list[int]] = {}
    commits: dict[str, set] = {}
    for r in records:
        p = r["project"]
        by_project.setdefault(p, [0, 0])[r["target"]] += 1
        commits.setdefault(p, set()).add(r.get("commit_id", ""))
    lines = [f"{'Project':<12}{'Graphs':>9}{'Vul':>9}{'Non-Vul':>9}{'Commits':>9}"]
    tot = [0, 0, 0]
    for p in sorted(by_project):
        safe, vuln = by_project[p]
        lines.append(f"{p:<12}{safe + vuln:>9}{vuln:>9}{safe:>9}{len(commits[p]):>9}")
        tot[0] += safe + vuln
        tot[1] += vuln
        tot[2] += safe
    lines.append(f"{'Total':<12}{tot[0]:>9}{tot[1]:>9}{tot[2]:>9}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    recs = fetch_devign_release(cache_dir=os.environ.get("HF_HOME"))
    print(summarise(recs))
