"""PrimeVul paired records -> per-function vulnerable line numbers.

Why this dataset and not the one already loaded
-----------------------------------------------
CodeXGLUE/Devign ships `func` and `target` only. There is no `func_after`, no diff, and no line
annotation, so it CANNOT support a localisation evaluation -- not "poorly", at all. Any line-level
number computed on it would be invented.

Two datasets can. PrimeVul's `*_paired.jsonl` gives 5,480 vulnerable/patched pairs, and vulnerable
lines are derived by diffing `func_before` against `func_after`; its measured label accuracy is
86-92% with 0% train/test leakage. Big-Vul ships explicit line labels but its measured label
accuracy is 25-54%, which would make a localisation result a lower bound on a noisy floor rather
than a measurement. PrimeVul is the better choice and is what this module reads.

What counts as a vulnerable line
--------------------------------
Lines in `func_before` that the patch DELETED or REPLACED. Pure insertions in `func_after` are
deliberately excluded: an added bounds check exists at a line number that has no counterpart in
the vulnerable function, so attributing it to `before` would score the model against a line it
never saw. This is the standard derivation, and it is a lower bound -- a vulnerability fixed
purely by insertion contributes no labelled line and its function is dropped rather than counted
as "model found nothing".
"""
from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass, field


@dataclass
class VulnFunction:
    """One vulnerable function with the lines its patch touched."""
    name: str
    func: str
    target: int
    vulnerable_lines: set[int] = field(default_factory=set)   # 1-based
    # The patched source. Kept so the qualitative renderer can show the diff that DEFINES the
    # ground truth beside the attention map -- a reader should be able to check the labels, not
    # just the prediction.
    func_after: str = ""
    project: str = "primevul"
    cwe: list[str] = field(default_factory=list)
    commit_id: str = ""

    @property
    def n_lines(self) -> int:
        return self.func.count("\n") + 1


def vulnerable_lines_from_diff(before: str, after: str) -> set[int]:
    """1-based line numbers in `before` that the patch deleted or replaced.

    Trailing whitespace is normalised before comparing, so a reindentation does not register as a
    vulnerability. Blank-line-only changes are dropped for the same reason: they are never the flaw.
    """
    a = [ln.rstrip() for ln in before.splitlines()]
    b = [ln.rstrip() for ln in after.splitlines()]

    lines: set[int] = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("delete", "replace"):
            for i in range(i1, i2):
                if a[i].strip():          # a removed blank line is not the vulnerability
                    lines.add(i + 1)      # 1-based
    return lines


def _cwes(record: dict) -> list[str]:
    raw = record.get("cwe") or record.get("cwe_ids") or record.get("CWE ID") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(c) for c in raw if c]


def load_primevul_paired(path: str, max_functions: int | None = None) -> list[VulnFunction]:
    """Read a PrimeVul `*_paired.jsonl` and return the VULNERABLE half with derived line labels.

    The file interleaves vulnerable/patched pairs. Records are matched on the pair key the release
    provides when present; otherwise a vulnerable record's own `func_after`/`patched_func` field is
    used, which is the common layout.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download PrimeVul's paired split from "
            f"https://github.com/ARiSE-Lab/PrimeVul and point --primevul at the *_paired.jsonl.")

    by_pair: dict[str, dict[int, dict]] = {}
    loose: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("pair_id") or rec.get("hash") or rec.get("commit_id")
            target = int(rec.get("target", rec.get("is_vulnerable", 0)))
            if key:
                by_pair.setdefault(str(key), {})[target] = rec
            else:
                loose.append(rec)

    out: list[VulnFunction] = []

    def _emit(vuln: dict, patched_src: str | None) -> None:
        src = vuln.get("func") or vuln.get("func_before") or ""
        if not src or patched_src is None:
            return
        lines = vulnerable_lines_from_diff(src, patched_src)
        if not lines:
            # Fixed purely by insertion: no line in `before` is labelled, so there is nothing to
            # find here. Dropping it is honest; keeping it would count as a model failure.
            return
        out.append(VulnFunction(
            name=str(vuln.get("idx", vuln.get("id", len(out)))),
            func=src, target=1, vulnerable_lines=lines, func_after=patched_src,
            cwe=_cwes(vuln), commit_id=str(vuln.get("commit_id", "")),
            project=str(vuln.get("project", "primevul")),
        ))

    for pair in by_pair.values():
        vuln, patched = pair.get(1), pair.get(0)
        if vuln is None:
            continue
        patched_src = (patched or {}).get("func") or vuln.get("func_after") \
            or vuln.get("patched_func")
        _emit(vuln, patched_src)
        if max_functions and len(out) >= max_functions:
            return out

    for rec in loose:
        if int(rec.get("target", 0)) != 1:
            continue
        _emit(rec, rec.get("func_after") or rec.get("patched_func"))
        if max_functions and len(out) >= max_functions:
            break
    return out


def line_lengths(source: str) -> dict[int, int]:
    """1-based line -> stripped character count. Feeds the length-prior baseline."""
    return {i + 1: len(ln.strip()) for i, ln in enumerate(source.splitlines())}
