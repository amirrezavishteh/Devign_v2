"""Locate (or explain how to obtain) the PrimeVul paired split used for localisation ground truth.

This deliberately does NOT scrape a download. PrimeVul is distributed through the authors' own
release channels, those URLs move, and a script that silently fetches "some file called
primevul_test_paired.jsonl" from wherever is exactly how a reproduction ends up evaluated against
data nobody can identify afterwards. So: it searches the usual locations, validates whatever it
finds, and otherwise prints precise instructions.

Validation matters as much as location. A paired file that yields no derivable line labels is
worse than a missing one -- the evaluation would run, report on a handful of functions, and look
fine.

Usage:
    python -m scripts.fetch_primevul --check
    python -m scripts.fetch_primevul --check --path data/primevul/primevul_test_paired.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

SEARCH_DIRS = ("data/primevul", "data", ".")
NAME_HINTS = ("paired",)

INSTRUCTIONS = """
PrimeVul paired split not found.

It is the ground truth for line-level localisation. CodeXGLUE/Devign cannot substitute: it ships
`func` and `target` only -- no `func_after`, no diff, no line annotation -- so a localisation
number computed on it would be invented.

To obtain it:

  1. Go to https://github.com/ARiSE-Lab/PrimeVul and follow the dataset link in the README
     (the authors host the splits off-repo; the link is kept current there).
  2. Download the PAIRED split -- files named like `primevul_{train,valid,test}_paired.jsonl`.
     The unpaired split has no patched counterpart and cannot yield line labels.
  3. Put it here:

         mkdir -p data/primevul
         mv primevul_test_paired.jsonl data/primevul/

  4. Re-run this check:

         python -m scripts.fetch_primevul --check

Why PrimeVul and not Big-Vul: Big-Vul ships explicit line labels but its measured label accuracy
is 25-54%, which would put a localisation result on a noisy floor. PrimeVul measures 86-92% with
0% train/test leakage, and its vulnerable lines are derived by diffing func_before against
func_after (see devign_data/primevul.py).
""".strip()


def find_paired_file(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for directory in SEARCH_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            low = name.lower()
            if low.endswith(".jsonl") and any(h in low for h in NAME_HINTS):
                return os.path.join(directory, name)
    return None


def check(path: str) -> int:
    from devign_data.primevul import load_primevul_paired

    print(f"[primevul] reading {path}")
    functions = load_primevul_paired(path)
    if not functions:
        print("[primevul] FAIL: the file parsed but yielded 0 functions with derivable line "
              "labels.\n"
              "  Most likely this is the UNPAIRED split, which has no patched counterpart to "
              "diff against.\n"
              "  Check for a filename containing `paired`.")
        return 2

    n_lines = sum(len(f.vulnerable_lines) for f in functions)
    cwes: dict[str, int] = {}
    for f in functions:
        for c in (f.cwe or ["unknown"]):
            cwes[c] = cwes.get(c, 0) + 1
    total_lines = sum(f.n_lines for f in functions)

    print(f"[primevul] OK: {len(functions)} vulnerable functions carrying line labels")
    print(f"[primevul]     {n_lines} labelled lines across {total_lines} total "
          f"({100.0 * n_lines / max(1, total_lines):.2f}% of lines)")
    print(f"[primevul]     median labelled lines per function: "
          f"{sorted(len(f.vulnerable_lines) for f in functions)[len(functions) // 2]}")
    top = sorted(cwes.items(), key=lambda kv: -kv[1])[:8]
    print("[primevul]     top CWEs: " + ", ".join(f"{c} ({n})" for c, n in top))
    # Fewer than 5 functions per CWE is an anecdote, not a per-CWE result; say so now rather than
    # discovering it when the per-CWE table turns out to be one row.
    usable = [c for c, n in cwes.items() if n >= 5]
    print(f"[primevul]     CWEs with >= 5 functions (eligible for a per-CWE row): {len(usable)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="explicit path to a *_paired.jsonl")
    ap.add_argument("--check", action="store_true", help="validate the file that is found")
    args = ap.parse_args()

    path = find_paired_file(args.path)
    if not path:
        print(INSTRUCTIONS)
        return 1
    print(f"[primevul] found {path}")
    return check(path) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
