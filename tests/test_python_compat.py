"""Guard against syntax that only parses on a newer Python than the project supports.

Development happens on Python 3.12; the A100 server that produces every reported number runs
3.10.12. `pyproject.toml` declares `requires-python = ">=3.10"`, and nothing enforced it.

This is not hypothetical. A one-line edit introduced

    return f"{agg["mean"]:.2f} +/- {agg["std"]:.2f}"

which is legal under PEP 701 on 3.12 and a hard SyntaxError on 3.10. The local suite passed
completely green, and the failure would have appeared only when the script ran on the server --
after a queued job had been waiting hours for its turn.

`compile()` cannot catch this, because the interpreter running the test is the newer one. So the
check is textual and deliberately narrow: it looks for the one 3.12-only construct likely to be
written by accident. Broader style checks belong in a linter, not here.
"""
from __future__ import annotations

import ast
import glob
import os
import re

# The floor declared in pyproject.toml.
MIN_PYTHON = (3, 10)

SKIP_DIRS = ("__pycache__", ".venv", "venv", ".git", "build", "dist", "Devign_Seminar")

# An f-string opened with " that contains a subscript also quoted with " -- PEP 701, 3.12+.
_NESTED_DQ = re.compile(r'f"[^"]*\{[^}]*\["')
_NESTED_SQ = re.compile(r"f'[^']*\{[^}]*\['")


def _sources() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if any(part in path for part in SKIP_DIRS):
            continue
        out.append(path)
    return out


def test_there_are_sources_to_check():
    """A silent zero-file glob would make every other test here vacuously pass."""
    assert len(_sources()) > 20


def test_no_fstring_nested_same_quote_subscripts():
    """PEP 701 syntax: legal on 3.12, SyntaxError on 3.10 and 3.11."""
    offenders = []
    for path in _sources():
        # This file necessarily quotes the offending construct in its own docstring as the
        # worked example, so scanning it would make the guard permanently red.
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if _NESTED_DQ.search(line) or _NESTED_SQ.search(line):
                    offenders.append(f"{os.path.basename(path)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"f-strings reusing their own quote character inside a subscript require Python 3.12, "
        f"but this project supports {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and trains on 3.10:\n  "
        + "\n  ".join(offenders))


def test_every_source_file_parses():
    """Cheap catch-all: a file that does not parse at all can never run anywhere."""
    broken = []
    for path in _sources():
        with open(path, encoding="utf-8") as fh:
            try:
                ast.parse(fh.read(), filename=path)
            except SyntaxError as exc:
                broken.append(f"{os.path.basename(path)}:{exc.lineno}: {exc.msg}")
    assert not broken, "files failed to parse:\n  " + "\n  ".join(broken)


def test_declared_python_floor_matches_this_guard():
    """If pyproject raises its floor to 3.12, this whole guard becomes dead weight."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', text)
    assert m, "could not find requires-python in pyproject.toml"
    assert (int(m.group(1)), int(m.group(2))) == MIN_PYTHON, (
        f"pyproject declares >={m.group(1)}.{m.group(2)} but this guard assumes "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}; update MIN_PYTHON and re-check whether the 3.12-only "
        f"syntax check is still needed.")
