"""Static-analyzer baselines for Table 3 (Cppcheck, Flawfinder).

If the real tools are installed and on PATH, we invoke them on each function (written to a temp
.c file) and flag the function vulnerable if the analyzer reports any warning. If the tools are
unavailable, we fall back to a transparent keyword/pattern heuristic that mimics the kinds of
findings these analyzers emit (e.g. strcpy/gets/sprintf for Flawfinder), so Table 3 still renders
with clearly-labeled approximate numbers. The fallback is marked in the returned `mode` field.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

# Flawfinder-style dangerous functions (subset of its built-in ruleset).
_FLAWFINDER_PATTERNS = [
    r"\bstrcpy\b", r"\bstrcat\b", r"\bsprintf\b", r"\bgets\b", r"\bscanf\b",
    r"\bsystem\b", r"\bexec[lv]p?\b", r"\bmemcpy\b", r"\bstrncpy\b", r"\bvsprintf\b",
    r"\bprintf\s*\(\s*[A-Za-z_]\w*\s*\)",  # printf(non-literal) format-string
]

# Cppcheck tends to catch memory errors / obvious UB.
_CPPCHECK_PATTERNS = [
    r"\bfree\s*\(\s*(\w+)\s*\).*?\b\1\b",   # use-after-free-ish (free then reuse)
    r"\bmalloc\b(?![\s\S]*\bif\b[\s\S]*NULL)",  # malloc without NULL check nearby
    r"\[\s*\w+\s*<=\s*\w+\s*\]",            # off-by-one style (rough)
]


def _resolve_tool(name: str, explicit_path: str | None) -> str | None:
    """Explicit config path first, then PATH lookup. None means genuinely unavailable."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    return shutil.which(name)


def _run_tool(cmd: list[str], source: str, suffix=".c") -> str:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tf:
        tf.write(source)
        path = tf.name
    try:
        full = cmd + [path]
        res = subprocess.run(full, capture_output=True, text=True, timeout=30)
        return (res.stdout or "") + (res.stderr or "")
    except Exception:
        return ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _cppcheck_predict(source: str, exe: str | None) -> int:
    if exe:
        out = _run_tool([exe, "--enable=warning,style", "--quiet", "--template={severity}"],
                        source)
        return 1 if re.search(r"error|warning", out, re.IGNORECASE) else 0
    flat = re.sub(r"\s+", " ", source)
    return 1 if any(re.search(p, flat) for p in _CPPCHECK_PATTERNS) else 0


def _flawfinder_predict(source: str, exe: str | None) -> int:
    if exe:
        out = _run_tool([exe, "--quiet", "--dataonly"], source)
        return 1 if re.search(r"\[\d+\]", out) else 0
    return 1 if any(re.search(p, source) for p in _FLAWFINDER_PATTERNS) else 0


def run_static_analyzer(name: str, functions,
                        tool_path: str | None = None) -> tuple[list[int], list[int], list[str], str]:
    """Returns (y_true, y_pred, projects, mode) for the given analyzer over a list of RawFunction.

    `mode` is "tool" only when the real binary was actually invoked; otherwise "heuristic-fallback"
    (renders as `<name> (heuristic-fallback)` / `regex-mimic (NOT <name>)` in report tables --
    it is never presented as evidence about the real analyzer).
    """
    exe = _resolve_tool(name, tool_path)
    mode = "tool" if exe else "heuristic-fallback"
    predict = _flawfinder_predict if name == "flawfinder" else _cppcheck_predict
    y_true, y_pred, projects = [], [], []
    for fn in functions:
        y_true.append(int(fn.target))
        y_pred.append(predict(fn.func, exe))
        projects.append(fn.project)
    return y_true, y_pred, projects, mode
