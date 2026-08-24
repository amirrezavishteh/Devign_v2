"""Project node attention onto source lines, and score the result against ground truth.

Attention is a distribution over AST nodes. Every node carries a source span, so a per-line score
is a projection of that distribution rather than a second model -- which is the whole reason this
comes for free with an attention readout and does not with Eq. 9.

Two aggregation choices, and the choice matters enough that it must never be implicit:

``max``  (default)
    score(line) = max attention over nodes whose span covers that line. Answers "is there a node
    here the model cared about". Insensitive to how finely the parser subdivides a line, which is
    the property that matters: a line parsed into 20 nodes is not more suspicious than the same
    line parsed into 3.

``sum``
    score(line) = total attention mass landing on that line. Answers "how much of the model's
    total attention fell here", and is therefore biased toward syntactically dense lines -- a long
    boolean condition accumulates mass simply by having more nodes.

Nodes span multiple lines (a `for` statement covers its whole body), and a node contributes its
score to every line it covers. Large structural nodes therefore smear attention across a block.
`max` degrades gracefully under that; `sum` multiplies it. Reported either way, never silently.
"""
from __future__ import annotations

import numpy as np

AGGREGATIONS = ("max", "sum")


class LocalizationUnavailable(RuntimeError):
    """Raised when a sample carries no line spans, rather than scoring line 0 and looking fine."""


def attention_to_lines(attn: np.ndarray, node_lines: np.ndarray | None,
                       num_nodes: int, aggregation: str = "max") -> dict[int, float]:
    """attn [num_nodes] (or [heads, num_nodes]) -> {line: score}, 1-based lines.

    Multi-head attention is averaged over heads first: the heads are separate distributions over
    the same nodes, and a per-line score has to be one number.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if node_lines is None:
        raise LocalizationUnavailable(
            "this sample has no per-node line spans (processed data predates format v3). "
            "Re-run `python -m scripts.prepare_data` to rebuild it.")

    a = np.asarray(attn, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=0)
    a = a[:num_nodes]
    spans = np.asarray(node_lines)[:num_nodes]

    scores: dict[int, float] = {}
    for weight, (lo, hi) in zip(a, spans):
        for line in range(int(lo), int(hi) + 1):
            if aggregation == "max":
                scores[line] = max(scores.get(line, 0.0), float(weight))
            else:
                scores[line] = scores.get(line, 0.0) + float(weight)
    return scores


def rank_lines(scores: dict[int, float]) -> list[int]:
    """Lines best-first. Ties break by line number so the ranking is deterministic."""
    return [line for line, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def localization_metrics(ranked: list[int], vulnerable: set[int]) -> dict[str, float] | None:
    """Standard line-level metrics for ONE function. None if there is nothing to find.

    `ranked` is the model's ordering, `vulnerable` the ground-truth line numbers.

    top_k            was any true line in the first k
    mrr              reciprocal rank of the FIRST true line
    normalised_rank  that rank as a fraction of the function's length (0 = first, 1 = last), so
                     functions of different sizes are comparable
    ifa              Initial False Alarms: clean lines a reviewer reads before the first true one.
                     The metric that matters operationally -- an IFA of 40 means the tool is not
                     usable however good its MRR looks.
    """
    hits = [i for i, line in enumerate(ranked) if line in vulnerable]
    if not ranked or not vulnerable or not hits:
        return None
    first = hits[0]
    n = len(ranked)
    return {
        "top_1": float(first < 1),
        "top_5": float(first < 5),
        "top_10": float(first < 10),
        "mrr": 1.0 / (first + 1),
        "normalised_rank": first / max(1, n - 1),
        "ifa": float(first),
        "n_lines": n,
        "n_vulnerable": len(vulnerable),
    }


def aggregate(per_function: list[dict]) -> dict[str, float]:
    """Mean over functions, plus the count the means rest on.

    The count is not decoration. These metrics are only defined on functions the model called
    vulnerable AND that have ground-truth lines, so a mean over 11 functions and a mean over 400
    are different claims and must not print identically.
    """
    if not per_function:
        return {"n_functions": 0}
    keys = ("top_1", "top_5", "top_10", "mrr", "normalised_rank", "ifa")
    out = {k: float(np.mean([f[k] for f in per_function])) for k in keys}
    out["ifa_median"] = float(np.median([f["ifa"] for f in per_function]))
    out["n_functions"] = len(per_function)
    return out


def random_baseline(n_lines: int, vulnerable: set[int], rng: np.random.Generator,
                    trials: int = 20) -> dict[str, float] | None:
    """Expected metrics from ranking lines at random. The floor every method must clear."""
    lines = list(range(1, n_lines + 1))
    runs = []
    for _ in range(trials):
        rng.shuffle(lines)
        m = localization_metrics(list(lines), vulnerable)
        if m:
            runs.append(m)
    return aggregate(runs) if runs else None


def length_prior_baseline(line_lengths: dict[int, int], vulnerable: set[int]) -> dict | None:
    """Rank lines by raw length, longest first. A trivial statistical prior.

    This row is not optional in the write-up. Long lines are disproportionately complex
    expressions, so length alone is a real signal -- if attention cannot beat it, the attention has
    learned nothing about vulnerability and any localisation claim collapses.
    """
    ranked = [line for line, _ in sorted(line_lengths.items(), key=lambda kv: (-kv[1], kv[0]))]
    return localization_metrics(ranked, vulnerable)
