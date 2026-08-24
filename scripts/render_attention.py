"""Phase 3.4: render attention as a heat overlay beside the actual fix diff.

Produces a self-contained HTML page: each selected function shown twice, once with per-line
attention shading and once as the before/after diff that defines the ground truth. A reader can
then judge for themselves whether the model looked where the patch did.

Selection is deliberate, not cherry-picked:

  * only TRUE POSITIVES are eligible -- a heat map over a function the model called clean shows
    nothing about localisation;
  * examples are spread across distinct CWEs so the page is not five instances of one pattern;
  * and at least one FAILURE is always included, chosen as the true positive whose top-ranked line
    is furthest from any ground-truth line. A qualitative section containing only successes is not
    evidence, so `--successes 4 --failures 1` is the default rather than something to remember.

The failure case is labelled with what the attention pointed at instead, because "it was wrong" is
much less useful than "it fired on the allocation three lines above the overflow".
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import os

import numpy as np
import torch

from devign_data.graph_builder import EDGE_TYPES
from devign_data.primevul import load_primevul_paired
from devign_data.word2vec_embed import NodeFeaturizer
from evaluation.localize import attention_to_lines, localization_metrics, rank_lines
from models.devign import build_model
from scripts.run_localization import _sample_from_source, conv_saliency
from training.utils import load_config, resolve_device_from_config, seed_from_config

_CSS = """
body { font: 13px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
       margin: 0; padding: 24px; background: #fbfbfd; color: #1a1a1a; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 6px; }
.sub { color: #666; font-size: 12px; margin-bottom: 20px; }
.case { border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px;
        margin-bottom: 22px; background: #fff; }
.case.fail { border-color: #d98b8b; background: #fffafa; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
         margin-right: 6px; }
.hit { background: #d8f0d8; color: #1c5c1c; }
.miss { background: #f5d5d5; color: #7d1f1f; }
.meta { color: #555; font-size: 12px; margin: 6px 0 10px; }
table { border-collapse: collapse; width: 100%; }
td { padding: 0 6px; vertical-align: top; white-space: pre-wrap; }
td.ln { color: #999; text-align: right; width: 38px; user-select: none; }
td.sc { color: #777; width: 52px; text-align: right; font-size: 11px; }
tr.truth td.src { box-shadow: inset 3px 0 0 #2f8f2f; }
tr.top1 td.ln { color: #b00; font-weight: 700; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.col h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
          color: #666; margin: 0 0 6px; }
.diff .add { background: #e6ffed; }
.diff .del { background: #ffeef0; }
.legend { font-size: 11px; color: #666; margin-top: 4px; }
"""


def _shade(score: float, hi: float) -> str:
    """Background colour for a line, scaled against the function's own maximum.

    Normalising per function rather than globally is the point: attention sums to 1 over nodes, so
    a 200-line function spreads thinner than a 20-line one and a shared scale would make long
    functions look uniformly cold regardless of how peaked they actually are.
    """
    if hi <= 0:
        return ""
    a = max(0.0, min(1.0, score / hi))
    return f"background: rgba(214,122,20,{a * 0.55:.3f});"


def _source_table(source: str, line_scores: dict, truth: set[int], top1: int | None) -> str:
    lines = source.splitlines()
    hi = max(line_scores.values()) if line_scores else 0.0
    rows = []
    for i, text in enumerate(lines, 1):
        score = line_scores.get(i, 0.0)
        cls = " ".join(filter(None, ["truth" if i in truth else "",
                                     "top1" if i == top1 else ""]))
        rows.append(
            f'<tr class="{cls}"><td class="ln">{i}</td>'
            f'<td class="sc">{score:.3f}</td>'
            f'<td class="src" style="{_shade(score, hi)}">{html.escape(text)}</td></tr>')
    return "<table>" + "".join(rows) + "</table>"


def _diff_table(before: str, after: str) -> str:
    rows = []
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                     lineterm="", n=3):
        if line.startswith("+++") or line.startswith("---"):
            continue
        cls = "add" if line.startswith("+") else "del" if line.startswith("-") else ""
        rows.append(f'<tr><td class="src {cls}">{html.escape(line)}</td></tr>')
    return '<table class="diff">' + "".join(rows) + "</table>"


def _score_function(model, batch, sample, is_mil):
    if is_mil:
        with torch.no_grad():
            logit, attn = model(batch, return_attention=True)
        return float(torch.sigmoid(logit)[0]), attn[0].cpu().numpy()
    with torch.no_grad():
        logit = model(batch)
    return float(torch.sigmoid(logit)[0]), conv_saliency(model, batch)[0]


def collect(cfg, model_name, model_dir, primevul_path, aggregation, device, limit):
    """Score every eligible function once; selection happens afterwards on the results."""
    featurizer = NodeFeaturizer.load(os.path.join(cfg["data"]["processed_dir"], "featurizer"))
    from devign_data.dataset import make_collate_fn
    collate = make_collate_fn(EDGE_TYPES, sparse=cfg["dataset"].get("sparse", True))

    model = build_model(model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=len(featurizer.type_vocab),
                        num_edge_types=len(EDGE_TYPES))
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    with open(os.path.join(model_dir, "meta.json")) as fh:
        threshold = float(json.load(fh).get("threshold", 0.5))

    is_mil = model_name == "mil"
    cases = []
    for fn in load_primevul_paired(primevul_path, max_functions=limit):
        sample = _sample_from_source(fn, cfg, featurizer)
        if sample is None:
            continue
        batch = collate([sample]).to(device)
        prob, score = _score_function(model, batch, sample, is_mil)
        if prob < threshold:
            continue                       # true positives only
        line_scores = attention_to_lines(score, sample.node_lines, sample.num_nodes, aggregation)
        ranked = rank_lines(line_scores)
        m = localization_metrics(ranked, fn.vulnerable_lines)
        if m is None:
            continue
        cases.append({"fn": fn, "prob": prob, "line_scores": line_scores,
                      "ranked": ranked, "metrics": m,
                      "top1": ranked[0] if ranked else None,
                      "distance": min((abs(ranked[0] - t) for t in fn.vulnerable_lines),
                                      default=10 ** 6) if ranked else 10 ** 6})
    return cases, threshold


def select(cases, n_success, n_fail):
    """Successes spread across CWEs, plus the worst misses. Never successes only."""
    hits = sorted([c for c in cases if c["metrics"]["top_1"] == 1.0],
                  key=lambda c: -c["prob"])
    misses = sorted([c for c in cases if c["metrics"]["top_1"] == 0.0],
                    key=lambda c: -c["distance"])

    chosen, seen_cwe = [], set()
    for c in hits:                                  # one per CWE first, for spread
        cwe = (c["fn"].cwe or ["unknown"])[0]
        if cwe not in seen_cwe:
            chosen.append(c)
            seen_cwe.add(cwe)
        if len(chosen) >= n_success:
            break
    for c in hits:                                  # top up if CWEs ran out
        if len(chosen) >= n_success:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen, misses[:n_fail]


def render(successes, failures, model_name, aggregation, threshold, n_total) -> str:
    parts = [f"<style>{_CSS}</style>",
             f"<h1>Attention vs the actual fix — {html.escape(model_name)}</h1>",
             f'<div class="sub">line aggregation <b>{aggregation}</b> &middot; decision threshold '
             f'{threshold:.3f} &middot; {len(successes)} success(es) and {len(failures)} '
             f'failure(s) drawn from {n_total} true positives. Green bar = a line the patch '
             f'changed. Orange = attention, scaled to each function&rsquo;s own maximum. '
             f'Red line number = the model&rsquo;s top-ranked line.</div>']

    for label, group, cls in (("Success", successes, ""), ("Failure", failures, " fail")):
        for c in group:
            fn, m = c["fn"], c["metrics"]
            cwe = ", ".join(fn.cwe) or "unknown CWE"
            badge = "hit" if m["top_1"] == 1.0 else "miss"
            parts.append(f'<div class="case{cls}">')
            parts.append(
                f'<h2>{label} &mdash; {html.escape(cwe)} '
                f'<span class="badge {badge}">Top-1 {"hit" if m["top_1"] else "miss"}</span>'
                f'<span class="badge">IFA {int(m["ifa"])}</span>'
                f'<span class="badge">MRR {m["mrr"]:.2f}</span></h2>')
            truth = ", ".join(str(t) for t in sorted(fn.vulnerable_lines))
            note = ""
            if m["top_1"] == 0.0:
                note = (f' &middot; attention peaked on line {c["top1"]}, '
                        f'{c["distance"]} line(s) from the nearest patched line')
            parts.append(f'<div class="meta">p(vulnerable) = {c["prob"]:.3f} &middot; '
                         f'ground-truth lines: {truth}{note}</div>')
            parts.append('<div class="cols">')
            parts.append('<div class="col"><h3>attention overlay</h3>'
                         + _source_table(fn.func, c["line_scores"], fn.vulnerable_lines,
                                         c["top1"]) + "</div>")
            parts.append('<div class="col"><h3>the patch that defines the truth</h3>'
                         + _diff_table(fn.func, getattr(fn, "func_after", "") or "") + "</div>")
            parts.append("</div></div>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["mil", "devign", "ggrn"], default="mil")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--aggregation", choices=["max", "sum"], default="max")
    ap.add_argument("--successes", type=int, default=4)
    ap.add_argument("--failures", type=int, default=1,
                    help="never 0 by default: a section of only successes is not evidence")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="artifacts/attention_examples.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_from_config(cfg)
    device = resolve_device_from_config(cfg)
    cases, threshold = collect(cfg, args.model, args.model_dir, args.primevul,
                               args.aggregation, device, args.limit)
    if not cases:
        raise SystemExit("no true positives with ground-truth lines; nothing to render")

    successes, failures = select(cases, args.successes, args.failures)
    if not failures:
        print("[render] WARNING: no Top-1 miss available, so this page shows successes only. "
              "Say so in the write-up rather than letting it read as a clean sweep.")
    html_doc = render(successes, failures, args.model, args.aggregation, threshold, len(cases))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    print(f"[render] {len(successes)} success(es), {len(failures)} failure(s) "
          f"from {len(cases)} true positives -> {args.out}")


if __name__ == "__main__":
    main()
