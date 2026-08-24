"""Evaluate line-level localisation from MIL attention against PrimeVul ground truth.

Protocol, fixed before any number is produced:

  * Scored on TRUE POSITIVES ONLY. Localising inside a function the model called clean is not a
    meaningful measurement, and the true-positive count every mean rests on is printed beside it.
  * Three baselines, not one. Random line ranking is the floor. The line-length prior is the row
    that decides whether the result means anything: long lines are disproportionately complex
    expressions, so if attention cannot beat length then attention has learned nothing about
    vulnerability. The Conv module gets the best localisation obtainable from it (see below).
  * Score entropy is reported per graph. Near-uniform scores mean the readout has collapsed to
    mean pooling, and any localisation win would be an artifact of the projection rather than of
    the model.

The Conv module row is deliberately weak, and that is a property of Eq. 9 rather than a choice
made here: its node axis is pooled away before the MLP heads, so there is no per-node quantity to
read out. The best available proxy is the gradient of the logit with respect to each node's
embedding (input saliency), which is what is used, and it is labelled a proxy wherever it appears.

Usage:
    python -m scripts.run_localization --config configs/a100_codexglue.yaml \
        --model mil --model-dir artifacts/seed1/mil/combined \
        --primevul data/primevul/test_paired.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from devign_data.dataset import GraphSample, make_collate_fn, node_line_spans
from devign_data.graph_builder import EDGE_TYPES, build_graph
from devign_data.primevul import line_lengths, load_primevul_paired
from devign_data.word2vec_embed import NodeFeaturizer
from evaluation.localize import (aggregate, attention_to_lines, length_prior_baseline,
                                localization_metrics, random_baseline, rank_lines)
from models.devign import build_model
from training.utils import load_config, resolve_device_from_config, seed_from_config


def normalised_entropy(a: np.ndarray) -> float:
    """Shannon entropy of one score row, divided by log(n) so it lands in [0, 1].

    1.0 means uniform -- the model spread its mass evenly and is attending to nothing in
    particular. 0.0 means everything sits on a single node.

    The input is renormalised to sum to 1 first. MIL attention already does, but input-gradient
    saliency does not: feeding it raw produced a "normalised" entropy of 1.027, which is
    impossible on [0, 1] and was the tell that the quantity was undefined for that arm. Both
    readouts now yield a comparable spread measure rather than one real number and one nonsense
    one.
    """
    a = np.asarray(a, dtype=np.float64)
    a = a[a > 0]
    if a.size <= 1:
        return 0.0
    total = a.sum()
    if total <= 0:
        return 0.0
    a = a / total
    return float(-(a * np.log(a)).sum() / np.log(a.size))


def conv_saliency(model, batch) -> np.ndarray:
    """|d logit / d node embedding| per node: the best localisation Eq. 9 can offer.

    The Conv module pools the node axis away before its MLP heads, so it exposes no per-node
    score of its own. Input-gradient saliency is the standard fallback for that situation.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    H, x = model.trunk(batch)
    H = H.detach().requires_grad_(True)
    x = x.detach().requires_grad_(True)
    logits = model.conv(H, x, batch.mask)
    logits.sum().backward()
    grad = torch.cat([H.grad, x.grad], dim=-1)
    return grad.abs().sum(-1).detach().cpu().numpy()


def _sample_from_source(fn, cfg, featurizer):
    """Parse and featurise one PrimeVul function into a GraphSample, or None if unusable."""
    graph = build_graph(fn.func, max_nodes=cfg["data"]["max_nodes"])
    if graph is None or graph.num_nodes == 0:
        return None
    code_feat, type_ids = featurizer.featurize_graph(graph)
    edges = {}
    for et in EDGE_TYPES:
        elist = graph.edges.get(et, [])
        edges[et] = (np.array(elist, dtype=np.int64).T if elist
                     else np.zeros((2, 0), dtype=np.int64))
    return GraphSample(
        code_feat=code_feat, type_ids=type_ids, edges=edges, num_nodes=graph.num_nodes,
        label=1, project=fn.project, name=fn.name,
        node_lines=node_line_spans(graph, fn.func),
    )


def evaluate_localization(cfg, model_name, model_dir, primevul_path, aggregation, device, limit):
    featurizer = NodeFeaturizer.load(os.path.join(cfg["data"]["processed_dir"], "featurizer"))
    collate = make_collate_fn(EDGE_TYPES, sparse=cfg["dataset"].get("sparse", True))

    model = build_model(model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=len(featurizer.type_vocab),
                        num_edge_types=len(EDGE_TYPES))
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                     map_location=device, weights_only=True))
    model.to(device).eval()

    with open(os.path.join(model_dir, "meta.json")) as fh:
        threshold = float(json.load(fh).get("threshold", 0.5))

    functions = load_primevul_paired(primevul_path, max_functions=limit)
    print(f"[localize] {len(functions)} PrimeVul functions carrying derived line labels")

    rows_model, rows_rand, rows_len = [], [], []
    entropies = []
    per_cwe = {}
    n_true_pos = n_pred_clean = n_unparsed = n_no_metric = 0
    rng = np.random.default_rng(cfg["project"]["seed"])
    is_mil = model_name == "mil"

    for fn in functions:
        sample = _sample_from_source(fn, cfg, featurizer)
        if sample is None:
            n_unparsed += 1
            continue
        batch = collate([sample]).to(device)

        if is_mil:
            with torch.no_grad():
                logit, attn = model(batch, return_attention=True)
            score = attn[0].cpu().numpy()                     # [heads, M]
        else:
            with torch.no_grad():
                logit = model(batch)
            score = conv_saliency(model, batch)[0]            # [M]

        # True positives only: a ranking inside a function the model called clean says nothing
        # about whether the model can point at a flaw it believes is there.
        if float(torch.sigmoid(logit)[0]) < threshold:
            n_pred_clean += 1
            continue
        n_true_pos += 1

        flat = score.mean(0) if score.ndim == 2 else score
        entropies.append(normalised_entropy(flat[:sample.num_nodes]))

        line_scores = attention_to_lines(score, sample.node_lines, sample.num_nodes, aggregation)
        m = localization_metrics(rank_lines(line_scores), fn.vulnerable_lines)
        if m is None:
            n_no_metric += 1
            continue
        rows_model.append(m)

        rb = random_baseline(fn.n_lines, fn.vulnerable_lines, rng)
        if rb:
            rows_rand.append(rb)
        lb = length_prior_baseline(line_lengths(fn.func), fn.vulnerable_lines)
        if lb:
            rows_len.append(lb)
        for cwe in (fn.cwe or ["unknown"]):
            per_cwe.setdefault(cwe, []).append(m)

    return {
        "model": model_name,
        "score_source": "attention" if is_mil else "input-gradient saliency (PROXY)",
        "aggregation": aggregation,
        "threshold": threshold,
        "counts": {
            "functions": len(functions), "true_positives": n_true_pos,
            "predicted_clean_skipped": n_pred_clean, "unparseable": n_unparsed,
            "no_metric": n_no_metric,
        },
        "model_scores": aggregate(rows_model) if rows_model else None,
        "random_baseline": aggregate(rows_rand) if rows_rand else None,
        "length_prior": aggregate(rows_len) if rows_len else None,
        "score_entropy": {
            "mean": float(np.mean(entropies)) if entropies else None,
            "median": float(np.median(entropies)) if entropies else None,
            "n": len(entropies),
        },
        # Fewer than 5 functions is not a per-CWE result, it is an anecdote.
        "per_cwe": {c: aggregate(v) for c, v in sorted(per_cwe.items()) if len(v) >= 5},
    }


def report(res):
    c = res["counts"]
    print()
    print(f"model {res['model']}  score source: {res['score_source']}")
    print(f"aggregation {res['aggregation']}   decision threshold {res['threshold']:.3f}")
    print(f"functions {c['functions']} | true positives {c['true_positives']} | "
          f"predicted-clean skipped {c['predicted_clean_skipped']} | "
          f"unparseable {c['unparseable']}")
    print()
    head = ("method", "Top-1", "Top-5", "MRR", "n.rank", "IFA", "IFAmed", "n")
    print("%-24s%8s%8s%8s%9s%8s%8s%6s" % head)
    print("-" * 79)
    for label, key in (("random ranking", "random_baseline"),
                       ("line-length prior", "length_prior"),
                       (res["model"] + " scores", "model_scores")):
        r = res.get(key)
        if not r or not r.get("n_functions"):
            print("%-24s%s" % (label, "not measured"))
            continue
        print("%-24s%8.3f%8.3f%8.3f%9.3f%8.1f%8.1f%6d" % (
            label, r["top_1"], r["top_5"], r["mrr"], r["normalised_rank"],
            r["ifa"], r["ifa_median"], r["n_functions"]))

    e = res["score_entropy"]
    if e["mean"] is not None:
        print()
        print(f"score entropy (0 = one node, 1 = uniform): mean {e['mean']:.3f} "
              f"median {e['median']:.3f} over {e['n']} graphs")
        if e["mean"] > 0.9:
            print("  WARNING: near-uniform. The pooling has collapsed toward mean pooling, so a "
                  "localisation win here would be an artifact of the projection, not the model.")

    if res["per_cwe"]:
        print()
        print("%-16s%8s%8s%8s%6s" % ("CWE", "Top-1", "MRR", "IFA", "n"))
        for cwe, r in res["per_cwe"].items():
            print("%-16s%8.3f%8.3f%8.1f%6d" % (cwe, r["top_1"], r["mrr"], r["ifa"],
                                               r["n_functions"]))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["mil", "devign", "ggrn"], default="mil")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--primevul", required=True, help="path to a PrimeVul *_paired.jsonl")
    ap.add_argument("--aggregation", choices=["max", "sum"], default="max")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_from_config(cfg)
    device = resolve_device_from_config(cfg)
    res = evaluate_localization(cfg, args.model, args.model_dir, args.primevul,
                               args.aggregation, device, args.limit)
    report(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[localize] wrote {args.out}")


if __name__ == "__main__":
    main()
