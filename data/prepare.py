"""End-to-end data preparation:

  raw functions -> parse to composite graphs -> train word2vec on the code corpus ->
  build the type vocabulary -> featurize each graph (x_v) -> cache DevignDataset + featurizer.

Also builds the token vocabulary + sequence samples used by the BiLSTM/CNN baselines, so the
baselines share the exact same train/val split and word2vec vectors as Devign.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from data.dataset import DevignDataset, build_samples, samples_from_graphs
from data.download import RawFunction, acquire_dataset
from data.graph_builder import EDGE_TYPES, build_graph
from data.word2vec_embed import (NodeFeaturizer, TypeVocab, build_corpus,
                                 train_word2vec)


def stratified_split(functions: list[RawFunction], train_frac: float, seed: int):
    """Two-way class-stratified split (the paper's exact 75/25). Kept for back-compat."""
    train, val, test = stratified_split3(functions, train_frac, 1.0 - train_frac, 0.0, seed)
    assert not test
    return train, val


def stratified_split3(functions: list[RawFunction], train_frac: float, val_frac: float,
                      test_frac: float, seed: int):
    """Class-stratified three-way split.

    The paper (Sec 3.3) shuffles and splits 75/25 with no test set, so its reported number is
    selected on the same data early stopping watches. We keep the 75% training fraction intact
    and halve the remainder, so Table 2 stays comparable while `test` gives an unbiased number.
    """
    import random
    rng = random.Random(seed)
    by_class: dict[int, list[RawFunction]] = {0: [], 1: []}
    for fn in functions:
        by_class[int(fn.target)].append(fn)
    train, val, test = [], [], []
    for _cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def commit_disjoint_split(functions: list[RawFunction], train_frac: float, val_frac: float,
                          test_frac: float, seed: int):
    """Split so that no commit contributes functions to more than one side.

    A random split lets several functions touched by the same vulnerability-fix commit land on
    both sides, which inflates scores: the model can recognise the commit rather than the flaw.
    The paper splits randomly, so this is an integrity check reported alongside -- not the
    headline configuration.
    """
    import random
    rng = random.Random(seed)
    by_commit: dict[str, list[RawFunction]] = {}
    for i, fn in enumerate(functions):
        # Functions with no commit id are their own group so they can't silently merge.
        key = fn.commit_id or f"__nocommit_{i}"
        by_commit.setdefault(key, []).append(fn)

    groups = list(by_commit.values())
    rng.shuffle(groups)
    total = len(functions)
    n_train, n_val = int(total * train_frac), int(total * val_frac)

    train, val, test = [], [], []
    for grp in groups:
        if len(train) < n_train:
            train.extend(grp)
        elif len(val) < n_val:
            val.extend(grp)
        else:
            test.extend(grp)
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def split_functions(functions: list[RawFunction], data_cfg: dict, seed: int):
    """Dispatch on data.paper_split / data.split_by. Returns (train, val, test)."""
    train_frac = data_cfg["train_split"]
    if data_cfg.get("paper_split"):
        val_frac, test_frac = 1.0 - train_frac, 0.0
    else:
        val_frac = data_cfg.get("val_split", (1.0 - train_frac) / 2)
        test_frac = data_cfg.get("test_split", (1.0 - train_frac) / 2)

    if data_cfg.get("split_by", "random") == "commit":
        return commit_disjoint_split(functions, train_frac, val_frac, test_frac, seed)
    return stratified_split3(functions, train_frac, val_frac, test_frac, seed)


# ----------------------------------------------------------------------------------------------
# Parallel graph construction
# ----------------------------------------------------------------------------------------------

_WORKER_MAX_NODES = 500
_WORKER_MAX_ERROR_FRACTION = 0.2


def _init_worker(max_nodes: int, max_error_fraction: float) -> None:
    global _WORKER_MAX_NODES, _WORKER_MAX_ERROR_FRACTION
    _WORKER_MAX_NODES = max_nodes
    _WORKER_MAX_ERROR_FRACTION = max_error_fraction


def _build_one(source: str):
    """Top-level so it is picklable under Windows spawn. Returns (CodeGraph|None, reason)."""
    from data.graph_builder import build_graph_with_reason
    try:
        return build_graph_with_reason(source, max_nodes=_WORKER_MAX_NODES,
                                       max_error_fraction=_WORKER_MAX_ERROR_FRACTION)
    except Exception:
        # A parser blow-up on one pathological function must not kill the whole run.
        return None, "exception"


def _build_one_serial(source: str, max_nodes: int, max_error_fraction: float):
    from data.graph_builder import build_graph_with_reason
    try:
        return build_graph_with_reason(source, max_nodes=max_nodes,
                                       max_error_fraction=max_error_fraction)
    except Exception:
        return None, "exception"


def build_graphs_parallel(functions: list[RawFunction], max_nodes: int, workers: int,
                          max_error_fraction: float = 0.2, verbose: bool = True):
    """Parse every function ONCE, in parallel. Returns [(RawFunction, CodeGraph)] for survivors.

    Drop reasons are reported separately: the >max_nodes filter is the paper's own (Sec 3.1,
    ~15% of data), while the parse filter is ours, and conflating them previously hid that the
    latter was silently discarding an extra 12.1% of the corpus.
    """
    sources = [fn.func for fn in functions]
    if workers <= 1 or len(sources) < 200:
        results = [_build_one_serial(s, max_nodes, max_error_fraction) for s in sources]
    else:
        import multiprocessing as mp
        chunk = max(1, len(sources) // (workers * 8))
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(max_nodes, max_error_fraction)) as pool:
            results = pool.map(_build_one, sources, chunksize=chunk)

    pairs, reasons = [], {}
    for fn, (graph, reason) in zip(functions, results):
        if graph is not None and graph.num_nodes > 0:
            pairs.append((fn, graph))
        else:
            reasons[reason or "empty"] = reasons.get(reason or "empty", 0) + 1

    if verbose:
        total = max(1, len(functions))
        dropped = total - len(pairs)
        print(f"[prepare] graphs built: {len(pairs)} kept ({100.0 * len(pairs) / total:.1f}%), "
              f"{dropped} dropped ({100.0 * dropped / total:.1f}%)")
        labels = {"size": f">{max_nodes} nodes (paper's filter)",
                  "parse": f"unparseable (no function_definition, or >{max_error_fraction:.0%} ERROR span)",
                  "empty": "no AST nodes", "exception": "parser exception"}
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"[prepare]   {count:6d} ({100.0 * count / total:4.1f}%)  "
                  f"{labels.get(reason, reason)}")
    return pairs


def prepare(config: dict, verbose: bool = True):
    data_cfg = config["data"]
    emb_cfg = config["embedding"]
    proc_dir = data_cfg["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    # 1. Acquire raw functions.
    raw_path = os.path.join(data_cfg["raw_dir"], "functions.jsonl")
    functions = acquire_dataset(config, out_path=raw_path)
    if verbose:
        n_vuln = sum(f.target for f in functions)
        print(f"[prepare] {len(functions)} raw functions ({n_vuln} vulnerable)")

    # 2. Parse every function exactly ONCE, in parallel, and keep the graphs for step 6.
    #    (The paper trains word2vec on the whole code corpus of the projects, so we fit on all
    #    parseable functions -- that is the corpus, not the labels.)
    workers = data_cfg.get("parse_workers") or max(1, (os.cpu_count() or 2) - 1)
    pairs = build_graphs_parallel(functions, data_cfg["max_nodes"], workers,
                                  max_error_fraction=data_cfg.get("max_error_fraction", 0.2),
                                  verbose=verbose)
    parseable = [fn for fn, _ in pairs]
    graphs = [g for _, g in pairs]
    graph_by_id = {id(fn): g for fn, g in pairs}

    # 3. Train word2vec on the code corpus.
    corpus = build_corpus(graphs)
    w2v = train_word2vec(
        corpus, dim=emb_cfg["word2vec_dim"], window=emb_cfg["word2vec_window"],
        min_count=emb_cfg["word2vec_min_count"], epochs=emb_cfg["word2vec_epochs"],
        workers=emb_cfg["word2vec_workers"],
    )
    if verbose:
        print(f"[prepare] word2vec vocab: {len(w2v.wv)} tokens, dim {w2v.vector_size}")

    # 4. Build type vocabulary.
    type_vocab = TypeVocab()
    for g in graphs:
        for n in g.nodes:
            type_vocab.add(n.type)
    if verbose:
        print(f"[prepare] type vocab: {len(type_vocab)} types")

    featurizer = NodeFeaturizer(
        w2v, type_vocab,
        internal_code=emb_cfg.get("internal_code", "mean_standardized"),
        oov_code=emb_cfg.get("oov_code", "random"),
        seed=config["project"]["seed"])

    # 5. Split (75/12.5/12.5 by default; data.paper_split restores the paper's exact 75/25).
    train_funcs, val_funcs, test_funcs = split_functions(
        parseable, data_cfg, config["project"]["seed"])
    if verbose:
        mode = data_cfg.get("split_by", "random")
        print(f"[prepare] split ({mode}): {len(train_funcs)} train / {len(val_funcs)} val / "
              f"{len(test_funcs)} test")

    # 5b. Fit code-feature standardization on the TRAINING graphs only, then persist the
    #     featurizer. This has to happen after the split (val/test statistics must not leak in)
    #     and before featurization, which consumes the fitted stats.
    featurizer.fit_standardizer(graph_by_id[id(fn)] for fn in train_funcs)
    if verbose and featurizer.code_mean is not None:
        print(f"[prepare] code features standardized on {len(train_funcs)} train graphs "
              f"(mean |mu| {np.abs(featurizer.code_mean).mean():.3f}, "
              f"mean sigma {featurizer.code_std.mean():.3f})")
    featurizer.save(os.path.join(proc_dir, "featurizer"))

    # 6. Featurize the graphs we already built -- no re-parsing.
    def _samples_for(funcs):
        return samples_from_graphs(((fn, graph_by_id[id(fn)]) for fn in funcs),
                                   featurizer, EDGE_TYPES)

    train_samples = _samples_for(train_funcs)
    val_samples = _samples_for(val_funcs)
    test_samples = _samples_for(test_funcs)

    DevignDataset(train_samples, len(type_vocab)).save(os.path.join(proc_dir, "train.pkl"))
    DevignDataset(val_samples, len(type_vocab)).save(os.path.join(proc_dir, "val.pkl"))
    if test_samples:
        DevignDataset(test_samples, len(type_vocab)).save(os.path.join(proc_dir, "test.pkl"))

    # 7. Sequence baseline data (token vocab + samples) sharing the same split.
    _prepare_sequences(train_funcs, val_funcs, test_funcs, w2v, proc_dir, config)

    # 8. Persist splits for reuse (XGBoost, imbalanced study, Q5 holdout).
    with open(os.path.join(proc_dir, "splits.pkl"), "wb") as f:
        pickle.dump({"train": train_funcs, "val": val_funcs, "test": test_funcs}, f)

    if verbose:
        print("[prepare] done.")
    return {
        "type_vocab_size": len(type_vocab),
        "code_dim": w2v.vector_size,
        "num_edge_types": len(EDGE_TYPES),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }


def _prepare_sequences(train_funcs, val_funcs, test_funcs, w2v, proc_dir, config):
    from models.baselines import SequenceSample, TokenVocab, tokenize_c
    max_len = config["data"]["max_nodes"]
    tok_vocab = TokenVocab()

    def to_samples(funcs, fit_vocab):
        out = []
        for fn in funcs:
            tokens = tokenize_c(fn.func)
            if not tokens:
                continue
            ids = [tok_vocab.add(t) if fit_vocab else tok_vocab.get(t) for t in tokens]
            out.append(SequenceSample(ids, int(fn.target), fn.project))
        return out

    train_seq = to_samples(train_funcs, fit_vocab=True)
    val_seq = to_samples(val_funcs, fit_vocab=False)
    test_seq = to_samples(test_funcs, fit_vocab=False)

    # Build a pretrained embedding matrix aligned to tok_vocab from the word2vec model.
    dim = w2v.vector_size
    embed = np.random.normal(0, 0.1, (len(tok_vocab), dim)).astype(np.float32)
    embed[0] = 0.0  # PAD
    for tok, idx in tok_vocab.tok2id.items():
        if tok in w2v.wv:
            embed[idx] = w2v.wv[tok]

    with open(os.path.join(proc_dir, "sequences.pkl"), "wb") as f:
        pickle.dump({
            "train": train_seq, "val": val_seq, "test": test_seq,
            "vocab_size": len(tok_vocab), "embed": embed, "max_len": max_len,
        }, f)
