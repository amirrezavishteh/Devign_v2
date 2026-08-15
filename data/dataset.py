"""Tensorizes composite code graphs and batches them with padding for variable-size graphs.

Batching strategy (paper requires "graph batching with padding for variable-size graphs"):
We pad every graph in a batch to the batch's max node count `M`. Node features become
[B, M, d]; adjacency matrices become dense [B, k, M, M]; a boolean node mask [B, M] marks real
vs padded nodes. The GGNN message passing (Eq. 3) is then a batched matmul A_p^T (W_p H), and the
Conv module consumes the fixed [B, M, *] tensors directly. Padding rows are zeroed and masked so
they contribute nothing to aggregation or pooling.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.download import RawFunction
from data.graph_builder import EDGE_TYPES, build_graph
from data.word2vec_embed import NodeFeaturizer


@dataclass
class GraphSample:
    code_feat: np.ndarray      # [m, code_dim] float32
    type_ids: np.ndarray       # [m] int64
    edges: dict[str, np.ndarray]  # edge_type -> [2, e] int64
    num_nodes: int
    label: int
    project: str
    name: str = ""


def dedupe_edges(arr: np.ndarray) -> np.ndarray:
    """Collapse repeated (src, dst) pairs in a [2, e] edge array.

    The paper defines the adjacency as binary, A in {0,1}^(k x m x m) (Sec 2.1): an edge either
    exists or it does not, and a relation asserted twice is still one edge. The heuristic DFG
    builder can emit the same pair more than once (e.g. a variable read twice in one expression),
    so we normalise here -- otherwise sparse accumulation would weight that edge double while the
    dense path, which assigns rather than accumulates, would not.
    """
    if arr.size == 0:
        return arr
    # Order-preserving unique over columns.
    _, keep = np.unique(arr[0].astype(np.int64) * (arr.max() + 1) + arr[1], return_index=True)
    return arr[:, np.sort(keep)]


def sample_from_graph(fn: RawFunction, graph, featurizer: NodeFeaturizer,
                      edge_types: list[str]) -> GraphSample:
    """Featurize an ALREADY-BUILT graph. Parsing is the expensive step, so callers that have
    already parsed (data/prepare.py) must use this rather than re-parsing via build_samples."""
    code_feat, type_ids = featurizer.featurize_graph(graph)
    edges = {}
    for et in edge_types:
        elist = graph.edges.get(et, [])
        if elist:
            arr = dedupe_edges(np.array(elist, dtype=np.int64).T)  # [2, e], binary A
        else:
            arr = np.zeros((2, 0), dtype=np.int64)
        edges[et] = arr
    return GraphSample(
        code_feat=code_feat, type_ids=type_ids, edges=edges,
        num_nodes=graph.num_nodes, label=int(fn.target),
        project=fn.project, name=fn.name,
    )


def samples_from_graphs(pairs, featurizer: NodeFeaturizer,
                        edge_types: list[str]) -> list[GraphSample]:
    """pairs: iterable of (RawFunction, CodeGraph) already parsed."""
    return [sample_from_graph(fn, g, featurizer, edge_types) for fn, g in pairs]


def build_samples(functions: list[RawFunction], featurizer: NodeFeaturizer,
                  edge_types: list[str], max_nodes: int) -> list[GraphSample]:
    """Parse + featurize each raw function into a GraphSample. Skips unparseable / oversized."""
    samples: list[GraphSample] = []
    for fn in functions:
        graph = build_graph(fn.func, max_nodes=max_nodes)
        if graph is None or graph.num_nodes == 0:
            continue
        samples.append(sample_from_graph(fn, graph, featurizer, edge_types))
    return samples


class DevignDataset(Dataset):
    def __init__(self, samples: list[GraphSample], type_vocab_size: int,
                 type_embedding: "torch.nn.Embedding | None" = None,
                 edge_types: list[str] = EDGE_TYPES):
        self.samples = samples
        self.type_vocab_size = type_vocab_size
        self.edge_types = edge_types

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"samples": self.samples, "type_vocab_size": self.type_vocab_size,
                         "edge_types": self.edge_types}, f)

    @classmethod
    def load(cls, path: str) -> "DevignDataset":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["samples"], d["type_vocab_size"], edge_types=d["edge_types"])


@dataclass
class GraphBatch:
    code_feat: torch.Tensor    # [B, M, code_dim]
    type_ids: torch.Tensor     # [B, M]
    adj: torch.Tensor | None   # [B, k, M, M] float32 dense A_p (None in sparse mode)
    mask: torch.Tensor         # [B, M] bool (True for real nodes)
    labels: torch.Tensor       # [B]
    num_nodes: torch.Tensor    # [B]
    projects: list[str]
    names: list[str]
    # Sparse mode: edges in the flattened [B*M] node space, so message passing never
    # materialises the M^2 adjacency. edge_index [2, E] = (src, dst); edge_type [E] = p.
    edge_index: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None
    edge_norm: torch.Tensor | None = None   # [E] 1/out-degree, only when normalize_adj is on

    def to(self, device) -> "GraphBatch":
        def _mv(t):
            return None if t is None else t.to(device)

        return GraphBatch(
            code_feat=self.code_feat.to(device),
            type_ids=self.type_ids.to(device),
            adj=_mv(self.adj),
            mask=self.mask.to(device),
            labels=self.labels.to(device),
            num_nodes=self.num_nodes.to(device),
            projects=self.projects,
            names=self.names,
            edge_index=_mv(self.edge_index),
            edge_type=_mv(self.edge_type),
            edge_norm=_mv(self.edge_norm),
        )


def make_collate_fn(edge_types: list[str], add_self_loops: bool = False,
                    normalize_adj: bool = False, sparse: bool = True):
    """Returns a collate_fn closing over the edge-type ordering used to stack adjacency matrices.

    `sparse=True` emits `edge_index`/`edge_type` in the flattened [B*M] node space instead of a
    dense [B, k, M, M] tensor. The two are mathematically identical (tests/test_sparse_
    equivalence.py asserts it), but the dense form costs 896 MB per batch at B=128, k=7, M=500 --
    it cannot run the paper's real 500-node graphs on an 8 GB GPU.

    `add_self_loops` / `normalize_adj` both default OFF: the paper defines A as raw binary in
    {0,1}^(k x m x m) (Sec 2.1).
    """
    k = len(edge_types)

    def collate(batch: list[GraphSample]) -> GraphBatch:
        B = len(batch)
        M = max(s.num_nodes for s in batch)
        code_dim = batch[0].code_feat.shape[1]

        code_feat = torch.zeros(B, M, code_dim, dtype=torch.float32)
        type_ids = torch.zeros(B, M, dtype=torch.long)
        mask = torch.zeros(B, M, dtype=torch.bool)
        labels = torch.zeros(B, dtype=torch.float32)
        num_nodes = torch.zeros(B, dtype=torch.long)
        projects, names = [], []
        src_all, dst_all, type_all = [], [], []

        for bi, s in enumerate(batch):
            m = s.num_nodes
            code_feat[bi, :m] = torch.from_numpy(s.code_feat)
            type_ids[bi, :m] = torch.from_numpy(s.type_ids)
            mask[bi, :m] = True
            labels[bi] = float(s.label)
            num_nodes[bi] = m
            projects.append(s.project)
            names.append(s.name)
            offset = bi * M  # flattened [B*M] node space
            for ei, et in enumerate(edge_types):
                e = s.edges.get(et)
                if e is None or e.shape[1] == 0:
                    continue
                src = torch.from_numpy(e[0]) + offset
                dst = torch.from_numpy(e[1]) + offset
                src_all.append(src)
                dst_all.append(dst)
                type_all.append(torch.full((src.numel(),), ei, dtype=torch.long))

            if add_self_loops:
                loop = torch.arange(m, dtype=torch.long) + offset
                for ei in range(k):
                    src_all.append(loop)
                    dst_all.append(loop)
                    type_all.append(torch.full((m,), ei, dtype=torch.long))

        if src_all:
            edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)])
            edge_type = torch.cat(type_all)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        if add_self_loops and edge_index.shape[1]:
            # Per-sample edges are already binary, but an injected loop can collide with a
            # genuine self-edge. Collapse so A stays in {0,1} and degrees are counted once.
            N = B * M
            key = (edge_type * N + edge_index[0]) * N + edge_index[1]
            uniq = torch.unique(key)
            dst_u = uniq % N
            rest = torch.div(uniq, N, rounding_mode="floor")
            edge_index = torch.stack([rest % N, dst_u])
            edge_type = torch.div(rest, N, rounding_mode="floor")

        edge_norm = None
        if normalize_adj:
            # 1/out-degree of the source within its own edge type, matching the dense path's
            # row-normalisation of A_p.
            flat = edge_type * (B * M) + edge_index[0]
            deg = torch.zeros(k * B * M, dtype=torch.float32)
            deg.index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.float32))
            edge_norm = 1.0 / deg[flat].clamp(min=1.0)

        # Bounds check on the CPU. An out-of-range index here would become an illegal memory
        # access inside index_add_/index_put_ on the GPU, which CUDA reports asynchronously as
        # "unspecified launch failure" at some unrelated later line -- killing a multi-hour run
        # with a traceback that points nowhere near the cause. Cost is one max() per batch.
        if edge_index.numel():
            n_slots = B * M
            hi = int(edge_index.max())
            lo = int(edge_index.min())
            if lo < 0 or hi >= n_slots:
                raise ValueError(
                    f"edge_index out of range: [{lo}, {hi}] outside [0, {n_slots}) "
                    f"(B={B}, M={M}). A graph's edges reference a node id >= its num_nodes.")
            t_hi = int(edge_type.max())
            if t_hi >= k or int(edge_type.min()) < 0:
                raise ValueError(f"edge_type out of range: max {t_hi} for {k} edge types")

        adj = None
        if not sparse:
            adj = torch.zeros(B, k, M, M, dtype=torch.float32)
            vals = edge_norm if edge_norm is not None else torch.ones(edge_index.shape[1])
            if edge_index.shape[1]:
                bi_idx = torch.div(edge_index[0], M, rounding_mode="floor")
                adj[bi_idx, edge_type, edge_index[0] % M, edge_index[1] % M] = vals

        return GraphBatch(
            code_feat=code_feat, type_ids=type_ids, adj=adj, mask=mask,
            labels=labels, num_nodes=num_nodes, projects=projects, names=names,
            edge_index=edge_index, edge_type=edge_type, edge_norm=edge_norm,
        )

    return collate


class BucketBySizeSampler(torch.utils.data.Sampler):
    """Yields size-bucketed micro-batches under a hard node budget.

    Two problems this solves on real data (synthetic graphs were 20-60 nodes, so neither showed):

    1. *Padding waste.* Real Devign functions run from a handful of nodes to the 500-node cap, so
       a randomly composed batch pads almost everything up to ~500. Sorting by size first means a
       batch's members are near-identical in size.
    2. *Peak memory.* The Conv module materialises [B, C, M, W] maps, so cost scales with B*M, not
       B. A batch of 128 500-node graphs is an order of magnitude heavier than 128 50-node ones.
       Capping `B*M` bounds peak memory regardless of the size distribution.

    Because a node-budget batch can be smaller than the paper's batch_size=128, the trainer
    accumulates gradients across micro-batches and steps once per `batch_size` graphs -- so the
    effective batch size stays exactly what Sec 3.3 specifies.
    """

    def __init__(self, sizes: list[int], batch_size: int, max_nodes_per_batch: int | None = None,
                 shuffle: bool = True, seed: int = 42):
        self.sizes = sizes
        self.batch_size = batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self._len = len(list(self._build(0)))

    def _build(self, epoch: int):
        import random
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.sizes)))
        if self.shuffle:
            rng.shuffle(order)                      # break size ties differently each epoch
        order.sort(key=lambda i: self.sizes[i])

        batches, cur, cur_max = [], [], 0
        for idx in order:
            nxt_max = max(cur_max, self.sizes[idx])
            over_count = len(cur) + 1 > self.batch_size
            over_nodes = (self.max_nodes_per_batch is not None
                          and cur and (len(cur) + 1) * nxt_max > self.max_nodes_per_batch)
            if cur and (over_count or over_nodes):
                batches.append(cur)
                cur, cur_max = [], 0
                nxt_max = self.sizes[idx]
            cur.append(idx)
            cur_max = nxt_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build(self.epoch)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return self._len
