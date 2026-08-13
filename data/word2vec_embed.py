"""Node feature construction: Code (word2vec, dim=100) + Type (label encoding), concatenated
into the initial node representation x_v (Sec 2.2.2).

For leaf AST nodes, "Code" is simply the token text. For internal nodes, the Code feature is the
mean of the word2vec vectors of all leaf tokens spanned by that subtree (the paper's Figure 2
shows internal nodes annotated with their full source span, e.g. "a+b / AddExp"; word2vec only
has a token-level vocabulary, so mean-pooling over the spanned tokens is the standard way to lift
a token embedding model to subtree-level "Code" features).
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec

from data.graph_builder import CodeGraph
from data.parser import ASTNode

UNK_TYPE = "<UNK_TYPE>"


def collect_leaf_tokens(nodes: list[ASTNode], node_id: int) -> list[str]:
    node = nodes[node_id]
    if node.is_leaf:
        return [node.code]
    toks: list[str] = []
    for c in node.children:
        toks.extend(collect_leaf_tokens(nodes, c))
    return toks or [node.code]


def build_corpus(graphs: Iterable[CodeGraph]) -> list[list[str]]:
    """One sentence per function: leaf tokens in natural code sequence (NCS) order."""
    corpus = []
    for g in graphs:
        leaves = sorted((n for n in g.nodes if n.is_leaf), key=lambda n: n.start_byte)
        tokens = [n.code for n in leaves]
        if tokens:
            corpus.append(tokens)
    return corpus


def train_word2vec(corpus: list[list[str]], dim: int = 100, window: int = 5,
                    min_count: int = 1, epochs: int = 10, workers: int = 4) -> Word2Vec:
    model = Word2Vec(
        sentences=corpus, vector_size=dim, window=window, min_count=min_count,
        workers=workers, epochs=epochs, sg=1,
    )
    return model


class TypeVocab:
    def __init__(self, types: Iterable[str] | None = None):
        self.type2id: dict[str, int] = {UNK_TYPE: 0}
        if types:
            for t in sorted(set(types)):
                self.add(t)

    def add(self, type_str: str) -> int:
        if type_str not in self.type2id:
            self.type2id[type_str] = len(self.type2id)
        return self.type2id[type_str]

    def get(self, type_str: str) -> int:
        return self.type2id.get(type_str, 0)

    def __len__(self) -> int:
        return len(self.type2id)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.type2id, f)

    @classmethod
    def load(cls, path: str) -> "TypeVocab":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.type2id = json.load(f)
        return vocab


class NodeFeaturizer:
    """Wraps a trained word2vec model + type vocabulary to produce x_v for every node."""

    def __init__(self, w2v: Word2Vec, type_vocab: TypeVocab):
        self.w2v = w2v
        self.type_vocab = type_vocab
        self.code_dim = w2v.vector_size

    def code_vector(self, nodes: list[ASTNode], node_id: int) -> np.ndarray:
        tokens = collect_leaf_tokens(nodes, node_id)
        vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        if not vecs:
            return np.zeros(self.code_dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def type_id(self, node: ASTNode) -> int:
        return self.type_vocab.get(node.type)

    def featurize_graph(self, graph: CodeGraph) -> tuple[np.ndarray, np.ndarray]:
        """Returns (code_matrix [m, code_dim] float32, type_ids [m] int64).

        Code vectors are the mean of word2vec vectors over each node's spanned leaf tokens.
        Computed bottom-up in a single O(m) pass instead of per-node leaf collection (which would
        be O(m) per node, i.e. O(m^2) per graph): since `nodes` is pre-order DFS, every child id is
        strictly greater than its parent's, so iterating ids in reverse processes every child
        before its parent, letting each node simply sum its direct children's (sum, count).
        """
        nodes = graph.nodes
        m = len(nodes)
        sums = np.zeros((m, self.code_dim), dtype=np.float32)
        counts = np.zeros(m, dtype=np.int64)
        for nid in range(m - 1, -1, -1):
            node = nodes[nid]
            if node.is_leaf:
                vec = self.w2v.wv[node.code] if node.code in self.w2v.wv else None
                if vec is not None:
                    sums[nid] = vec
                    counts[nid] = 1
            else:
                for cid in node.children:
                    sums[nid] += sums[cid]
                    counts[nid] += counts[cid]
        safe_counts = np.maximum(counts, 1).reshape(-1, 1)
        code_mat = (sums / safe_counts).astype(np.float32)
        code_mat[counts == 0] = 0.0
        type_ids = np.array([self.type_id(n) for n in nodes], dtype=np.int64)
        return code_mat, type_ids

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.w2v.save(os.path.join(out_dir, "word2vec.model"))
        self.type_vocab.save(os.path.join(out_dir, "type_vocab.json"))

    @classmethod
    def load(cls, out_dir: str) -> "NodeFeaturizer":
        w2v = Word2Vec.load(os.path.join(out_dir, "word2vec.model"))
        type_vocab = TypeVocab.load(os.path.join(out_dir, "type_vocab.json"))
        return cls(w2v, type_vocab)
