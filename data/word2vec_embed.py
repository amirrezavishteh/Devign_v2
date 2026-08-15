"""Node feature construction: Code (word2vec, dim=100) + Type (label encoding), concatenated
into the initial node representation x_v (Sec 2.2.2).

For leaf AST nodes, "Code" is simply the token text. For internal nodes, the Code feature is the
mean of the word2vec vectors of all leaf tokens spanned by that subtree (the paper's Figure 2
shows internal nodes annotated with their full source span, e.g. "a+b / AddExp"; word2vec only
has a token-level vocabulary, so mean-pooling over the spanned tokens is the standard way to lift
a token embedding model to subtree-level "Code" features), standardized against corpus statistics
so that the averaging does not collapse the upper AST toward a single constant vector.

See `NodeFeaturizer` for the alternatives and the measurements behind the defaults.
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


INTERNAL_CODE_MODES = {"zero", "mean", "mean_standardized"}
OOV_CODE_MODES = {"zero", "random"}


class NodeFeaturizer:
    """Wraps a trained word2vec model + type vocabulary to produce x_v for every node.

    `internal_code` controls how non-leaf nodes get their Code feature:

    ``"mean_standardized"`` (default)
        Subtree mean-pooling as in ``"mean"``, then standardized per dimension against corpus
        statistics fitted on the TRAINING graphs only (`fit_standardizer`). This is the mode that
        addresses both failures below at once: the mean's near-constant component is exactly what
        subtracting the corpus mean removes, and dividing by the per-dimension std restores the
        variance that mean-pooling shrinks. Internal nodes keep their lexical content.

    ``"zero"``
        Leaves carry their token vector; internal nodes carry only their Type embedding, with a
        zero code vector, and the GGNN is expected to aggregate lexical information up the
        AST/REV_AST edges. Measured on this corpus, that expectation does not hold: 43.8% of nodes
        are internal, another 8% of leaf-token occurrences fall below `word2vec_min_count`, so
        **49.6% of all nodes end up with an all-zero code half** -- while `model.time_steps` is 6
        against a **median AST depth of 11**, so the upper half of a typical tree never receives a
        token vector at all. Retained for comparison.

    ``"mean"``
        Plain subtree mean-pooling, unstandardized. Measured on real FFmpeg/QEMU functions this
        washes the signal out: **root-node vectors from different functions had cosine similarity
        0.834**, and internal-node variance was only 0.756x leaf variance. Retained for comparison;
        prefer ``mean_standardized``, which keeps the content and fixes the flatness.

    `oov_code` controls leaves whose token fell below `word2vec_min_count`:

    ``"random"`` (default)
        One fixed random vector, scaled to the corpus's typical vector norm, shared by all OOV
        tokens. Zero is *not* a neutral value -- it is a specific point in the space that, under
        ``internal_code: zero``, is also what every internal node gets, so "rare identifier" and
        "internal AST node" become indistinguishable. A dedicated vector separates them.

    ``"zero"``
        The previous behaviour. Retained for comparison.
    """

    def __init__(self, w2v: Word2Vec, type_vocab: TypeVocab,
                 internal_code: str = "mean_standardized", oov_code: str = "random",
                 code_mean: np.ndarray | None = None, code_std: np.ndarray | None = None,
                 oov_vector: np.ndarray | None = None, seed: int = 42):
        if internal_code not in INTERNAL_CODE_MODES:
            raise ValueError(
                f"internal_code must be one of {sorted(INTERNAL_CODE_MODES)}, got {internal_code!r}")
        if oov_code not in OOV_CODE_MODES:
            raise ValueError(
                f"oov_code must be one of {sorted(OOV_CODE_MODES)}, got {oov_code!r}")
        self.w2v = w2v
        self.type_vocab = type_vocab
        self.code_dim = w2v.vector_size
        self.internal_code = internal_code
        self.oov_code = oov_code
        self.code_mean = None if code_mean is None else np.asarray(code_mean, dtype=np.float32)
        self.code_std = None if code_std is None else np.asarray(code_std, dtype=np.float32)
        if oov_vector is not None:
            self.oov_vector = np.asarray(oov_vector, dtype=np.float32)
        elif oov_code == "random":
            self.oov_vector = self._make_oov_vector(seed)
        else:
            self.oov_vector = np.zeros(self.code_dim, dtype=np.float32)

    def _make_oov_vector(self, seed: int) -> np.ndarray:
        """A single fixed vector at the corpus's typical magnitude, so OOV leaves are distinct
        from internal nodes without being outliers."""
        vectors = self.w2v.wv.vectors
        scale = (float(np.linalg.norm(vectors, axis=1).mean()) / np.sqrt(self.code_dim)
                 if len(vectors) else 1.0)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, self.code_dim).astype(np.float32)

    def leaf_vector(self, token: str) -> np.ndarray:
        wv = self.w2v.wv
        return wv[token] if token in wv else self.oov_vector

    def code_vector(self, nodes: list[ASTNode], node_id: int) -> np.ndarray:
        tokens = collect_leaf_tokens(nodes, node_id)
        if not tokens:
            return np.zeros(self.code_dim, dtype=np.float32)
        return np.mean([self.leaf_vector(t) for t in tokens], axis=0).astype(np.float32)

    def type_id(self, node: ASTNode) -> int:
        return self.type_vocab.get(node.type)

    # -- standardization -------------------------------------------------------------------

    def fit_standardizer(self, graphs) -> None:
        """Fit per-dimension mean/std over the given graphs' node code vectors.

        Call with the TRAINING graphs only -- val/test statistics must not leak in. Accumulated in
        a streaming pass so the whole corpus's node matrix is never materialised.
        """
        if self.internal_code != "mean_standardized":
            return
        n = 0
        total = np.zeros(self.code_dim, dtype=np.float64)
        total_sq = np.zeros(self.code_dim, dtype=np.float64)
        for g in graphs:
            mat = self._featurize_mean(g.nodes).astype(np.float64)
            n += mat.shape[0]
            total += mat.sum(axis=0)
            total_sq += (mat * mat).sum(axis=0)
        if n == 0:
            return
        mean = total / n
        std = np.sqrt(np.maximum(total_sq / n - mean * mean, 0.0))
        # A dimension with no variance carries no information; dividing by ~0 would turn float
        # noise into huge features, so leave those dimensions on their original scale.
        std[std < 1e-6] = 1.0
        self.code_mean = mean.astype(np.float32)
        self.code_std = std.astype(np.float32)

    # -- featurization ---------------------------------------------------------------------

    def featurize_graph(self, graph: CodeGraph) -> tuple[np.ndarray, np.ndarray]:
        """Returns (code_matrix [m, code_dim] float32, type_ids [m] int64)."""
        nodes = graph.nodes
        type_ids = np.array([self.type_id(n) for n in nodes], dtype=np.int64)

        if self.internal_code == "zero":
            # Only leaves carry lexical content; internal nodes are identified by Type alone.
            code_mat = np.zeros((len(nodes), self.code_dim), dtype=np.float32)
            for nid, node in enumerate(nodes):
                if node.is_leaf:
                    code_mat[nid] = self.leaf_vector(node.code)
            return code_mat, type_ids

        code_mat = self._featurize_mean(nodes)
        if self.internal_code == "mean_standardized" and self.code_mean is not None:
            code_mat = (code_mat - self.code_mean) / self.code_std
        return code_mat.astype(np.float32), type_ids

    def _featurize_mean(self, nodes: list[ASTNode]) -> np.ndarray:
        """Subtree mean-pooling.

        Computed bottom-up in a single O(m) pass instead of per-node leaf collection (which would
        be O(m) per node, i.e. O(m^2) per graph): since `nodes` is pre-order DFS, every child id is
        strictly greater than its parent's, so iterating ids in reverse processes every child
        before its parent, letting each node simply sum its direct children's (sum, count).
        """
        m = len(nodes)
        sums = np.zeros((m, self.code_dim), dtype=np.float32)
        counts = np.zeros(m, dtype=np.int64)
        for nid in range(m - 1, -1, -1):
            node = nodes[nid]
            if node.is_leaf:
                # OOV leaves contribute their dedicated vector and DO count, so a subtree made
                # entirely of rare identifiers is still distinguishable from an empty one.
                sums[nid] = self.leaf_vector(node.code)
                counts[nid] = 1
            else:
                for cid in node.children:
                    sums[nid] += sums[cid]
                    counts[nid] += counts[cid]
        safe_counts = np.maximum(counts, 1).reshape(-1, 1)
        code_mat = (sums / safe_counts).astype(np.float32)
        code_mat[counts == 0] = 0.0
        return code_mat

    # -- persistence -----------------------------------------------------------------------

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.w2v.save(os.path.join(out_dir, "word2vec.model"))
        self.type_vocab.save(os.path.join(out_dir, "type_vocab.json"))
        # Persisted so inference featurizes exactly the way training did -- including the fitted
        # standardization statistics, without which inference would be on a different scale.
        with open(os.path.join(out_dir, "featurizer.json"), "w", encoding="utf-8") as f:
            json.dump({
                "internal_code": self.internal_code,
                "oov_code": self.oov_code,
                "oov_vector": self.oov_vector.tolist(),
                "code_mean": None if self.code_mean is None else self.code_mean.tolist(),
                "code_std": None if self.code_std is None else self.code_std.tolist(),
            }, f)

    @classmethod
    def load(cls, out_dir: str) -> "NodeFeaturizer":
        w2v = Word2Vec.load(os.path.join(out_dir, "word2vec.model"))
        type_vocab = TypeVocab.load(os.path.join(out_dir, "type_vocab.json"))
        meta_path = os.path.join(out_dir, "featurizer.json")
        # Featurizers saved before these options existed used mean-pooling and zero for OOV.
        meta = {"internal_code": "mean", "oov_code": "zero"}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta.update(json.load(f))
        return cls(w2v, type_vocab,
                   internal_code=meta["internal_code"], oov_code=meta["oov_code"],
                   code_mean=meta.get("code_mean"), code_std=meta.get("code_std"),
                   oov_vector=meta.get("oov_vector"))
