"""Pin the tree-sitter-c grammar by its OBSERVABLE behaviour, not just its version string.

Why this test exists
--------------------
Node *types* are one half of every node feature: `NodeInit` embeds a label-encoded `ASTNode.type`
and concatenates it with the word2vec code vector. A grammar bump that renames, splits or merges
node types therefore changes the type vocabulary, which changes the model -- silently, with no
error anywhere, and with results that are no longer comparable to a run from before the bump.

A pinned commit SHA would be the textbook guard, but the `tree-sitter-c` PyPI wheel does not
expose one. Hashing the grammar's actual output over a fixed canary is a stricter check than a SHA
anyway: it fails only when something that can move a number actually moved, and it fails loudly.

If this test fails after an intentional dependency upgrade: re-run the reproduction from
`scripts.prepare_data` onwards, then update the expected values here in the same commit, so the
change is recorded rather than absorbed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md

from devign_data.parser import flatten_ast

# A deliberately ordinary C function that exercises the constructs the graph builder keys on:
# declarations, a call, pointer arithmetic, a loop, a conditional and a return.
CANARY_SOURCE = """
int copy_prefix(char *dst, const char *src, int n) {
    int i = 0;
    if (dst == NULL || src == NULL) {
        return -1;
    }
    for (i = 0; i < n; i++) {
        dst[i] = src[i];
        if (src[i] == '\\0') {
            break;
        }
    }
    return i;
}
"""

# Measured on tree-sitter 0.25.2 + tree-sitter-c 0.24.2 (grammar ABI 15), the pair this
# reproduction's numbers were produced with.
EXPECTED_TYPE_HASH = "94906f22d4c565761da9275bf2cc4ee5b2041fb2fc21a63de1fa64e0ca202d6a"
EXPECTED_NODE_COUNT = 88


def _type_multiset_hash(source: str) -> str:
    """A stable digest of which node types the grammar produces, and how many of each."""
    nodes, _ = flatten_ast(source)
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.type] = counts.get(node.type, 0) + 1
    payload = ";".join(f"{t}:{counts[t]}" for t in sorted(counts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_grammar_versions_are_the_verified_pair():
    """The runtime/grammar pair the reproduction was measured on."""
    assert md.version("tree-sitter") == "0.25.2"
    assert md.version("tree-sitter-c") == "0.24.2"


def test_canary_node_type_vocabulary_is_unchanged():
    """A grammar bump that renames node types must fail here, not silently shift the model."""
    assert _type_multiset_hash(CANARY_SOURCE) == EXPECTED_TYPE_HASH


def test_canary_node_count_is_unchanged():
    nodes, _ = flatten_ast(CANARY_SOURCE)
    assert len(nodes) == EXPECTED_NODE_COUNT
