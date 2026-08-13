"""C source -> AST parsing via tree-sitter.

Joern (the tool used in the original paper) requires a separate JVM/Scala toolchain and is not
used here; tree-sitter gives us a fast, dependency-light AST that is sufficient to derive the
AST/CFG/DFG/NCS subgraphs described in Sec 2.2.2 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import tree_sitter_c
from tree_sitter import Language, Node, Parser

_LANGUAGE = Language(tree_sitter_c.language())
_PARSER = Parser(_LANGUAGE)

# Tree-sitter node types that should NOT be flattened into the graph (punctuation / anonymous
# tokens with no semantic value for vulnerability patterns), keeps graphs smaller and cleaner.
_SKIP_LEAF_TYPES = {
    "(", ")", "{", "}", "[", "]", ";", ",", ":", ".", "...",
}


@dataclass
class ASTNode:
    id: int
    type: str
    code: str
    start_byte: int
    end_byte: int
    is_leaf: bool
    parent: Optional[int] = None
    children: list = field(default_factory=list)


def parse_source(source: str) -> Node:
    """Parse a single C function (or snippet) and return the tree-sitter root node."""
    tree = _PARSER.parse(source.encode("utf-8"))
    return tree.root_node


def flatten_ast(source: str, return_error: bool = False):
    """Walk a tree-sitter tree and produce a flat list of ASTNode with parent/child links.

    Node ids are assigned in pre-order DFS, which conveniently keeps them clustered in source
    order so that downstream NCS-edge construction (Sec "Natural Code Sequence") is a simple
    by-position sort of the leaf nodes.

    Returns (nodes, src_bytes), or (nodes, src_bytes, has_error) when `return_error` is set --
    the flag comes from the parse we already did, so callers that filter on syntax errors (the
    paper drops functions Joern could not parse, Sec 3.1) don't pay for a second parse.
    """
    src_bytes = source.encode("utf-8")
    root = parse_source(source)
    had_error = root.has_error

    nodes: list[ASTNode] = []

    def visit(ts_node: Node, parent_id: Optional[int]) -> int:
        text = src_bytes[ts_node.start_byte:ts_node.end_byte].decode("utf-8", errors="replace")
        # Collapse long subtrees' text (e.g. whole compound_statement) to a short label; only
        # leaf tokens need the literal code text for the word2vec corpus.
        is_leaf = len(ts_node.children) == 0
        display_code = text if is_leaf else ts_node.type
        node_id = len(nodes)
        node = ASTNode(
            id=node_id,
            type=ts_node.type,
            code=display_code.strip() or ts_node.type,
            start_byte=ts_node.start_byte,
            end_byte=ts_node.end_byte,
            is_leaf=is_leaf,
            parent=parent_id,
        )
        nodes.append(node)
        if parent_id is not None:
            nodes[parent_id].children.append(node_id)
        for child in ts_node.children:
            if is_leaf:
                continue
            # Skip pure-punctuation leaves entirely (do not even materialize them as nodes).
            if len(child.children) == 0:
                child_text = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                if child_text in _SKIP_LEAF_TYPES:
                    continue
            visit(child, node_id)
        return node_id

    visit(root, None)
    if return_error:
        return nodes, src_bytes, had_error
    return nodes, src_bytes


def has_syntax_error(source: str) -> bool:
    root = parse_source(source)
    return root.has_error
