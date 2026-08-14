"""Builds the composite multi-edge graph g(V, X, A) described in Sec 2.2.2 of the Devign paper.

V = V_ast (every AST node). On top of the AST backbone we derive, heuristically:
  - REV_AST : reverse of the AST parent->child edge (lets messages flow up the tree too; see
              the note in config.yaml on why k=7 adjacency matrices are used).
  - CFG     : approximate control-flow graph built by a recursive structural walk over
              if/while/for/do/switch/break/continue/goto/return constructs.
  - NCS     : natural code sequence, connecting AST leaf nodes left-to-right by source position.
  - DFG_R/DFG_W/DFG_C : LastRead / LastWrite / ComputedFrom edges (Allamanis et al. 2017,
              cited as [27] in the paper) computed via a single linear pass over identifier
              occurrences in source order.

This is a lightweight stand-in for Joern's code-property-graph extraction (which needs a
separate JVM/Scala toolchain). It is not a byte-for-byte reproduction of Joern's CFG/DFG, but it
captures the same structural intuition and is sufficient to drive the rest of the Devign pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from data.parser import ASTNode, flatten_ast

EDGE_TYPES = ["AST", "REV_AST", "CFG", "NCS", "DFG_R", "DFG_W", "DFG_C"]

_DECLARATOR_WRAPPERS = {"pointer_declarator", "array_declarator"}
_FOR_WHILE_KEYWORDS = {"for", "while", "do"}


@dataclass
class CodeGraph:
    nodes: list[ASTNode]
    edges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


# ----------------------------------------------------------------------------------------------
# AST / REV_AST / NCS
# ----------------------------------------------------------------------------------------------

def _build_ast_edges(nodes: list[ASTNode]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ast_edges, rev_edges = [], []
    for n in nodes:
        if n.parent is not None:
            ast_edges.append((n.parent, n.id))
            rev_edges.append((n.id, n.parent))
    return ast_edges, rev_edges


def _build_ncs_edges(nodes: list[ASTNode]) -> list[tuple[int, int]]:
    leaves = sorted((n for n in nodes if n.is_leaf), key=lambda n: n.start_byte)
    return [(leaves[i].id, leaves[i + 1].id) for i in range(len(leaves) - 1)]


# ----------------------------------------------------------------------------------------------
# CFG
# ----------------------------------------------------------------------------------------------

class _CFGBuilder:
    def __init__(self, nodes: list[ASTNode]):
        self.nodes = nodes
        self.edges: list[tuple[int, int]] = []
        self.labels: dict[str, int] = {}
        self._collect_labels()

    def _collect_labels(self) -> None:
        for n in self.nodes:
            if n.type == "labeled_statement":
                for cid in n.children:
                    if self.nodes[cid].type == "statement_identifier":
                        self.labels[self.nodes[cid].code] = n.id
                        break

    def build(self, body_id: int) -> tuple[Optional[int], list[int]]:
        return self._stmt(body_id, break_stack=[], continue_stack=[])

    def _seq(self, stmt_ids: list[int], break_stack, continue_stack) -> tuple[Optional[int], list[int]]:
        first_entry = None
        prev_exits: list[int] = []
        for sid in stmt_ids:
            s_entry, s_exits = self._stmt(sid, break_stack, continue_stack)
            if s_entry is None:
                continue
            if first_entry is None:
                first_entry = s_entry
            for pe in prev_exits:
                self.edges.append((pe, s_entry))
            prev_exits = s_exits
        return first_entry, prev_exits

    def _stmt(self, node_id: int, break_stack, continue_stack) -> tuple[Optional[int], list[int]]:
        node = self.nodes[node_id]
        t = node.type
        if t == "compound_statement":
            return self._seq(node.children, break_stack, continue_stack)
        if t == "if_statement":
            return self._if(node, break_stack, continue_stack)
        if t == "while_statement":
            return self._while(node, break_stack, continue_stack)
        if t == "for_statement":
            return self._for(node, break_stack, continue_stack)
        if t == "do_statement":
            return self._do(node, break_stack, continue_stack)
        if t == "switch_statement":
            return self._switch(node, break_stack, continue_stack)
        if t == "labeled_statement":
            return self._labeled(node, break_stack, continue_stack)
        if t == "goto_statement":
            label_name = next((self.nodes[c].code for c in node.children
                                if self.nodes[c].type == "statement_identifier"), None)
            target = self.labels.get(label_name)
            if target is not None:
                self.edges.append((node.id, target))
            return node.id, []
        if t == "break_statement":
            if break_stack:
                break_stack[-1].append(node.id)
            return node.id, []
        if t == "continue_statement":
            if continue_stack:
                self.edges.append((node.id, continue_stack[-1]))
            return node.id, []
        if t == "return_statement":
            return node.id, []
        return node.id, [node.id]

    def _if(self, node, break_stack, continue_stack):
        cond = None
        others = []
        for cid in node.children:
            c = self.nodes[cid]
            if c.type == "parenthesized_expression" and cond is None:
                cond = c.id
            elif c.type in ("if", "else"):
                continue
            else:
                others.append(c)
        if cond is None:
            cond = node.id
        exits: list[int] = []
        consequence = others[0] if others else None
        alt = others[1] if len(others) > 1 else None
        if consequence is not None:
            c_entry, c_exits = self._stmt(consequence.id, break_stack, continue_stack)
            if c_entry is not None:
                self.edges.append((cond, c_entry))
                exits.extend(c_exits)
        if alt is not None:
            if alt.type == "else_clause":
                inner = [c for c in alt.children if self.nodes[c].type != "else"]
                if inner:
                    a_entry, a_exits = self._stmt(inner[0], break_stack, continue_stack)
                    if a_entry is not None:
                        self.edges.append((cond, a_entry))
                        exits.extend(a_exits)
            else:
                a_entry, a_exits = self._stmt(alt.id, break_stack, continue_stack)
                if a_entry is not None:
                    self.edges.append((cond, a_entry))
                    exits.extend(a_exits)
        else:
            exits.append(cond)
        return cond, exits

    def _while(self, node, break_stack, continue_stack):
        cond, body = None, None
        for cid in node.children:
            c = self.nodes[cid]
            if c.type == "parenthesized_expression":
                cond = c.id
            elif c.type != "while":
                body = c.id
        if cond is None:
            cond = node.id
        break_acc: list[int] = []
        break_stack.append(break_acc)
        continue_stack.append(cond)
        if body is not None:
            b_entry, b_exits = self._stmt(body, break_stack, continue_stack)
            if b_entry is not None:
                self.edges.append((cond, b_entry))
            for be in b_exits:
                self.edges.append((be, cond))
        break_stack.pop()
        continue_stack.pop()
        return cond, [cond] + break_acc

    def _for(self, node, break_stack, continue_stack):
        children = [c for c in node.children if self.nodes[c].type != "for"]
        if not children:
            return None, []
        body_id = children[-1]
        header = children[:-1]
        entry = None
        prev: list[int] = []
        for h in header:
            if entry is None:
                entry = h
            for p in prev:
                self.edges.append((p, h))
            prev = [h]
        cond = header[-1] if header else node.id
        if entry is None:
            entry = cond
        break_acc: list[int] = []
        break_stack.append(break_acc)
        continue_stack.append(cond)
        b_entry, b_exits = self._stmt(body_id, break_stack, continue_stack)
        if b_entry is not None:
            self.edges.append((cond, b_entry))
        for be in b_exits:
            self.edges.append((be, cond))
        break_stack.pop()
        continue_stack.pop()
        return entry, [cond] + break_acc

    def _do(self, node, break_stack, continue_stack):
        others = [c for c in node.children if self.nodes[c].type not in _FOR_WHILE_KEYWORDS]
        body_id = others[0] if others else None
        cond = next((c for c in others if self.nodes[c].type == "parenthesized_expression"), node.id)
        break_acc: list[int] = []
        break_stack.append(break_acc)
        continue_stack.append(cond)
        entry = None
        if body_id is not None:
            entry, b_exits = self._stmt(body_id, break_stack, continue_stack)
            for be in b_exits:
                self.edges.append((be, cond))
        if entry is None:
            entry = cond
        self.edges.append((cond, entry))
        break_stack.pop()
        continue_stack.pop()
        return entry, [cond] + break_acc

    def _switch(self, node, break_stack, continue_stack):
        cond = next((c for c in node.children if self.nodes[c].type == "parenthesized_expression"), None)
        body = next((c for c in node.children if self.nodes[c].type == "compound_statement"), None)
        if cond is None:
            cond = node.id
        break_acc: list[int] = []
        break_stack.append(break_acc)
        case_entries: list[int] = []
        prev_exits: list[int] = []
        last_exits: list[int] = []
        for cid in (self.nodes[body].children if body is not None else []):
            cnode = self.nodes[cid]
            if cnode.type == "case_statement":
                stmt_kids = []
                seen_label = False
                for kid in cnode.children:
                    knode = self.nodes[kid]
                    if knode.type in ("case", "default"):
                        continue
                    if not seen_label and knode.type not in (
                        "compound_statement", "if_statement", "while_statement", "for_statement",
                        "do_statement", "switch_statement", "return_statement", "break_statement",
                        "continue_statement", "expression_statement", "declaration",
                        "labeled_statement", "goto_statement",
                    ):
                        seen_label = True
                        continue
                    stmt_kids.append(kid)
                s_entry, s_exits = self._seq(stmt_kids, break_stack, continue_stack)
                case_entry = s_entry if s_entry is not None else cid
            else:
                s_entry, s_exits = self._stmt(cid, break_stack, continue_stack)
                case_entry = s_entry if s_entry is not None else cid
            case_entries.append(case_entry)
            for pe in prev_exits:
                self.edges.append((pe, case_entry))
            prev_exits = s_exits
            last_exits = s_exits
        for ce in case_entries:
            self.edges.append((cond, ce))
        break_stack.pop()
        exits = break_acc + (last_exits if case_entries else [cond])
        return cond, exits

    def _labeled(self, node, break_stack, continue_stack):
        inner = next((c for c in node.children if self.nodes[c].type != "statement_identifier"), None)
        if inner is None:
            return node.id, [node.id]
        i_entry, i_exits = self._stmt(inner, break_stack, continue_stack)
        if i_entry is not None:
            self.edges.append((node.id, i_entry))
        return node.id, i_exits


def _build_cfg_edges(nodes: list[ASTNode]) -> list[tuple[int, int]]:
    func_def = next((n for n in nodes if n.type == "function_definition"), None)
    if func_def is None:
        return []
    body = next((nodes[c] for c in func_def.children if nodes[c].type == "compound_statement"), None)
    if body is None:
        return []
    builder = _CFGBuilder(nodes)
    builder.build(body.id)
    return builder.edges


# ----------------------------------------------------------------------------------------------
# DFG_R / DFG_W / DFG_C  (LastRead / LastWrite / ComputedFrom, following Allamanis et al. 2017)
# ----------------------------------------------------------------------------------------------

def _classify_identifier(nodes: list[ASTNode], node: ASTNode) -> str:
    """Return 'write' or 'read' for an `identifier` leaf based on its syntactic role."""
    parent = nodes[node.parent] if node.parent is not None else None
    if parent is None:
        return "read"
    if parent.type in ("init_declarator", "assignment_expression"):
        return "write" if parent.children and parent.children[0] == node.id else "read"
    if parent.type == "update_expression":
        return "write"
    if parent.type in ("parameter_declaration", "declaration"):
        return "write"
    if parent.type in _DECLARATOR_WRAPPERS:
        return _classify_identifier(nodes, parent) if parent.parent is not None else "write"
    return "read"


def _collect_identifier_descendants(nodes: list[ASTNode], node_id: int, exclude: int) -> list[int]:
    out = []
    node = nodes[node_id]
    if node.id != exclude and node.type == "identifier":
        out.append(node.id)
    for cid in node.children:
        out.extend(_collect_identifier_descendants(nodes, cid, exclude))
    return out


def _build_dfg_edges(nodes: list[ASTNode]) -> dict[str, list[tuple[int, int]]]:
    dfg_r: list[tuple[int, int]] = []
    dfg_w: list[tuple[int, int]] = []
    dfg_c: list[tuple[int, int]] = []

    last_write: dict[str, int] = {}
    last_read: dict[str, int] = {}

    identifiers = sorted((n for n in nodes if n.type == "identifier"), key=lambda n: n.start_byte)

    for ident in identifiers:
        var = ident.code
        role = _classify_identifier(nodes, ident)
        parent = nodes[ident.parent] if ident.parent is not None else None

        if role == "write":
            if parent is not None and parent.type in ("init_declarator", "assignment_expression") and \
                    parent.children and parent.children[0] == ident.id:
                rhs_ids = parent.children[1:]
                for rid in rhs_ids:
                    for leaf_id in _collect_identifier_descendants(nodes, rid, exclude=-1):
                        dfg_c.append((ident.id, leaf_id))
            last_write[var] = ident.id
        else:
            if var in last_write:
                dfg_w.append((ident.id, last_write[var]))
            if var in last_read:
                dfg_r.append((ident.id, last_read[var]))
            last_read[var] = ident.id

    return {"DFG_R": dfg_r, "DFG_W": dfg_w, "DFG_C": dfg_c}


# ----------------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------------

def build_graph(source: str, max_nodes: Optional[int] = None,
                drop_parse_errors: bool = True,
                max_error_fraction: float = 0.2) -> Optional[CodeGraph]:
    """Parse a C function and build its composite multi-edge graph.

    Returns None if the snippet exceeds `max_nodes` (the paper filters out functions with > 500
    nodes, ~15% of the raw data, Sec 3.1) or fails to parse ("We filter out those functions
    without ASTs and CFGs or with oblivious errors in ASTs and CFGs", Sec 3.1).

    "Oblivious errors" means gross failures, not any error at all -- so the parse filter rejects a
    function only when no `function_definition` ever formed, or when ERROR nodes span more than
    `max_error_fraction` of the source. Rejecting on *any* ERROR node discarded 12.1% of the real
    corpus (~3,300 functions, measured), overwhelmingly ordinary C whose macros or GNU extensions
    tree-sitter flags locally while still recovering the surrounding structure correctly.
    """
    return build_graph_with_reason(source, max_nodes, drop_parse_errors, max_error_fraction)[0]


def build_graph_with_reason(source: str, max_nodes: Optional[int] = None,
                            drop_parse_errors: bool = True,
                            max_error_fraction: float = 0.2):
    """Like `build_graph`, but also returns why a function was rejected.

    Reason is one of None (kept), "empty", "parse", "size". data/prepare.py reports these
    separately so the two filters stay visible: conflating them previously hid the fact that the
    parse filter alone was discarding 12.1% of the corpus.
    """
    nodes, _, parse_info = flatten_ast(source, return_error=True)
    if not nodes:
        return None, "empty"
    if drop_parse_errors and not parse_info.is_usable(max_error_fraction):
        return None, "parse"
    if max_nodes is not None and len(nodes) > max_nodes:
        return None, "size"

    ast_edges, rev_ast_edges = _build_ast_edges(nodes)
    ncs_edges = _build_ncs_edges(nodes)
    cfg_edges = _build_cfg_edges(nodes)
    dfg_edges = _build_dfg_edges(nodes)

    edges = {
        "AST": ast_edges,
        "REV_AST": rev_ast_edges,
        "CFG": cfg_edges,
        "NCS": ncs_edges,
        **dfg_edges,
    }
    return CodeGraph(nodes=nodes, edges=edges), None
