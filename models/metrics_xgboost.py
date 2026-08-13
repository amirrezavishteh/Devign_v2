"""Metrics + XGBoost baseline [25] (Leopard-style program metrics + gradient boosting).

The paper collects 4 complexity metrics and 11 vulnerability metrics per function via Joern. We
compute an analogous 15-dim hand-crafted feature vector directly from the tree-sitter graph
(cyclomatic complexity, nesting depth, loop/branch counts, pointer/alloc/free usage, etc.). If
xgboost is installed it is used with optional Bayesian hyperparameter search (scikit-optimize);
otherwise we fall back to sklearn's GradientBoostingClassifier so the pipeline always runs.
"""
from __future__ import annotations

import numpy as np

from data.graph_builder import CodeGraph, build_graph

_LOOP_TYPES = {"for_statement", "while_statement", "do_statement"}
_BRANCH_TYPES = {"if_statement", "case_statement", "switch_statement"}
_CALL_TYPE = "call_expression"

FEATURE_NAMES = [
    "num_nodes", "num_ast_edges", "num_cfg_edges", "num_dfg_edges",   # complexity
    "cyclomatic", "max_nesting", "num_loops", "num_branches",
    "num_calls", "num_returns", "num_assignments", "num_pointers",
    "num_alloc", "num_free", "num_identifiers",                        # vulnerability metrics
]


def _max_nesting(graph: CodeGraph) -> int:
    depth = {}
    best = 0
    for n in graph.nodes:
        if n.parent is None:
            depth[n.id] = 0
        else:
            inc = 1 if graph.nodes[n.parent].type in (_LOOP_TYPES | _BRANCH_TYPES) else 0
            depth[n.id] = depth.get(n.parent, 0) + inc
        best = max(best, depth[n.id])
    return best


def extract_features(graph: CodeGraph) -> np.ndarray:
    types = [n.type for n in graph.nodes]
    codes = [n.code for n in graph.nodes]
    num_loops = sum(t in _LOOP_TYPES for t in types)
    num_branches = sum(t in _BRANCH_TYPES for t in types)
    num_calls = sum(t == _CALL_TYPE for t in types)
    dfg = len(graph.edges.get("DFG_R", [])) + len(graph.edges.get("DFG_W", [])) + \
        len(graph.edges.get("DFG_C", []))
    alloc = sum(1 for c in codes if c in ("malloc", "calloc", "realloc", "alloca"))
    free = sum(1 for c in codes if c == "free")
    pointers = sum(1 for t in types if t in ("pointer_declarator", "pointer_expression"))
    feats = [
        len(graph.nodes),
        len(graph.edges.get("AST", [])),
        len(graph.edges.get("CFG", [])),
        dfg,
        num_branches + num_loops + 1,                       # cyclomatic complexity approximation
        _max_nesting(graph),
        num_loops,
        num_branches,
        num_calls,
        sum(t == "return_statement" for t in types),
        sum(t == "assignment_expression" for t in types),
        pointers,
        alloc,
        free,
        sum(t == "identifier" for t in types),
    ]
    return np.array(feats, dtype=np.float32)


def featurize_functions(functions) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, projects = [], [], []
    for fn in functions:
        g = build_graph(fn.func, max_nodes=500)
        if g is None:
            continue
        X.append(extract_features(g))
        y.append(int(fn.target))
        projects.append(fn.project)
    return np.stack(X), np.array(y), projects


def train_xgboost(X_train, y_train, n_estimators=200, max_depth=6, bayes_opt_iters=0):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4,
        )
        if bayes_opt_iters and bayes_opt_iters > 0:
            try:
                from skopt import BayesSearchCV
                from skopt.space import Integer, Real
                search = BayesSearchCV(
                    model,
                    {
                        "max_depth": Integer(3, 10),
                        "n_estimators": Integer(100, 400),
                        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                    },
                    n_iter=bayes_opt_iters, cv=3, n_jobs=4, random_state=42,
                )
                search.fit(X_train, y_train)
                return search.best_estimator_
            except Exception:
                pass
        model.fit(X_train, y_train)
        return model
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
