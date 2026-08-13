"""Q3 / Table 2 ablation: single-edge graphs vs the composite graph.

For each edge type in {AST, CFG, NCS, DFG_R, DFG_W, DFG_C} we train a Devign (and optionally
Ggrn) model whose graph batch only carries that one edge type, then compare to the composite
(all 7) model. We reuse the already-prepared graph samples and simply restrict the collate
function to the selected edge type(s).
"""
from __future__ import annotations

import os

from data.dataset import DevignDataset
from models.devign import build_model
from training.trainer import TrainConfig, train_model


def _loaders_for_edges(cfg, edge_types, project=None):
    from scripts.train import _loader, _subset, make_collate_from_cfg  # avoid an import cycle

    proc = cfg["data"]["processed_dir"]
    train_ds = _subset(DevignDataset.load(os.path.join(proc, "train.pkl")), project)
    val_ds = _subset(DevignDataset.load(os.path.join(proc, "val.pkl")), project)
    collate = make_collate_from_cfg(cfg, edge_types)
    return (train_ds, val_ds,
            _loader(train_ds, cfg, collate, shuffle=True),
            _loader(val_ds, cfg, collate, shuffle=False))


def run_single_edge(cfg, model_name, edge_type, device, epochs=None, verbose=False, project=None):
    edge_types = [edge_type]
    train_ds, val_ds, train_loader, val_loader = _loaders_for_edges(cfg, edge_types, project)
    model = build_model(model_name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                        type_vocab_size=train_ds.type_vocab_size, num_edge_types=1)
    tr = cfg["training"]
    n_pos = sum(s.label for s in train_ds.samples)
    pos_weight = ((len(train_ds) - n_pos) / n_pos) if n_pos > 0 else 1.0
    tcfg = TrainConfig(lr=tr["lr"], batch_size=tr["batch_size"],
                       epochs=epochs or tr["epochs"], patience=tr["early_stopping_patience"],
                       l2_weight=tr["l2_weight"], grad_clip=tr["grad_clip"], device=device,
                       pos_weight=pos_weight)
    model, best = train_model(model, train_loader, val_loader, tcfg, verbose=verbose)
    return best
