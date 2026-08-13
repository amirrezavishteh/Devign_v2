"""Train the Table-2 baselines (BiLSTM, BiLSTM+Att, CNN, Metrics+XGBoost) on the prepared data.

The sequence baselines (BiLSTM/CNN) reuse the cached token sequences + word2vec embedding matrix
from data/prepare.py. Metrics+XGBoost re-featurizes the persisted train/val function split.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.baselines import (BiLSTMClassifier, CNNClassifier, SequenceDataset,
                             make_seq_collate)
from training.metrics import binary_metrics
from training.trainer import TrainConfig, evaluate, train_model


def _filter_project(samples, project):
    if not project or project == "combined":
        return samples
    return [s for s in samples if s.project == project]


def _seq_loaders(cfg, project=None):
    proc = cfg["data"]["processed_dir"]
    with open(os.path.join(proc, "sequences.pkl"), "rb") as f:
        seq = pickle.load(f)
    train_samples = _filter_project(seq["train"], project)
    val_samples = _filter_project(seq["val"], project)
    collate = make_seq_collate(seq["max_len"])
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(SequenceDataset(train_samples, seq["max_len"]),
                              batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SequenceDataset(val_samples, seq["max_len"]),
                            batch_size=bs, shuffle=False, collate_fn=collate)
    return {**seq, "train": train_samples, "val": val_samples}, train_loader, val_loader


def _train_config(cfg, device, pos_weight):
    tr = cfg["training"]
    return TrainConfig(
        lr=tr["lr"], batch_size=tr["batch_size"], epochs=tr["epochs"],
        patience=tr["early_stopping_patience"], l2_weight=tr["l2_weight"],
        grad_clip=tr["grad_clip"], device=device, pos_weight=pos_weight,
    )


def _pos_weight_seq(samples):
    n_pos = sum(s.label for s in samples)
    n_neg = len(samples) - n_pos
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def train_bilstm(cfg, device, attention: bool, verbose=True, project=None):
    seq, train_loader, val_loader = _seq_loaders(cfg, project)
    bcfg = cfg["baselines"]["bilstm_att" if attention else "bilstm"]
    model = BiLSTMClassifier(
        vocab_size=seq["vocab_size"], embed_dim=seq["embed"].shape[1],
        hidden_dim=bcfg["hidden_dim"], layers=bcfg["layers"], attention=attention,
        dropout=cfg["model"]["dropout"], pretrained=seq["embed"],
    )
    pw = _pos_weight_seq(seq["train"])
    model, best = train_model(model, train_loader, val_loader,
                              _train_config(cfg, device, pw), verbose=verbose)
    return model, best


def train_cnn(cfg, device, verbose=True, project=None):
    seq, train_loader, val_loader = _seq_loaders(cfg, project)
    ccfg = cfg["baselines"]["cnn"]
    model = CNNClassifier(
        vocab_size=seq["vocab_size"], embed_dim=seq["embed"].shape[1],
        num_filters=ccfg["num_filters"], filter_sizes=tuple(ccfg["filter_sizes"]),
        dropout=cfg["model"]["dropout"], pretrained=seq["embed"],
    )
    pw = _pos_weight_seq(seq["train"])
    model, best = train_model(model, train_loader, val_loader,
                              _train_config(cfg, device, pw), verbose=verbose)
    return model, best


def train_metrics_xgboost(cfg, verbose=True, project=None):
    from models.metrics_xgboost import featurize_functions, train_xgboost
    proc = cfg["data"]["processed_dir"]
    with open(os.path.join(proc, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    train_funcs = _filter_project(splits["train"], project)
    val_funcs = _filter_project(splits["val"], project)
    Xtr, ytr, _ = featurize_functions(train_funcs)
    Xva, yva, proj_va = featurize_functions(val_funcs)
    xcfg = cfg["baselines"]["metrics_xgboost"]
    model = train_xgboost(Xtr, ytr, n_estimators=xcfg["n_estimators"],
                          max_depth=xcfg["max_depth"], bayes_opt_iters=xcfg["bayes_opt_iters"])
    preds = model.predict(Xva)
    metrics = binary_metrics(yva, preds)
    if verbose:
        print(f"  metrics+xgboost val: acc {metrics['accuracy']:.2f} f1 {metrics['f1']:.2f}")
    return model, metrics, (Xva, yva, proj_va, preds)
