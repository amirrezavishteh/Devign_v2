"""Generic training loop shared by the graph models (Devign/Ggrn) and the sequence baselines.

Implements the paper's training configuration (Sec 3.3): Adam, lr=1e-4, batch_size=128, L2
regularization via weight_decay, and early stopping with patience=100 epochs on validation F1.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import binary_metrics


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 100
    l2_weight: float = 1e-4
    grad_clip: float = 5.0
    device: str = "cpu"
    monitor: str = "f1"   # early-stopping metric on validation
    pos_weight: float | None = None
    # When the loader yields node-budget micro-batches (see data.dataset.BucketBySizeSampler),
    # accumulate gradients until `batch_size` graphs have been seen before stepping, so the
    # effective batch size stays the paper's 128 no matter how the micro-batches fall out.
    accumulate_to_batch_size: bool = True


def _move_batch(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


def _labels_of(batch):
    if hasattr(batch, "labels"):
        return batch.labels
    return batch["labels"]


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_probs, all_labels, all_projects = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(_labels_of(batch).cpu().numpy())
        projects = batch.projects if hasattr(batch, "projects") else batch["projects"]
        all_projects.extend(projects)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)
    metrics = binary_metrics(labels, preds)
    return metrics, probs, labels, all_projects


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                cfg: TrainConfig, verbose: bool = True):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)
    pos_weight = torch.tensor([cfg.pos_weight], device=device) if cfg.pos_weight else None
    # `sum` reduction so gradient accumulation across unequal micro-batches weights each GRAPH
    # equally; we divide by the number of graphs actually accumulated before stepping.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                     reduction="sum" if cfg.accumulate_to_batch_size else "mean")

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pending = 0      # graphs accumulated since the last optimizer step
        n_graphs = 0
        optimizer.zero_grad(set_to_none=True)

        def _step(pending_graphs: int):
            """Rescale accumulated grads to a mean over `pending_graphs`, clip, step."""
            if pending_graphs <= 0:
                return
            if cfg.accumulate_to_batch_size:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= pending_graphs
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            batch = _move_batch(batch, device)
            labels = _labels_of(batch).to(device)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()

            bs = labels.shape[0]
            total_loss += loss.item() / bs if cfg.accumulate_to_batch_size else loss.item()
            n_batches += 1
            n_graphs += bs

            if not cfg.accumulate_to_batch_size:
                _step(1)
            else:
                pending += bs
                if pending >= cfg.batch_size:
                    _step(pending)
                    pending = 0
        _step(pending)  # flush the epoch's trailing partial batch

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        score = val_metrics[cfg.monitor]
        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or improved):
            print(f"  epoch {epoch:3d} | loss {total_loss / max(1, n_batches):.4f} "
                  f"| val acc {val_metrics['accuracy']:.2f} f1 {val_metrics['f1']:.2f} "
                  f"| best {cfg.monitor} {best_score:.2f}")

        if epochs_no_improve >= cfg.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no improvement for {cfg.patience})")
            break

    model.load_state_dict(best_state)
    return model, best_metrics
