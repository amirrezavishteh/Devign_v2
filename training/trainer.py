"""Generic training loop shared by the graph models (Devign/Ggrn) and the sequence baselines.

Implements the paper's training configuration (Sec 3.3): Adam, lr=1e-4, batch_size=128, L2
regularization via weight_decay, and early stopping with patience=100 epochs on validation F1.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import best_threshold, binary_metrics, prob_metrics


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 100
    l2_weight: float = 1e-4
    grad_clip: float = 5.0
    device: str = "cpu"
    # Model-selection metric on validation. Defaults to AUC, NOT F1: at a fixed 0.5 threshold on
    # this ~45%-positive data, a degenerate "almost everything is vulnerable" classifier scores
    # F1 ~60-67% and a genuinely trained model never beats it, so monitoring F1 restores an
    # early, untrained checkpoint. A constant predictor scores AUC 50, so AUC cannot be gamed
    # that way. See training/metrics.py.
    monitor: str = "auc"
    # Tune the decision threshold on validation after restoring the best checkpoint, and report
    # at that threshold instead of an arbitrary 0.5. See metrics.best_threshold for why the
    # objective is guarded F1 rather than plain F1 (degenerate cut) or MCC (tanks F1).
    tune_threshold: bool = True
    threshold_objective: str = "f1_guarded"
    pos_weight: float | None = None
    # Learning-rate schedule. "none" is the paper (Sec 3.3 fixes lr=1e-4 for the whole run);
    # "plateau" halves the LR when the monitored validation metric stalls.
    #
    # Why this exists: Eq. 9's readout multiplies two MLP outputs, so at init the whole batch's
    # logits span ~6e-4 and the gradient reaching the GGNN trunk is ~53x weaker than it would be
    # through a linear head (measured). At lr=1e-4 the first ~10 epochs are spent escaping that
    # dead zone rather than learning. A larger initial LR plus decay-on-plateau gets past it
    # without giving up the fine convergence the paper's small LR buys later.
    lr_schedule: str = "none"
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-6
    # When the loader yields node-budget micro-batches (see data.dataset.BucketBySizeSampler),
    # accumulate gradients until `batch_size` graphs have been seen before stepping, so the
    # effective batch size stays the paper's 128 no matter how the micro-batches fall out.
    accumulate_to_batch_size: bool = True
    # Per-epoch checkpoint. Set by the training entry points so that a crash (shared-GPU fault,
    # preemption, OOM) costs minutes rather than the whole model: re-running resumes from the
    # last completed epoch instead of epoch 1. None disables checkpointing.
    checkpoint_path: str | None = None
    # Skip a micro-batch that OOMs rather than aborting the run. The node-budget sampler bounds
    # memory, but a co-tenant grabbing VRAM mid-run can still push a single batch over.
    skip_oom_batches: bool = True


def make_train_config(cfg: dict, device: str, train_labels=None, epochs: int | None = None,
                      **overrides) -> TrainConfig:
    """Build a TrainConfig from config.yaml. Single source of truth for every training entry point.

    `train_labels` is an iterable of 0/1 labels used only when `training.class_weighting` is on.
    Class weighting defaults OFF: the paper does no reweighting, and on this ~45%-positive data it
    biases the model toward the all-positive regime that then wins checkpoint selection.
    """
    tr = cfg["training"]
    pos_weight = None
    if tr.get("class_weighting", False) and train_labels is not None:
        labels = [int(x) for x in train_labels]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    kwargs = dict(
        lr=tr["lr"], batch_size=tr["batch_size"],
        epochs=epochs or tr["epochs"], patience=tr["early_stopping_patience"],
        l2_weight=tr["l2_weight"], grad_clip=tr["grad_clip"], device=device,
        monitor=tr.get("monitor", "auc"), tune_threshold=tr.get("tune_threshold", True),
        threshold_objective=tr.get("threshold_objective", "f1_guarded"),
        pos_weight=pos_weight,
        lr_schedule=tr.get("lr_schedule", "none"), lr_factor=tr.get("lr_factor", 0.5),
        lr_patience=tr.get("lr_patience", 5), min_lr=tr.get("min_lr", 1e-6),
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


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
    metrics = prob_metrics(labels, probs, threshold)
    return metrics, probs, labels, all_projects


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                cfg: TrainConfig, verbose: bool = True):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)
    # Every metric `monitor` can name (auc/f1/accuracy/mcc) is higher-is-better, hence mode="max".
    scheduler = None
    if cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.min_lr)
    elif cfg.lr_schedule not in ("none", None):
        raise ValueError(f"unknown training.lr_schedule: {cfg.lr_schedule!r} (use none|plateau)")
    pos_weight = torch.tensor([cfg.pos_weight], device=device) if cfg.pos_weight else None
    # `sum` reduction so gradient accumulation across unequal micro-batches weights each GRAPH
    # equally; we divide by the number of graphs actually accumulated before stepping.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                     reduction="sum" if cfg.accumulate_to_batch_size else "mean")

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    epochs_no_improve = 0
    start_epoch = 1

    # Resume a run interrupted by a GPU fault / preemption / OOM.
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # Without this a resumed run silently restarts at the full LR, so the schedule depends on
        # how many times the job was preempted.
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_state = ckpt["best_state"]
        best_score = ckpt["best_score"]
        best_metrics = ckpt["best_metrics"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        start_epoch = ckpt["epoch"] + 1
        if verbose:
            print(f"  resumed from {cfg.checkpoint_path} at epoch {start_epoch} "
                  f"(best {cfg.monitor} {best_score:.2f})")

    def _save_checkpoint(epoch: int) -> None:
        if not cfg.checkpoint_path:
            return
        os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
        tmp = cfg.checkpoint_path + ".tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_state": best_state,
                    "best_score": best_score, "best_metrics": best_metrics,
                    "epochs_no_improve": epochs_no_improve,
                    "scheduler": scheduler.state_dict() if scheduler else None}, tmp)
        os.replace(tmp, cfg.checkpoint_path)   # atomic: a crash mid-write can't corrupt it

    for epoch in range(start_epoch, cfg.epochs + 1):
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
            try:
                logits = model(batch)
                loss = criterion(logits, labels)
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                if not cfg.skip_oom_batches:
                    raise
                # A co-tenant can grab VRAM mid-run; drop this micro-batch rather than lose
                # the whole training run. Accumulated grads stay valid for the next step.
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                torch.cuda.empty_cache()
                print("  [warn] skipped a micro-batch after CUDA OOM (freeing cache)")
                continue

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

        val_metrics, val_probs, _, _ = evaluate(model, val_loader, device)
        score = val_metrics[cfg.monitor]
        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(score)

        if verbose and (epoch % 5 == 0 or epoch == 1 or improved):
            # `spread` is the width of the predicted-probability band. Eq. 9's multiplicative
            # readout starts almost degenerate (~6e-4 wide), so a spread still under ~0.05 after a
            # few epochs means the head has not come alive and nothing downstream is meaningful.
            spread = float(val_probs.max() - val_probs.min()) if val_probs.size else 0.0
            print(f"  epoch {epoch:3d} | loss {total_loss / max(1, n_batches):.4f} "
                  f"| val acc {val_metrics['accuracy']:.2f} f1 {val_metrics['f1']:.2f} "
                  f"| spread {spread:.3f} | lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| best {cfg.monitor} {best_score:.2f}")

        _save_checkpoint(epoch)

        if epochs_no_improve >= cfg.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no improvement for {cfg.patience})")
            break

    model.load_state_dict(best_state)
    # Training completed normally; drop the resume checkpoint so a later re-run retrains cleanly
    # rather than resuming a finished model.
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        try:
            os.remove(cfg.checkpoint_path)
        except OSError:
            pass

    # Pick the operating point on validation using the restored (best) weights, then report at it.
    # Done after restoration so the threshold belongs to the checkpoint we actually ship.
    final_val, probs, labels, _ = evaluate(model, val_loader, device)
    threshold = (best_threshold(labels, probs, cfg.threshold_objective)
                 if cfg.tune_threshold else 0.5)
    best_metrics = prob_metrics(labels, probs, threshold)
    if verbose:
        print(f"  restored best ({cfg.monitor} {best_score:.2f}) | threshold {threshold:.3f} "
              f"-> val acc {best_metrics['accuracy']:.2f} f1 {best_metrics['f1']:.2f} "
              f"auc {best_metrics['auc']:.2f}")
    return model, best_metrics
