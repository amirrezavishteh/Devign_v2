"""Measure, at initialisation, what each readout does to the gradient reaching the GGNN trunk.

This exists so the Eq. 9 finding is reproducible rather than quoted. The paper's readout is

    y = Sigmoid( AVG( MLP(Z) (elemwise*) MLP(Y) ) )                            (Eq. 9)

Both MLP heads start near zero, so their product starts near zero squared, and because
d(z*y)/dz = y and d(z*y)/dy = z, BOTH branches are throttled by the other's smallness at once.
The consequence is not subtle: the whole batch's logits span ~1e-3, every probability sits on top
of 0.5, and the trunk barely learns until the head escapes on its own.

Four readouts are compared on the SAME trunk with the SAME initialisation and the SAME batch, so
the only thing varying is the readout:

    conv_affine_off  Eq. 9 exactly as written                 <- the paper
    conv_affine_on   Eq. 9 + the repo's learnable scale/bias  <- the fix under test
    ggrn_sum         Eq. 5, flat summation of per-node MLPs   <- the paper's own ablation arm
    linear_head      Linear on mean-pooled [H, x]             <- the reference: nothing clever

`linear_head` is the yardstick. It is the simplest thing that could possibly work, so the ratio
between its trunk gradient and Eq. 9's is the attenuation Eq. 9 imposes for free.

Usage:
    python -m scripts.measure_readout --config configs/a100_codexglue.yaml
    python -m scripts.measure_readout --config configs/smoke.yaml --json out.json
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from devign_data.graph_builder import EDGE_TYPES
from models.conv_module import ConvModule
from models.devign import _Trunk
from scripts.train import load_graph_loaders
from training.utils import load_config, resolve_device, seed_from_config


class _LinearHead(nn.Module):
    """The control arm: mean-pool [H, x] over real nodes, one Linear to a scalar."""

    def __init__(self, hidden_dim: int, init_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim + init_dim, 1)

    def forward(self, H, x, mask):
        node = torch.cat([H, x], dim=-1)
        m = mask.float().unsqueeze(-1)
        pooled = (node * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled).squeeze(-1)


class _GgrnSum(nn.Module):
    """Eq. 5: y = Sigmoid( SUM_j MLP([H_j, x_j]) ), masked to real nodes."""

    def __init__(self, hidden_dim: int, init_dim: int, mlp_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + init_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1))

    def forward(self, H, x, mask):
        node = torch.cat([H, x], dim=-1)
        return (self.mlp(node).squeeze(-1) * mask.float()).sum(dim=1)


def _trunk_grad_norm(trunk: nn.Module) -> float:
    total = 0.0
    for p in trunk.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def measure_one(name: str, head_factory, cfg: dict, batch, device: str, seed: int) -> dict:
    """Build a fresh trunk+head at `seed`, run one forward/backward, report the diagnostics.

    Re-seeding per variant is what makes the comparison fair: every arm gets a bit-identical
    trunk, so a difference in trunk gradient is caused by the readout and nothing else.
    """
    seed_from_config(cfg, seed)
    emb, m = cfg["embedding"], cfg["model"]
    trunk = _Trunk(
        code_dim=emb["word2vec_dim"], type_vocab_size=int(batch.type_ids.max()) + 1,
        type_dim=emb["type_dim"], num_edge_types=len(EDGE_TYPES),
        hidden_dim=m["hidden_dim"], time_steps=m["time_steps"],
        aggregation=m["aggregation"], type_init_std=emb.get("type_init_std", 1.0),
    ).to(device)
    head = head_factory(m["hidden_dim"], trunk.init_dim).to(device)

    trunk.train(); head.train()
    trunk.zero_grad(set_to_none=True)

    H, x = trunk(batch)
    logits = head(H, x, batch.mask)
    labels = batch.labels.to(device).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    with torch.no_grad():
        probs = torch.sigmoid(logits)
    return {
        "readout": name,
        "logit_min": float(logits.min()),
        "logit_max": float(logits.max()),
        "logit_range": float(logits.max() - logits.min()),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "prob_spread": float(probs.max() - probs.min()),
        "loss": float(loss),
        "trunk_grad_norm": _trunk_grad_norm(trunk),
    }


def measure(cfg: dict, device: str, batch, seed: int) -> list[dict]:
    m = cfg["model"]
    conv_cfg = dict(m["conv"])
    mlp_hidden = conv_cfg["mlp_hidden"]
    pos_rate = float(batch.labels.float().mean())

    def conv_factory(affine: bool):
        c = dict(conv_cfg)
        c["logit_affine"] = affine
        def make(hidden_dim, init_dim):
            return ConvModule(hidden_dim, init_dim, c, mlp_hidden,
                              dropout=m["dropout"], pos_rate=pos_rate if affine else None)
        return make

    variants = [
        ("conv_affine_off", conv_factory(False)),
        ("conv_affine_on", conv_factory(True)),
        ("ggrn_sum", lambda h, i: _GgrnSum(h, i, mlp_hidden)),
        ("linear_head", lambda h, i: _LinearHead(h, i)),
    ]
    return [measure_one(n, f, cfg, batch, device, seed) for n, f in variants]


def report(rows: list[dict]) -> None:
    ref = next((r for r in rows if r["readout"] == "linear_head"), None)
    print(f"\n{'readout':<18} {'logit range':>13} {'prob spread':>13} "
          f"{'trunk ||grad||':>15} {'vs linear':>13}")
    print("-" * 76)
    for r in rows:
        ratio = ""
        if ref and r["trunk_grad_norm"] > 0:
            # Relative to the linear control. Eq. 5's sum runs STRONGER than the control (it adds
            # a per-node logit over every node, so it scales with graph size), and printing that
            # as "0.0x weaker" inverted the finding -- say which direction it goes.
            factor = ref["trunk_grad_norm"] / r["trunk_grad_norm"]
            ratio = (f"{factor:.1f}x weaker" if factor >= 1.0
                     else f"{1.0 / factor:.1f}x stronger")
        if r["readout"] == "linear_head":
            ratio = "(reference)"
        print(f"{r['readout']:<18} {r['logit_range']:>13.3e} {r['prob_spread']:>13.3e} "
              f"{r['trunk_grad_norm']:>15.3e} {ratio:>13}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    device = resolve_device(cfg["project"]["device"])
    seed_from_config(cfg, seed)

    _, _, train_loader, _ = load_graph_loaders(cfg, EDGE_TYPES)
    batch = next(iter(train_loader)).to(device)
    print(f"[measure_readout] batch: {batch.code_feat.shape[0]} graphs x "
          f"{batch.code_feat.shape[1]} padded nodes, {batch.edge_index.shape[1]} edges, "
          f"{float(batch.labels.float().mean()) * 100:.1f}% positive | device={device} seed={seed}")

    rows = measure(cfg, device, batch, seed)
    report(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": args.config, "seed": seed, "device": device,
                       "rows": rows}, f, indent=2)
        print(f"[measure_readout] wrote {args.json}")


if __name__ == "__main__":
    main()
