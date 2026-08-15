"""The two Tier-1 fixes: the readout's dead start, and the fixed learning rate.

Eq. 9 ends in a product of two MLP heads, so at initialisation both factors are near zero. Measured
on this data that means the whole batch's logits span ~6e-4 (every probability lands in
0.5000-0.5008) and the gradient reaching the GGNN trunk is ~53x weaker than through a linear head
on the same pooled features. The visible symptom in a real run was `f1 0.00` for the first eight
epochs while the model discovered nothing but the class prior.

These tests pin the fixes: the bias starts the model AT the prior, and the LR schedule exists and
is wired to the monitored metric.
"""
from __future__ import annotations

import copy

import torch

from models.conv_module import ConvModule, prior_logit
from training.trainer import TrainConfig, make_train_config
from training.utils import load_config

Z, D, B = 24, 16, 8


def _conv_cfg() -> dict:
    cfg = copy.deepcopy(load_config("config.yaml")["model"]["conv"])
    cfg["conv_channels"] = 8
    cfg["mlp_hidden"] = 12
    return cfg


def _logits(pos_rate, **overrides):
    cfg = _conv_cfg()
    cfg.update(overrides)
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=cfg, mlp_hidden=12, dropout=0.0,
                     pos_rate=pos_rate)
    mod.eval()
    torch.manual_seed(1)
    H, x = torch.randn(B, 40, Z), torch.randn(B, 40, D)
    mask = torch.ones(B, 40, dtype=torch.bool)
    with torch.no_grad():
        return mod(H, x, mask)


def test_prior_logit_matches_the_class_balance():
    assert prior_logit(0.5) == 0.0
    assert prior_logit(None) == 0.0
    assert prior_logit(0.426) < 0.0        # qemu is minority-positive
    assert prior_logit(0.51) > 0.0         # ffmpeg is majority-positive
    assert abs(1 / (1 + torch.tensor(-prior_logit(0.426)).exp()) - 0.426) < 1e-5


def test_untrained_model_predicts_the_class_prior_not_half():
    """Without the bias the model starts at p=0.5 for every graph regardless of the split's
    balance, and has to reach the prior through a multiplicative bottleneck.

    `logit_scale_init` is pinned to 1.0 here so this isolates the bias; the scale is exercised by
    test_logit_scale_init_amplifies_the_gradient_into_the_trunk.
    """
    qemu_rate = 0.426
    with_bias = torch.sigmoid(_logits(qemu_rate, logit_scale_init=1.0)).mean().item()
    without_bias = torch.sigmoid(_logits(qemu_rate, logit_affine=False)).mean().item()

    assert abs(with_bias - qemu_rate) < 0.01, f"expected ~{qemu_rate}, got {with_bias}"
    assert abs(without_bias - 0.5) < 0.01, f"paper readout should start at 0.5, got {without_bias}"


def test_default_scale_init_counteracts_the_measured_attenuation():
    """The default is derived from the ~53x gradient attenuation, so it must not be left at the
    neutral 1.0 -- a silent revert here costs most of the model's learning speed."""
    assert load_config("config.yaml")["model"]["conv"]["logit_scale_init"] >= 10.0


def test_paper_readout_starts_in_a_razor_thin_probability_band():
    """Documents the failure the affine addresses, so a regression here is visible."""
    probs = torch.sigmoid(_logits(None, logit_affine=False))
    assert (probs.max() - probs.min()).item() < 0.01


def test_logit_scale_init_amplifies_the_gradient_into_the_trunk():
    """`scale` multiplies the gradient flowing back through both branches, which is the lever
    against the measured ~53x attenuation."""
    grads = {}
    for scale in (1.0, 10.0):
        cfg = _conv_cfg()
        cfg["logit_scale_init"] = scale
        torch.manual_seed(0)
        mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=cfg, mlp_hidden=12, dropout=0.0)
        torch.manual_seed(1)
        H = torch.randn(B, 40, Z, requires_grad=True)
        x = torch.randn(B, 40, D)
        mod(H, x, torch.ones(B, 40, dtype=torch.bool)).sum().backward()
        grads[scale] = H.grad.norm().item()

    assert grads[10.0] > 5 * grads[1.0], grads


def test_lr_schedule_is_configured_and_plateau_is_the_default():
    tr = load_config("config.yaml")["training"]
    assert tr["lr_schedule"] == "plateau"
    cfg = make_train_config(load_config("config.yaml"), "cpu")
    assert cfg.lr_schedule == "plateau"
    assert cfg.lr_factor < 1.0 and cfg.lr_patience >= 1


def test_lr_schedule_defaults_to_paper_behaviour_when_absent():
    """A config predating this key must keep the paper's fixed learning rate, not silently
    acquire a schedule."""
    cfg = load_config("config.yaml")
    cfg["training"].pop("lr_schedule")
    assert make_train_config(cfg, "cpu").lr_schedule == "none"
    assert TrainConfig().lr_schedule == "none"


def test_unknown_lr_schedule_is_rejected_loudly():
    """A typo must not silently fall back to a fixed LR after a multi-hour run."""
    import pytest

    from training.trainer import train_model

    with pytest.raises(ValueError, match="lr_schedule"):
        train_model(torch.nn.Linear(2, 1), [], [],
                    TrainConfig(lr_schedule="cosine", epochs=1), verbose=False)
