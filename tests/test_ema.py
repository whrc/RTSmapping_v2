"""Unit tests for training.ema.EMAModel."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.ema import EMAModel


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def test_ema_init_matches_model_state():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    for k, v in m.state_dict().items():
        assert torch.equal(ema.shadow[k], v)


def test_ema_update_converges_to_target():
    """Repeated updates with a fixed target drive shadow toward target."""
    m = _tiny_model()
    ema = EMAModel(m, decay=0.5)
    # Replace m's weights with a constant target.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    for _ in range(20):
        ema.update(m)
    # After 20 updates at decay=0.5, shadow is within 1e-6 of target.
    for v in ema.shadow.values():
        assert torch.allclose(v, torch.ones_like(v), atol=1e-5)


def test_ema_swap_in_restores_live_weights():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    # Snapshot live weights.
    live_before = {k: v.clone() for k, v in m.state_dict().items()}
    # Modify EMA shadow so it differs.
    for v in ema.shadow.values():
        v.mul_(0)
    with ema.swap_in(m):
        # Inside the context, model holds EMA (all zeros).
        for v in m.state_dict().values():
            assert torch.allclose(v, torch.zeros_like(v))
    # After exit, live weights are restored.
    for k, v in m.state_dict().items():
        assert torch.equal(v, live_before[k])


def test_ema_swap_in_restores_on_exception():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    live_before = {k: v.clone() for k, v in m.state_dict().items()}
    for v in ema.shadow.values():
        v.mul_(0)
    try:
        with ema.swap_in(m):
            raise RuntimeError("simulated")
    except RuntimeError:
        pass
    for k, v in m.state_dict().items():
        assert torch.equal(v, live_before[k])


def test_ema_state_dict_roundtrip():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    sd = ema.state_dict()
    # Build a fresh EMA and load the saved state.
    fresh = EMAModel(_tiny_model(), decay=0.5)
    fresh.load_state_dict(sd)
    assert fresh.decay == 0.9
    for k in ema.shadow:
        assert torch.equal(fresh.shadow[k], ema.shadow[k])
