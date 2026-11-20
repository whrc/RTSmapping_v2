"""Tests for the `cgm` tile existence gate (docs/cgm.md).

Covers models.segmentation.{gate_logits, ExistenceGate, build_model with model.cgm},
losses.{tile_targets, tile_existence_loss}, the train/validate loop wiring in
scripts/train.py, and the shipped cgm screen configs. CPU only; small resnet18
models at 64 px except the one real-recipe forward check.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from losses import build_loss, tile_existence_loss, tile_targets  # noqa: E402
from models import build_model  # noqa: E402
from models.segmentation import ExistenceGate, gate_logits  # noqa: E402
from training.freeze import build_param_groups, freeze_backbone  # noqa: E402
from train import PrecisionSetup, _train_one_epoch, _validate  # noqa: E402
from utils.config import load_config, validate_training_config  # noqa: E402

CGM = {"pooling": "max", "dropout": 0.2, "init_bias": 3.0, "loss_weight": 0.1, "min_pos_px": 50}


def _cfg(cgm: dict | None = None, **model_overrides) -> dict:
    model = {"architecture": "unetplusplus", "backbone": "resnet18", "pretrained": False,
             "output_bias_prior": 0.005, **model_overrides}
    if cgm is not None:
        model["cgm"] = dict(cgm)
    return {"channels": {"rgb": True, "extra": [{"name": "ndvi", "band": 0}]}, "model": model}


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p) - torch.log1p(-p)


# ---------------------------------------------------------------------------
# Default-off parity and the build contract
# ---------------------------------------------------------------------------


def test_no_cgm_key_builds_the_plain_model():
    """Without model.cgm the model is the plain smp UNet++, with no classification head."""
    torch.manual_seed(0)
    model = build_model(_cfg())
    assert type(model) is smp.UnetPlusPlus
    assert model.classification_head is None
    assert not any("classification_head" in k for k in model.state_dict())


def test_cgm_model_contract():
    """Gated model: (B,1,H,W) logits, (B,1) tile logit, encoder/head delegation intact."""
    model = build_model(_cfg(CGM)).eval()
    assert isinstance(model, ExistenceGate)
    with torch.no_grad():
        y = model(torch.randn(2, 4, 64, 64))
    assert y.shape == (2, 1, 64, 64)
    assert model.last_tile_logit.shape == (2, 1)
    assert model.encoder is model.base.encoder
    assert model.segmentation_head[0].bias is not None
    assert "last_tile_logit" not in model.state_dict()


def test_cgm_inner_model_has_the_plain_architecture():
    """The cgm model's inner UNet++ has exactly the plain model's tensors (names and shapes);
    the head is the only addition. Same seed gives the same encoder init, but smp draws the
    head's init from the RNG before the decoder's, so the decoder init differs like a seed change."""
    torch.manual_seed(0)
    plain = build_model(_cfg()).state_dict()
    torch.manual_seed(0)
    gated = build_model(_cfg(CGM)).state_dict()
    inner = {k[len("base."):]: v for k, v in gated.items()
             if k.startswith("base.") and "classification_head" not in k}
    assert inner.keys() == plain.keys()
    assert all(inner[k].shape == plain[k].shape for k in plain)
    assert all(torch.equal(inner[k], plain[k]) for k in plain if k.startswith("encoder."))


def test_cgm_head_trains_with_the_decoder_and_survives_freeze():
    model = build_model(_cfg(CGM))
    freeze_backbone(model)
    head = list(model.base.classification_head.parameters())
    assert all(p.requires_grad for p in head)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    groups = {g["name"]: {id(p) for p in g["params"]} for g in
              build_param_groups(model, decoder_lr=1e-3, backbone_lr=1e-4, weight_decay=0.0)}
    assert all(id(p) in groups["decoder"] for p in head)
    assert not any(id(p) in groups["backbone"] for p in head)


@pytest.mark.parametrize("pooling", ["max", "avg"])
def test_cgm_pooling_is_passed_to_smp(pooling):
    model = build_model(_cfg({**CGM, "pooling": pooling}))
    pool = model.base.classification_head[0]
    expected = torch.nn.AdaptiveMaxPool2d if pooling == "max" else torch.nn.AdaptiveAvgPool2d
    assert isinstance(pool, expected)


@pytest.mark.parametrize("overrides", [{"architecture": "fpn"}, {"fusion": "chan_attn"}])
def test_cgm_rejects_unsupported_combinations(overrides):
    with pytest.raises(ValueError, match="model.cgm"):
        build_model(_cfg(CGM, **overrides))


# ---------------------------------------------------------------------------
# Gate arithmetic
# ---------------------------------------------------------------------------


def test_gate_logits_equals_logit_of_product():
    torch.manual_seed(0)
    d = torch.randn(3, 1, 8, 8, dtype=torch.float64) * 4
    z = torch.randn(3, 1, dtype=torch.float64) * 4
    expected = _logit(torch.sigmoid(d) * torch.sigmoid(z)[:, :, None, None])
    torch.testing.assert_close(gate_logits(d, z).double(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_logits_finite_values_and_grads_at_extremes(dtype):
    vals = torch.tensor([-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0])
    d = vals.repeat(len(vals)).reshape(len(vals), 1, 1, len(vals)).to(dtype).requires_grad_(True)
    z = vals.reshape(-1, 1).to(dtype).requires_grad_(True)
    out = gate_logits(d, z)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(d.grad.float()).all() and torch.isfinite(z.grad.float()).all()


def test_gate_never_raises_the_dense_probability():
    """σ(gated) = σ(d)·σ(z) ≤ σ(d): the gate can only suppress."""
    d = torch.linspace(-10, 10, 50).reshape(1, 1, 5, 10)
    for zval in (-5.0, 0.0, 3.0, 30.0):
        g = gate_logits(d, torch.full((1, 1), zval))
        assert (g <= d + 1e-4).all()


def test_neutral_start_gate_is_open_and_input_independent():
    """At init the tile logit is init_bias for every input, so the gated map is the dense map
    shifted down by at most |log σ(3)| ≈ 0.049 in log-probability, with the same ranking."""
    torch.manual_seed(0)
    model = build_model(_cfg(CGM)).eval()
    x = torch.randn(3, 4, 64, 64)
    with torch.no_grad():
        gated = model(x)
        dense, z = model.base(x)
    torch.testing.assert_close(z, torch.full_like(z, 3.0))
    expected = _logit(torch.sigmoid(dense.double()) * torch.sigmoid(torch.tensor(3.0, dtype=torch.float64)))
    torch.testing.assert_close(gated.double(), expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(gated.flatten().argsort(), dense.flatten().argsort())


# ---------------------------------------------------------------------------
# Tile labels and loss
# ---------------------------------------------------------------------------


def _label_with(n_pos: int, n_ignore: int = 0, size: int = 16) -> torch.Tensor:
    lab = torch.zeros(size * size, dtype=torch.long)
    lab[:n_pos] = 1
    lab[n_pos:n_pos + n_ignore] = 255
    return lab.reshape(size, size)


def test_tile_targets_rule():
    labels = torch.stack([_label_with(0), _label_with(50), _label_with(49), _label_with(1),
                          _label_with(0, n_ignore=200), _label_with(120)])
    target, keep = tile_targets(labels, min_pos_px=50)
    assert keep.tolist() == [True, True, False, False, True, True]
    assert target[keep].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_tile_loss_matches_bce_on_kept_tiles():
    labels = torch.stack([_label_with(0), _label_with(10), _label_with(80)])
    z = torch.tensor([[0.3], [5.0], [-0.7]], requires_grad=True)
    loss = tile_existence_loss(z, labels, min_pos_px=50)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.3, -0.7]), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert z.grad[1].item() == 0.0  # the dropped tile gets no gradient


def test_tile_loss_is_zero_when_no_tile_is_kept():
    z = torch.tensor([[1.0], [2.0]], requires_grad=True)
    loss = tile_existence_loss(z, torch.stack([_label_with(5), _label_with(30)]), min_pos_px=50)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


# ---------------------------------------------------------------------------
# Train / validate wiring, package round trip
# ---------------------------------------------------------------------------


def _loader(n_batches: int = 2, bs: int = 4, size: int = 64) -> list[dict]:
    g = torch.Generator().manual_seed(0)
    batches = []
    for b in range(n_batches):
        labels = torch.zeros(bs, size, size, dtype=torch.long)
        labels[0, 10:30, 10:30] = 1          # 400 px → positive tile
        labels[1, 0:3, 0:3] = 1              # 9 px → dropped
        batches.append({"image": torch.randn(bs, 4, size, size, generator=g), "label": labels,
                        "tile_id": [f"t{b}_{i}" for i in range(bs)]})
    return batches


def _fp32() -> PrecisionSetup:
    return PrecisionSetup(requested="fp32", effective="fp32", autocast_dtype=None,
                          use_scaler=False, scaler=None)


def _loss_cfg() -> dict:
    return {"loss": {"function": "focal", "focal_gamma": 2.0, "focal_alpha": 0.25},
            "data": {"label_ignore_index": 255}}


@pytest.mark.parametrize("with_cgm", [True, False])
def test_train_one_epoch_adds_the_tile_loss_only_when_enabled(with_cgm):
    torch.manual_seed(0)
    model = build_model(_cfg(CGM if with_cgm else None))
    head_before = copy.deepcopy(model.base.classification_head.state_dict()) if with_cgm else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = _train_one_epoch(model, _loader(), build_loss(_loss_cfg()), opt, _fp32(),
                           torch.device("cpu"), grad_clip_norm=1.0, ema=None,
                           exposure_counter={}, cgm_cfg=CGM if with_cgm else None)
    assert out["train_nan_steps"] == 0.0
    if with_cgm:
        assert out["train_tile_loss"] > 0.0 and torch.isfinite(torch.tensor(out["train_tile_loss"]))
        head_after = model.base.classification_head.state_dict()
        assert any(not torch.equal(head_before[k], head_after[k]) for k in head_after)
    else:
        assert "train_tile_loss" not in out


@pytest.mark.parametrize("with_cgm", [True, False])
def test_validate_reports_tile_loss_only_when_enabled(with_cgm):
    cfg = {**_cfg(CGM if with_cgm else None), **_loss_cfg(),
           "metrics": {"pr_auc_ratios": [5, 10, 20]}}
    metrics, _, _ = _validate(build_model(cfg), _loader(1), build_loss(cfg), _fp32(),
                              torch.device("cpu"), cfg)
    assert "val_loss" in metrics
    if with_cgm:
        assert metrics["val_tile_loss"] > 0.0
    else:
        assert "val_tile_loss" not in metrics


def test_package_round_trip_reproduces_logits(tmp_path):
    """What predictor.load_deployment_package does: rebuild from the config with
    pretrained=False, strict-load the saved state_dict, same logits."""
    torch.manual_seed(0)
    model = build_model(_cfg(CGM)).eval()
    with torch.no_grad():  # move the head off its init so the check is not trivial
        model.base.classification_head[3].weight.normal_()
    torch.save(model.state_dict(), tmp_path / "weights.pth")
    torch.manual_seed(123)
    loaded = build_model(_cfg(CGM))
    loaded.load_state_dict(torch.load(tmp_path / "weights.pth", weights_only=True))
    loaded.eval()
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


# ---------------------------------------------------------------------------
# Shipped configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling", ["max", "avg"])
def test_cgm_configs_differ_from_their_control_only_in_cgm(pooling):
    arm = load_config(REPO_ROOT / "configs" / f"cgm_{pooling}_seed42.yaml")
    ctrl = load_config(REPO_ROOT / "configs" / "aug_trivialaugment_deploy.yaml")
    validate_training_config(arm)
    assert arm["model"].pop("cgm") == {**CGM, "pooling": pooling}
    assert arm["mlflow"].pop("run_name") == f"cgm_{pooling}_seed42"
    ctrl["mlflow"].pop("run_name")
    assert arm == ctrl


def test_real_recipe_forward():
    """EfficientNet-B5 + NDVI as configured: (1,1,512,512) logits, a 513-parameter head."""
    cfg = load_config(REPO_ROOT / "configs" / "cgm_max_seed42.yaml")
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).eval()
    assert sum(p.numel() for p in model.base.classification_head.parameters()) == 513
    with torch.no_grad():
        y = model(torch.randn(1, 4, 512, 512))
    assert y.shape == (1, 1, 512, 512)
    assert model.last_tile_logit.shape == (1, 1)
