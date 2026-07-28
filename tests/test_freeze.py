"""Unit tests for training.freeze.{freeze_backbone, unfreeze_backbone, build_param_groups}."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.freeze import build_param_groups, freeze_backbone, unfreeze_backbone


class _FakeModel(nn.Module):
    """Minimal stand-in: exposes `.encoder` with its own parameters plus a decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 1))
        self.segmentation_head = nn.Sequential(nn.Linear(1, 1))


def test_freeze_backbone_disables_grad_on_encoder_only():
    m = _FakeModel()
    freeze_backbone(m)
    for p in m.encoder.parameters():
        assert p.requires_grad is False
    for p in m.decoder.parameters():
        assert p.requires_grad is True


def test_unfreeze_backbone_restores_grad():
    m = _FakeModel()
    freeze_backbone(m)
    unfreeze_backbone(m)
    for p in m.encoder.parameters():
        assert p.requires_grad is True


def test_build_param_groups_partitions_by_id():
    """Every param appears in exactly one group; encoder params go to 'backbone'."""
    m = _FakeModel()
    groups = build_param_groups(m, decoder_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-2)
    assert {g["name"] for g in groups} == {"decoder", "backbone"}

    all_model = {id(p) for p in m.parameters()}
    all_grouped = {id(p) for g in groups for p in g["params"]}
    assert all_model == all_grouped
    # No overlap between groups.
    decoder_ids = {id(p) for g in groups if g["name"] == "decoder" for p in g["params"]}
    backbone_ids = {id(p) for g in groups if g["name"] == "backbone" for p in g["params"]}
    assert not decoder_ids & backbone_ids
    # Encoder params are in 'backbone'.
    assert {id(p) for p in m.encoder.parameters()} == backbone_ids


def test_build_param_groups_lrs_set():
    m = _FakeModel()
    groups = build_param_groups(m, decoder_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-2)
    for g in groups:
        if g["name"] == "decoder":
            assert g["lr"] == 1e-3
        else:
            assert g["lr"] == 1e-4
        assert g["weight_decay"] == 1e-2


def test_optimizer_respects_frozen_encoder():
    """After freeze, encoder weights don't change through an optimizer step."""
    m = _FakeModel()
    groups = build_param_groups(m, decoder_lr=1e-1, backbone_lr=1e-1, weight_decay=0.0)
    optim = torch.optim.AdamW(groups)
    freeze_backbone(m)

    # Snapshot encoder weights.
    enc_before = {id(p): p.detach().clone() for p in m.encoder.parameters()}
    # Fake a loss that touches everything and step.
    x = torch.randn(2, 8)
    y = m.encoder(x)
    y = m.decoder(y)
    loss = y.sum()
    loss.backward()
    optim.step()

    for p in m.encoder.parameters():
        assert torch.equal(p.detach(), enc_before[id(p)]), (
            "Frozen encoder weights changed after optimizer step"
        )


# --- LLRD param groups (§8.2a, second-wave Step 4) ---------------------------
import pytest  # noqa: E402
from training.freeze import build_llrd_param_groups  # noqa: E402
from models.foundation import FoundationSegmenter  # noqa: E402


def _vit():
    return FoundationSegmenter("vit_base_patch16_dinov3", pretrained=False)


def test_build_llrd_param_groups_decay_and_coverage():
    m = _vit()
    groups = build_llrd_param_groups(m, lr=1e-4, weight_decay=0.05, llrd_decay=0.7)
    bb = [g for g in groups if g["name"] == "backbone"]
    dec = [g for g in groups if g["name"] == "decoder"]
    assert len(dec) == 1 and dec[0]["lr_scale"] == 1.0
    scales = [g["lr_scale"] for g in bb]
    assert scales == sorted(scales)              # stem (min) → top (max)
    assert scales[-1] == pytest.approx(1.0)      # top encoder layer keeps full LR
    assert scales[0] < scales[-1]                # early layers decayed
    # every model param in exactly one group
    grouped = [p for g in groups for p in g["params"]]
    assert {id(p) for p in grouped} == {id(p) for p in m.parameters()}
    assert len(grouped) == len(list(m.parameters()))


def test_build_llrd_rejects_bad_decay():
    m = _vit()
    with pytest.raises(ValueError, match="llrd_decay"):
        build_llrd_param_groups(m, lr=1e-4, weight_decay=0.0, llrd_decay=1.5)


def test_build_llrd_param_groups_works_for_hierarchical_sam2():
    """LLRD must map Hiera layers too — `.blocks` lives under the features_only wrapper.

    fm_sam2_rgb set `llrd_decay: null` believing Hiera had no `.blocks`, so its 34M-param
    encoder fine-tuned at a flat LR while every DINOv3 arm got a 0.75 taper. It has 16
    blocks; this pins the reach-through (2026-07-28).
    """
    m = FoundationSegmenter("sam2_hiera_tiny", pretrained=False)
    groups = build_llrd_param_groups(m, lr=1e-4, weight_decay=0.05, llrd_decay=0.75)
    bb = [g for g in groups if g["name"] == "backbone"]
    assert len(bb) > 1, "no per-layer encoder groups → LLRD silently degenerate"
    scales = [g["lr_scale"] for g in bb]
    assert scales == sorted(scales) and scales[-1] == pytest.approx(1.0)
    grouped = [p for g in groups for p in g["params"]]
    assert {id(p) for p in grouped} == {id(p) for p in m.parameters()}


def test_build_llrd_raises_when_no_blocks_anywhere():
    """An encoder with no `.blocks` at either level must fail loudly, not silently flatten."""
    m = _FakeModel()
    with pytest.raises(ValueError, match="blocks"):
        build_llrd_param_groups(m, lr=1e-4, weight_decay=0.0, llrd_decay=0.75)
