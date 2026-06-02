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
