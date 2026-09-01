"""Tests for the beaconless CNN models (models.cnn).

Architecture facts (verified against Optics Express 33(15):31010, 2025):
- 3 stages of 3x3 conv + BatchNorm2d + ReLU + 2x2 MaxPool.
- 512x512 input -> 64x64 feature maps (128 channels) -> AdaptiveAvgPool2d((18,18))
  -> flatten 128*18*18 = 41472 -> MLP 4x512 ReLU -> 78 outputs.
- CNN1 total params == 22_155_854.
- CNNL == CNN1 + length head Linear(1,512) (1024 params) + 512*512 extra first-MLP
  weights (262_144) == 22_419_022.
"""

import torch
import pytest

from models.cnn import (
    BaseBeaconlessCNN,
    CNN1,
    CNN1Freq,
    CNN1Star,
    CNNL,
    SEBlock,
    StarBlock,
    count_parameters,
)


def test_cnn1_forward_shape():
    """CNN1() on (2, 3, 512, 512) must output (2, 78)."""
    torch.manual_seed(0)
    model = CNN1()
    images = torch.randn(2, 3, 512, 512)
    with torch.inference_mode():
        out = model(images)
    assert out.shape == (2, 78)


def test_cnnl_forward_shape():
    """CNNL() on (2, 3, 512, 512) + length (2,) must output (2, 78)."""
    torch.manual_seed(0)
    model = CNNL()
    images = torch.randn(2, 3, 512, 512)
    length = torch.randn(2)
    with torch.inference_mode():
        out = model(images, length)
    assert out.shape == (2, 78)


def test_cnn1_length_head_absent():
    """CNN1 must have NO length head; CNNL must have one."""
    cnn1 = CNN1()
    cnnl = CNNL()
    assert not hasattr(cnn1, "length_head")
    assert hasattr(cnnl, "length_head")


def test_param_count_cnn1():
    """CNN1 must have exactly 22_155_854 trainable params (±2)."""
    model = CNN1()
    n = count_parameters(model)
    assert abs(n - 22_155_854) <= 2, f"CNN1 params = {n}, expected ~22_155_854"


def test_param_count_cnnl():
    """CNNL must have exactly 22_419_022 trainable params (±2)."""
    model = CNNL()
    n = count_parameters(model)
    assert abs(n - 22_419_022) <= 2, f"CNNL params = {n}, expected ~22_419_022"


def test_mse_backward():
    """Forward + MSE loss + backward must complete with finite gradients."""
    torch.manual_seed(0)
    model = CNN1()
    images = torch.randn(2, 3, 512, 512)
    target = torch.randn(2, 78)
    out = model(images)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mse_backward_cuda():
    """Optional GPU backward smoke test (skipped if no CUDA)."""
    torch.manual_seed(0)
    model = CNN1().cuda()
    images = torch.randn(2, 3, 512, 512, device="cuda")
    target = torch.randn(2, 78, device="cuda")
    out = model(images)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()


def test_length_head_effect():
    """Gradient must flow into the length head: d(output)/d(length) finite & nonzero."""
    torch.manual_seed(0)
    model = CNNL()
    images = torch.randn(1, 3, 512, 512)
    length = torch.randn(1, requires_grad=True)
    out = model(images, length)
    grad = torch.autograd.grad(out[0, 0], length, create_graph=False)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().item() > 0


def test_cnn1freq_forward_shape():
    """CNN1Freq() on (2, 3, 512, 512) must output (2, 78)."""
    torch.manual_seed(0)
    model = CNN1Freq()
    images = torch.randn(2, 3, 512, 512)
    with torch.inference_mode():
        out = model(images)
    assert out.shape == (2, 78)


def test_cnn1freq_has_freq_branch():
    """CNN1Freq must expose a working FrequencyBranch; CNN1 must not."""
    cnn1 = CNN1()
    freq = CNN1Freq()
    assert not hasattr(cnn1, "freq_branch")
    assert hasattr(freq, "freq_branch")
    torch.manual_seed(0)
    images = torch.randn(2, 3, 512, 512)
    with torch.inference_mode():
        feats = freq.freq_branch(images)
    assert feats.shape == (2, freq.freq_branch.freq_size)
    assert torch.isfinite(feats).all()


def test_cnn1freq_param_count():
    """CNN1Freq must have more params than CNN1 (the spectral branch adds weights)."""
    torch.manual_seed(0)
    n_cnn1 = count_parameters(CNN1())
    n_freq = count_parameters(CNN1Freq())
    assert n_freq > n_cnn1
    assert abs(n_freq - 22_680_222) <= 2, f"CNN1Freq params = {n_freq}"


def test_cnn1freq_mse_backward():
    """Forward + MSE loss + backward must complete with finite gradients."""
    torch.manual_seed(0)
    model = CNN1Freq()
    images = torch.randn(2, 3, 512, 512)
    target = torch.randn(2, 78)
    out = model(images)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    # Gradients must flow into the frequency branch in particular.
    fb_grads = [p.grad for p in model.freq_branch.parameters() if p.grad is not None]
    assert len(fb_grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()
    for g in fb_grads:
        assert torch.isfinite(g).all()


def test_starblock_shape_and_residual():
    """StarBlock preserves spatial dims and channel count (residual add)."""
    torch.manual_seed(0)
    block = StarBlock(dim=32, mlp_ratio=4)
    x = torch.randn(2, 32, 64, 64)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_seblock_shape():
    """SEBlock preserves (B, C, H, W) and reweights with a sigmoid in [0,1]."""
    torch.manual_seed(0)
    se = SEBlock(channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    out = se(x)
    assert out.shape == x.shape


def test_cnn1star_forward_shape():
    """CNN1Star() on (2, 3, 512, 512) must output (2, 78)."""
    torch.manual_seed(0)
    model = CNN1Star()
    images = torch.randn(2, 3, 512, 512)
    with torch.inference_mode():
        out = model(images)
    assert out.shape == (2, 78)


def test_cnn1star_se_switch():
    """CNN1Star(use_se=True) must have an SEBlock tail; use_se=False must not."""
    with_se = CNN1Star(use_se=True)
    without = CNN1Star(use_se=False)
    assert any(isinstance(m, SEBlock) for m in with_se.features)
    assert not any(isinstance(m, SEBlock) for m in without.features)


def test_cnn1star_param_count():
    """CNN1Star must be smaller than CNN1 (efficient StarNet extractor)."""
    torch.manual_seed(0)
    n_cnn1 = count_parameters(CNN1())
    n_star = count_parameters(CNN1Star())
    assert n_star < n_cnn1
    assert abs(n_star - 10_852_878) <= 2, f"CNN1Star params = {n_star}"


def test_cnn1star_mse_backward():
    """Forward + MSE loss + backward must complete with finite gradients."""
    torch.manual_seed(0)
    model = CNN1Star(use_se=True)
    images = torch.randn(2, 3, 512, 512)
    target = torch.randn(2, 78)
    out = model(images)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()


def test_cnn1star_loads_old_star_features_checkpoint():
    """CNN1Star must load a state_dict saved under the old ``star_features`` keys."""
    torch.manual_seed(0)
    model = CNN1Star(use_se=True)
    # Re-save every key under the legacy naming: `features.` -> `star_features.`.
    legacy = {}
    for k, v in model.state_dict().items():
        new_k = ("star_features." + k[len("features."):]) if k.startswith("features.") else k
        legacy[new_k] = v
    missing, unexpected = model.load_state_dict(legacy, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
