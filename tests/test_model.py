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

from models.cnn import BaseBeaconlessCNN, CNN1, CNNL, count_parameters


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
