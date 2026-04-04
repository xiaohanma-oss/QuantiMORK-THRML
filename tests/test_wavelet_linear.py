"""Tests for WaveletLinear module."""

import pytest
import torch
import torch.nn as nn

from quantimork_thrml.wavelet_linear import WaveletLinear


class TestWaveletLinearShape:
    """WaveletLinear must preserve input shape."""

    def test_basic_shape(self):
        layer = WaveletLinear(512, 512, n_levels=3)
        x = torch.randn(2, 4, 512)
        y = layer(x)
        assert y.shape == x.shape

    def test_2d_input(self):
        layer = WaveletLinear(256, 256, n_levels=2)
        x = torch.randn(8, 256)
        y = layer(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("D,L", [(128, 2), (256, 3), (512, 3), (1024, 4)])
    def test_various_sizes(self, D, L):
        layer = WaveletLinear(D, D, n_levels=L)
        x = torch.randn(2, 4, D)
        y = layer(x)
        assert y.shape == x.shape


class TestWaveletLinearParams:
    """WaveletLinear should have significantly fewer params than dense."""

    def test_fewer_than_dense(self):
        D = 512
        wl = WaveletLinear(D, D, n_levels=3)
        dense = nn.Linear(D, D)
        assert wl.num_params() < dense.weight.numel() + dense.bias.numel()

    def test_param_count_512_3levels(self):
        """3-level on D=512:
        detail_1: Linear(256, 256) = 256*256 + 256 = 65,792
        detail_2: Linear(128, 128) = 128*128 + 128 = 16,512
        detail_3: Linear(64, 64)   = 64*64 + 64   = 4,160
        approx:   Linear(64, 64)   = 64*64 + 64   = 4,160
        Total = 90,624
        Dense Linear(512, 512) = 512*512 + 512 = 262,656
        """
        wl = WaveletLinear(512, 512, n_levels=3)
        assert wl.num_params() == 90_624
        # ~2.9x compression
        assert wl.num_params() < 262_656 / 2


class TestWaveletLinearGradient:
    """Gradients must flow through WaveletLinear."""

    def test_gradient_nonzero(self):
        layer = WaveletLinear(512, 512, n_levels=3)
        x = torch.randn(2, 4, 512, requires_grad=True)
        y = layer(x)
        # Use .pow(2).sum() — plain .sum() cancels detail gradients
        # because IDWT interleaves (a+d) and (a-d), which sum to 2a.
        loss = y.pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestWaveletLinearConstraints:
    """Validate input constraints."""

    def test_unequal_dims_raises(self):
        with pytest.raises(ValueError, match="in_features == out_features"):
            WaveletLinear(512, 256, n_levels=3)

    def test_indivisible_dim_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            WaveletLinear(100, 100, n_levels=3)


class TestExtractEnergyParams:
    """extract_energy_params should return per-level weight info."""

    def test_structure(self):
        layer = WaveletLinear(512, 512, n_levels=3)
        params = layer.extract_energy_params()
        assert len(params) == 4  # 3 detail + 1 approx
        assert params[0]["band"] == "detail"
        assert params[0]["level"] == 1
        assert params[0]["weight"].shape == (256, 256)
        assert params[-1]["band"] == "approx"
        assert params[-1]["weight"].shape == (64, 64)

    def test_detached(self):
        """Exported weights should be detached copies."""
        layer = WaveletLinear(512, 512, n_levels=3)
        params = layer.extract_energy_params()
        for p in params:
            assert not p["weight"].requires_grad
