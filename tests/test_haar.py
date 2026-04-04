"""Tests for Haar DWT/IDWT correctness."""

import pytest
import torch
from math import sqrt

from quantimork_thrml.haar import haar_dwt_1d, haar_idwt_1d, tree_positions


class TestHaarRoundtrip:
    """DWT → IDWT must recover the original signal exactly."""

    def test_1d_simple(self, step_signal):
        coeffs = haar_dwt_1d(step_signal.unsqueeze(0), n_levels=2)
        recovered = haar_idwt_1d(coeffs).squeeze(0)
        assert torch.allclose(recovered, step_signal, atol=1e-6)

    def test_3d_random(self, sample_input):
        """(B, S, D) tensor roundtrip."""
        coeffs = haar_dwt_1d(sample_input, n_levels=3)
        recovered = haar_idwt_1d(coeffs)
        assert torch.allclose(recovered, sample_input, atol=1e-6)

    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
    def test_various_levels(self, n_levels):
        D = 2 ** (n_levels + 2)  # ensure divisible
        x = torch.randn(4, D)
        coeffs = haar_dwt_1d(x, n_levels=n_levels)
        recovered = haar_idwt_1d(coeffs)
        assert torch.allclose(recovered, x, atol=1e-6)


class TestHaarCoefficients:
    """Verify known Haar coefficients for simple signals."""

    def test_step_signal_level1(self):
        """[1,1,0,0] → approx=[√2/2, 0], detail=[√2/2, 0] (unnormalized)."""
        x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        coeffs = haar_dwt_1d(x, n_levels=1)
        # After one level: approx = (even+odd)/√2, detail = (even-odd)/√2
        # even=[1,0], odd=[1,0]
        # approx = [2/√2, 0] = [√2, 0]
        # detail = [0, 0]
        s2 = 1.0 / sqrt(2)
        expected_approx = torch.tensor([[1.0 * s2 + 1.0 * s2, 0.0]])
        expected_detail = torch.tensor([[1.0 * s2 - 1.0 * s2, 0.0]])
        assert torch.allclose(coeffs["approx"], expected_approx, atol=1e-6)
        assert torch.allclose(coeffs["details"][0], expected_detail, atol=1e-6)

    def test_constant_signal(self):
        """Constant signal → all detail coefficients are zero."""
        x = torch.ones(1, 8) * 3.0
        coeffs = haar_dwt_1d(x, n_levels=3)
        for d in coeffs["details"]:
            assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


class TestHaarShapes:
    """Verify output shapes match expectations."""

    def test_coefficient_sizes(self):
        x = torch.randn(2, 4, 512)
        coeffs = haar_dwt_1d(x, n_levels=3)
        assert coeffs["details"][0].shape == (2, 4, 256)  # finest detail
        assert coeffs["details"][1].shape == (2, 4, 128)
        assert coeffs["details"][2].shape == (2, 4, 64)   # coarsest detail
        assert coeffs["approx"].shape == (2, 4, 64)       # coarsest approx

    def test_invalid_dim_raises(self):
        x = torch.randn(2, 4, 100)  # 100 not divisible by 8
        with pytest.raises(ValueError):
            haar_dwt_1d(x, n_levels=3)


class TestTreePositions:
    def test_3_levels_512(self):
        positions = tree_positions(3, 512)
        assert len(positions) == 4  # 3 detail + 1 approx
        assert positions[0] == {"level": 1, "band": "detail", "size": 256}
        assert positions[1] == {"level": 2, "band": "detail", "size": 128}
        assert positions[2] == {"level": 3, "band": "detail", "size": 64}
        assert positions[3] == {"level": 3, "band": "approx", "size": 64}
