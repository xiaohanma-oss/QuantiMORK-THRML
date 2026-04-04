"""Shared fixtures and tolerances for QuantiMORK-THRML tests."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


HAAR_ROUNDTRIP_TOL = 1e-6
GRADIENT_MIN = 1e-8

# thrml verification tolerances (K-dependent, same pattern as PLN-THRML)
ENERGY_TOL_BY_K = {4: 0.20, 8: 0.15, 16: 0.10}
ACTIVATION_MSE_BY_K = {4: 0.15, 8: 0.10, 16: 0.08}


def energy_tol(k=16):
    return ENERGY_TOL_BY_K.get(k, 0.10)


def activation_mse_tol(k=16):
    return ACTIVATION_MSE_BY_K.get(k, 0.08)


@pytest.fixture
def device():
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.device("cpu")


@pytest.fixture(params=[256, 512], ids=["D=256", "D=512"])
def feature_dim(request):
    return request.param


@pytest.fixture(params=[2, 3], ids=["L=2", "L=3"])
def n_levels(request):
    return request.param


@pytest.fixture
def sample_input(device):
    """(B=2, S=4, D=512) random tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 512, device=device)


@pytest.fixture
def step_signal():
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
