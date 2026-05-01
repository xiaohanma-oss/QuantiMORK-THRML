"""WaveletGraph: tensor packing, divergence/gradient, Laplacian."""

import torch

from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.haar import haar_dwt_1d


def test_coeffs_to_rho_roundtrip():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    x = torch.randn(2, 4, 8)
    coeffs = haar_dwt_1d(x, n_levels=2)
    rho = g.coeffs_to_rho(coeffs)
    assert rho.shape == (2, g.n_nodes)
    coeffs_back = g.rho_to_coeffs(rho)
    for d_in, d_out in zip(coeffs["details"], coeffs_back["details"]):
        assert torch.allclose(d_in, d_out)
    assert torch.allclose(coeffs["approx"], coeffs_back["approx"])


def test_divergence_adjoint_of_gradient():
    g = WaveletGraph(S=3, D=8, n_levels=2)
    p = torch.randn(g.n_nodes)
    u = torch.randn(g.n_edges)
    lhs = (u * g.gradient(p)).sum()
    rhs = (p * g.divergence(u)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_laplacian_is_psd():
    g = WaveletGraph(S=3, D=8, n_levels=2)
    for _ in range(5):
        v = torch.randn(g.n_nodes)
        quad = (v * g.laplacian_apply(v)).sum().item()
        assert quad >= -1e-5


def test_laplacian_constant_in_kernel():
    g = WaveletGraph(S=3, D=8, n_levels=2)
    one = torch.ones(g.n_nodes)
    Lone = g.laplacian_apply(one)
    assert Lone.abs().max().item() < 1e-5
