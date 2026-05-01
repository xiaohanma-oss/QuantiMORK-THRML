"""LerayProjector: divergence-free projection + idempotency + autograd."""

import torch

from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector


def test_divergence_free_after_projection():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    leray = LerayProjector(g, tol=1e-6, max_iter=200)
    u_raw = torch.randn(2, g.n_edges)
    u_proj = leray(u_raw)
    div = g.divergence(u_proj)
    assert div.abs().max().item() < 1e-3


def test_projection_is_idempotent():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    leray = LerayProjector(g, tol=1e-6, max_iter=200)
    u_raw = torch.randn(2, g.n_edges)
    u1 = leray(u_raw)
    u2 = leray(u1)
    assert torch.allclose(u1, u2, atol=1e-3)


def test_backward_runs_and_is_finite():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    leray = LerayProjector(g, tol=1e-5, max_iter=100)
    u_raw = torch.randn(2, g.n_edges, requires_grad=True)
    u_proj = leray(u_raw)
    (u_proj.pow(2).sum()).backward()
    assert u_raw.grad is not None
    assert torch.isfinite(u_raw.grad).all().item()
