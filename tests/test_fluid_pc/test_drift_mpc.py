"""MPCDrift solenoidal-first conformance (IFN §10.5 / §11.1)."""

import torch

from quantimork_thrml.fluid_pc.advection import ConservativeAdvection
from quantimork_thrml.fluid_pc.drift_mpc import MPCDrift
from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector


def _make_mpc(graph, **kw):
    leray = LerayProjector(graph, tol=1e-6, max_iter=100)
    adv = ConservativeAdvection(graph, kappa_init=0.01)
    defaults = dict(horizon=2, inner_steps=3, inner_lr=0.05)
    defaults.update(kw)
    return MPCDrift(graph, leray, adv, **defaults)


def test_mpc_returns_divergence_free_velocity():
    torch.manual_seed(0)
    g = WaveletGraph(S=4, D=8, n_levels=2)
    mpc = _make_mpc(g)
    rho = torch.rand(2, g.n_nodes)
    rho = rho / rho.sum(-1, keepdim=True)
    v = rho.clone()
    u = mpc(rho, v)
    div = g.divergence(u)
    assert div.abs().max().item() < 1e-4


def test_alpha_optimization_produces_nonzero_velocity():
    """MPC should drive alpha away from zero when terminal target differs
    from current rho — i.e., the inner Adam loop is wired up correctly."""
    torch.manual_seed(0)
    g = WaveletGraph(S=4, D=8, n_levels=2)
    mpc = _make_mpc(g, inner_steps=5, inner_lr=0.1, terminal_weight=10.0)
    rho = torch.rand(1, g.n_nodes)
    rho = rho / rho.sum(-1, keepdim=True)
    target = torch.zeros_like(rho)
    target[0, 0] = 1.0
    with torch.enable_grad():
        alpha_star = mpc._solve(rho, target)
    assert alpha_star.abs().max().item() > 0
    assert torch.isfinite(alpha_star).all().item()


def test_leray_is_no_op_on_solenoidal_velocity():
    """Regression: u = U @ alpha is divergence-free, so Leray should leave it
    essentially unchanged (the §10.10 safety-net role)."""
    torch.manual_seed(0)
    g = WaveletGraph(S=4, D=8, n_levels=2)
    leray = LerayProjector(g, tol=1e-6, max_iter=100)
    alpha = torch.randn(2, g.n_cycles)
    u = g.apply_U(alpha)
    u_proj = leray(u)
    rel = (u_proj - u).norm(dim=-1) / u.norm(dim=-1).clamp_min(1e-8)
    assert rel.max().item() < 0.05
