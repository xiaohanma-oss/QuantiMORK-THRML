"""Cross-scale PC predictor (IFN §10.4)."""

import torch

from quantimork_thrml.fluid_pc.cross_scale_predictor import CrossScalePredictor
from quantimork_thrml.fluid_pc.graph import WaveletGraph


def test_predictor_shape_preserved():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    pred = CrossScalePredictor(g)
    rho = torch.rand(2, g.n_nodes)
    out = pred(rho)
    assert out.shape == rho.shape


def test_predictor_uses_all_three_thetas():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    pred = CrossScalePredictor(g)
    rho = torch.rand(1, g.n_nodes, requires_grad=True)
    out = pred(rho)
    out.sum().backward()
    assert pred.theta_lat.grad is not None
    assert pred.theta_up.grad is not None
    assert pred.theta_down.grad is not None
    assert pred.theta_lat.grad.abs().item() > 0
    assert pred.theta_up.grad.abs().item() > 0
    assert pred.theta_down.grad.abs().item() > 0


def test_predictor_per_node_connectivity_under_TSU_limit():
    """Sanity: each node's predicted value depends on at most ~5 neighbors
    (TSU 12-neighbor budget)."""
    g = WaveletGraph(S=4, D=8, n_levels=2)
    n_v = g.n_nodes
    deg = torch.zeros(n_v, dtype=torch.long)
    deg.scatter_add_(0, g.edge_src, torch.ones_like(g.edge_src))
    deg.scatter_add_(0, g.edge_dst, torch.ones_like(g.edge_dst))
    assert deg.max().item() <= 12
