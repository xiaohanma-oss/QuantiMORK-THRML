"""Cycle-space basis verification (IFN §10.10)."""

import torch

from quantimork_thrml.fluid_pc.cycle_basis import build_cycle_basis
from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.topology import build_topology


def test_K_cyc_matches_euler_formula():
    topo = build_topology(S=4, D=8, n_levels=2)
    U, n_cyc = build_cycle_basis(topo)
    # K_cyc = |E| - |V| + C, where C is the number of connected components.
    # For this graph C = 2 (cross-edges within each s split into two trees
    # detail_1[0,1]<->detail_2[0]<->approx_2[0] and the b=2,3 mirror).
    assert n_cyc == topo.n_edges - topo.n_nodes + 2


def test_basis_lies_in_kernel_of_BT():
    topo = build_topology(S=4, D=8, n_levels=2)
    g = WaveletGraph(S=4, D=8, n_levels=2)
    torch.manual_seed(1)
    alpha = torch.randn(3, g.n_cycles)
    u = g.apply_U(alpha)
    div = g.divergence(u)
    assert div.abs().max().item() < 1e-5


def test_apply_U_shapes():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    alpha = torch.randn(2, g.n_cycles)
    u = g.apply_U(alpha)
    assert u.shape == (2, g.n_edges)


def test_apply_U_is_autograd_safe():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    alpha = torch.randn(2, g.n_cycles, requires_grad=True)
    u = g.apply_U(alpha)
    (u ** 2).sum().backward()
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all().item()


def test_face_local_basis_kernel_property():
    """§10.10 face-local cycles must lie in ker(B^T) just like BFS-tree ones."""
    g = WaveletGraph(S=4, D=8, n_levels=2, cycle_basis="face_local")
    alpha = torch.randn(2, g.n_cycles)
    u = g.apply_U(alpha)
    div = g.divergence(u)
    assert div.abs().max().item() < 1e-5


def test_face_local_columns_have_four_entries():
    """Each parent-child-square face contributes a 4-edge cycle."""
    from quantimork_thrml.fluid_pc.cycle_basis import build_face_local_basis
    topo = build_topology(S=4, D=8, n_levels=2)
    U, n = build_face_local_basis(topo)
    Ud = U.to_dense()
    nnz_per_col = (Ud != 0).sum(dim=0)
    assert (nnz_per_col[:n] == 4).all().item()


def test_basis_columns_are_signed_pm1():
    topo = build_topology(S=4, D=8, n_levels=2)
    U, _ = build_cycle_basis(topo)
    Ud = U.to_dense()
    nz = Ud[Ud != 0]
    assert torch.all((nz == 1) | (nz == -1)).item()
