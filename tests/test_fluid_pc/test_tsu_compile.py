"""TSU factor-graph verification of the FluidPC density-update + velocity-solve steps."""

import numpy as np
import pytest
import torch

from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector
from quantimork_thrml.fluid_pc.topology import build_topology
from quantimork_thrml.fluid_pc.tsu_compile import (
    build_density_update_graph,
    build_velocity_solve_graph,
    host_density_update,
    host_leray_projection,
    run_density_update_verification,
    run_velocity_solve_verification,
)


@pytest.mark.slow
def test_density_update_posterior_matches_torch_advection():
    np.random.seed(7)
    topo = build_topology(S=4, D=8, n_levels=2)

    rho = np.abs(np.random.randn(topo.n_nodes).astype(np.float32))
    rho /= rho.sum()
    u = 0.1 * np.random.randn(topo.n_edges).astype(np.float32)

    graph = build_density_update_graph(
        topo, rho, u, dt=0.1, kappa=0.01, precision=2.0, max_nodes=8)
    result = run_density_update_verification(graph, seed=7, n_batches=200)

    assert result["mse"] < 0.05


@pytest.mark.slow
def test_host_density_update_conserves_mass():
    np.random.seed(11)
    topo = build_topology(S=4, D=8, n_levels=2)
    rho = np.abs(np.random.randn(topo.n_nodes).astype(np.float32))
    rho /= rho.sum()
    u = 0.1 * np.random.randn(topo.n_edges).astype(np.float32)
    rho_new = host_density_update(topo, rho, u, dt=0.1, kappa=0.01)
    assert abs(rho_new.sum() - rho.sum()) < 1e-4


@pytest.mark.slow
def test_velocity_solve_posterior_matches_analytic_leray_softpenalty():
    np.random.seed(7)
    topo = build_topology(S=4, D=8, n_levels=2)
    u_raw = 0.5 * np.random.randn(topo.n_edges).astype(np.float32)
    g = build_velocity_solve_graph(
        topo, u_raw, lambda_div=10.0, precision=1.0, max_nodes=8)
    res = run_velocity_solve_verification(g, seed=7, n_batches=200)
    assert res["mse"] < 0.01


def test_velocity_solve_approaches_leray_as_lambda_grows():
    np.random.seed(13)
    topo = build_topology(S=4, D=8, n_levels=2)
    u_raw_np = 0.5 * np.random.randn(topo.n_edges).astype(np.float32)

    g = WaveletGraph(S=4, D=8, n_levels=2)
    leray = LerayProjector(g, tol=1e-7, max_iter=300)
    u_proj = leray(torch.from_numpy(u_raw_np)[None]).squeeze(0).numpy()

    soft = host_leray_projection(topo, u_raw_np, lambda_div=1000.0)
    err = np.mean((soft - u_proj) ** 2)
    assert err < 1e-3
