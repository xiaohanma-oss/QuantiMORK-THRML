"""TSU factor-graph verification of the FluidPC density-update step."""

import numpy as np
import pytest

from quantimork_thrml.fluid_pc.topology import build_topology
from quantimork_thrml.fluid_pc.tsu_compile import (
    build_density_update_graph,
    host_density_update,
    run_density_update_verification,
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
