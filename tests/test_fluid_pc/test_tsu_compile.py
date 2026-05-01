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
    compile_fluid_pc_iteration,
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


@pytest.mark.slow
def test_reaction_step_posterior_matches_torch_predictor():
    import torch
    from quantimork_thrml.fluid_pc.cross_scale_predictor import CrossScalePredictor
    from quantimork_thrml.fluid_pc.graph import WaveletGraph
    from quantimork_thrml.fluid_pc.tsu_compile import (
        build_reaction_step_graph, run_reaction_verification,
    )
    np.random.seed(13)
    topo = build_topology(S=4, D=8, n_levels=2)
    g = WaveletGraph(S=4, D=8, n_levels=2)
    pred = CrossScalePredictor(g)
    rho_np = np.abs(np.random.randn(topo.n_nodes).astype(np.float32))
    rho_np /= rho_np.sum()
    eta = 0.1
    with torch.no_grad():
        a_hat = pred(torch.from_numpy(rho_np)[None])
        torch_target = (torch.from_numpy(rho_np)[None]
                        - eta * (torch.from_numpy(rho_np)[None] - a_hat))
    graph = build_reaction_step_graph(
        topo, rho_np,
        theta_lat=float(pred.theta_lat),
        theta_up=float(pred.theta_up),
        theta_down=float(pred.theta_down),
        bias=float(pred.bias),
        eta=eta, precision=2.0, max_nodes=8)
    r = run_reaction_verification(graph, seed=13, n_batches=200)
    assert r["mse"] < 0.01
    np.testing.assert_allclose(
        graph["target"], torch_target[0, :8].numpy(), atol=1e-5)


@pytest.mark.slow
def test_compile_fluid_pc_iteration_runs_end_to_end():
    np.random.seed(3)
    topo = build_topology(S=4, D=8, n_levels=2)
    rho = np.abs(np.random.randn(topo.n_nodes).astype(np.float32))
    rho /= rho.sum()
    u_raw = 0.3 * np.random.randn(topo.n_edges).astype(np.float32)
    out = compile_fluid_pc_iteration(
        topo, rho, u_raw, dt=0.1, kappa=0.01,
        lambda_div=100.0, max_nodes=8, n_batches=200)
    assert out["velocity_mse"] < 0.05
    assert out["density_mse"] < 0.05
    assert out["rho_new"].shape == (8,)


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
