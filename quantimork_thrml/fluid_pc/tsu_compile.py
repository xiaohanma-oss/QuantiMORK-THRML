"""
TSU factor-graph compilation of the FluidPC density-update step.

Phase 1 scope (this module):
    The density update is linear in rho given a clamped velocity u:
        rho_new = rho - dt * div(u ⊙ rho_upwind) - dt * kappa * L * rho
                = M @ rho     where  M = I - dt * A_upwind - dt * kappa * L

    A_upwind is computed on the host using the upwind direction signs, then
    `target = M @ rho_init` is plugged in as the bias of a single Gaussian
    block. The factor graph posterior mean then equals `target`, matching
    `ConservativeAdvection.forward` exactly up to sampling noise.

This mirrors `pmode_verify.build_pmode_level_graph` (single-block Gaussian
with `W @ x` absorbed into the bias) but uses our wavelet-graph topology
instead of WaveletLinear's per-level matrix.

Phase 2 (deferred): velocity-solve graph (bilinear u<->p coupling for Leray
projection), reaction graph (cross-level couplings via WaveletLinear's
extract_energy_params), and full K-iteration alternation.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    SamplingSchedule,
    sample_states,
)
from thrml.factor import FactorSamplingProgram

from quantimork_thrml.gaussian_ebm import (
    ContinuousNode,
    CouplingFactor,
    GaussianSampler,
    LinearFactor,
    QuadraticFactor,
)
from quantimork_thrml.fluid_pc.topology import FluidGraphTopology


DEFAULT_PRECISION = 1.0
DEFAULT_SCHEDULE = SamplingSchedule(
    n_warmup=200, n_samples=2000, steps_per_sample=3)
DEFAULT_N_BATCHES = 200
MAX_NODES = 16


def host_density_update(
    topology: FluidGraphTopology,
    rho_init: np.ndarray,
    u: np.ndarray,
    dt: float,
    kappa: float,
) -> np.ndarray:
    """Compute the analytic next-step density on the host (numpy).

    Mirrors `quantimork_thrml.fluid_pc.advection.ConservativeAdvection.forward`
    but in numpy so it can be plugged into a thrml LinearFactor bias.
    """
    src = np.asarray(topology.edge_src, dtype=np.int64)
    dst = np.asarray(topology.edge_dst, dtype=np.int64)
    rho_src = rho_init[src]
    rho_dst = rho_init[dst]
    upwind_rho = np.where(u >= 0, rho_src, rho_dst)
    phi = u * upwind_rho

    div_phi = np.zeros_like(rho_init)
    np.add.at(div_phi, src, phi)
    np.add.at(div_phi, dst, -phi)

    grad_rho = rho_init[src] - rho_init[dst]
    lap_rho = np.zeros_like(rho_init)
    np.add.at(lap_rho, src, grad_rho)
    np.add.at(lap_rho, dst, -grad_rho)

    return rho_init - dt * div_phi - dt * kappa * lap_rho


def build_density_update_graph(
    topology: FluidGraphTopology,
    rho_init: np.ndarray,
    u_clamped: np.ndarray,
    dt: float,
    kappa: float,
    precision: float = DEFAULT_PRECISION,
    max_nodes: int = MAX_NODES,
) -> dict:
    """Build a Gaussian factor graph whose posterior mean is the next-step
    density given clamped u, dt, kappa.

    Args:
        topology: FluidGraphTopology defining nodes/edges.
        rho_init: (|V|,) numpy array — clamped initial density.
        u_clamped: (|E|,) numpy array — clamped velocity field.
        dt: scalar time step.
        kappa: scalar diffusion coefficient.
        precision: PC quadratic precision A_ii (free node self-precision).
        max_nodes: cap on the number of free nodes (matches pmode_verify
            convention; restricted to keep verification tractable).

    Returns:
        dict with `program`, `spec`, `target` (the analytic next-step density
        truncated to the first `max_nodes` entries), `n_nodes`.
    """
    target_full = host_density_update(
        topology, rho_init, u_clamped, dt, kappa)

    n_free = min(max_nodes, topology.n_nodes)
    target = target_full[:n_free].astype(np.float32)

    # Diagonal precision for each free output node.
    prec_diag = jnp.full((n_free,), float(precision), dtype=jnp.float32)
    # Linear bias absorbs the clamped target: b_i = -prec * target_i so the
    # posterior mean is target_i (analogous to pmode_verify's b = -prec*Wx).
    bias = -precision * jnp.asarray(target, dtype=jnp.float32)

    out_nodes = [ContinuousNode() for _ in range(n_free)]
    out_block = Block(out_nodes)

    quad_fac = QuadraticFactor(1.0 / prec_diag, out_block)
    lin_fac = LinearFactor(bias, out_block)

    sampler = GaussianSampler()
    spec = BlockGibbsSpec(
        [out_block], [],
        {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)},
    )
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=[quad_fac, lin_fac],
        other_interaction_groups=[],
    )

    return {
        "program": program,
        "spec": spec,
        "out_block": out_block,
        "n_nodes": n_free,
        "target": target,
        "precision": precision,
    }


def run_density_update_verification(
    graph: dict,
    seed: int = 0,
    n_batches: int = DEFAULT_N_BATCHES,
    schedule: SamplingSchedule = DEFAULT_SCHEDULE,
) -> dict:
    """Sample the density-update factor graph and return MSE versus the
    analytic next-step density.
    """
    spec = graph["spec"]
    prog = graph["program"]
    target = graph["target"]
    n_free = graph["n_nodes"]

    key = jax.random.key(seed)
    init_state = []
    for block in spec.free_blocks:
        key, sk = jax.random.split(key)
        init_state.append(
            0.1 * jax.random.normal(sk, (n_batches, len(block.nodes))))
    keys = jax.random.split(key, n_batches)
    observe_blocks = list(spec.free_blocks)

    samples = jax.jit(jax.vmap(
        lambda s, c, k_: sample_states(k_, prog, schedule, s, c, observe_blocks)
    ))(init_state, [], keys)

    sampled_means = []
    for i in range(n_free):
        node_samples = samples[0][:, :, i]
        sampled_means.append(float(jnp.mean(node_samples)))
    sampled_means = jnp.asarray(sampled_means, dtype=jnp.float32)

    target_jax = jnp.asarray(target, dtype=jnp.float32)
    mse = float(jnp.mean((sampled_means - target_jax) ** 2))
    return {
        "sampled_values": [float(v) for v in sampled_means],
        "target_values": [float(v) for v in target],
        "mse": mse,
        "n_nodes": n_free,
    }


def host_leray_projection(
    topology: FluidGraphTopology,
    u_raw: np.ndarray,
    lambda_div: float,
) -> np.ndarray:
    """Analytic posterior mean of `(I + lambda * B B^T)^{-1} u_raw`.

    As lambda -> infinity this approaches the Leray projection of `u_raw`
    onto the divergence-free subspace.
    """
    n_e = topology.n_edges
    n_v = topology.n_nodes
    src = np.asarray(topology.edge_src, dtype=np.int64)
    dst = np.asarray(topology.edge_dst, dtype=np.int64)
    B = np.zeros((n_e, n_v), dtype=np.float32)
    np.add.at(B, (np.arange(n_e), src), 1.0)
    np.add.at(B, (np.arange(n_e), dst), -1.0)
    BBt = B @ B.T
    sys = np.eye(n_e, dtype=np.float32) + lambda_div * BBt
    return np.linalg.solve(sys, u_raw.astype(np.float32))


def build_velocity_solve_graph(
    topology: FluidGraphTopology,
    u_raw: np.ndarray,
    lambda_div: float = 10.0,
    precision: float = 1.0,
    max_nodes: int = MAX_NODES,
) -> dict:
    """Build a Gaussian factor graph for the Leray velocity-solve step.

    Energy:  E(u) = 0.5 * prec * ||u - u_raw||^2
                  + 0.5 * lambda * ||B^T u||^2
                  = 0.5 * u^T (prec * I + lambda * B B^T) u - prec * u^T u_raw

    Encoded on a single `u_block` of size `max_nodes`:
      * `QuadraticFactor` diagonal precision  prec + lambda * 2
        (each edge contributes |+1|^2 + |-1|^2 = 2 to (BB^T)_ee).
      * `LinearFactor` bias  -prec * u_raw  (so posterior mean shifts to u_raw).
      * `CouplingFactor` per off-diagonal (e1, e2) of BB^T (edges sharing a
        node), weight  +/- lambda  depending on whether the shared endpoint
        carries the same incidence sign on both edges.

    Posterior mean = (prec*I + lambda*B B^T)^{-1} * (prec * u_raw).
    """
    src_full = np.asarray(topology.edge_src, dtype=np.int64)
    dst_full = np.asarray(topology.edge_dst, dtype=np.int64)
    n_free = min(max_nodes, topology.n_edges)
    src = src_full[:n_free]
    dst = dst_full[:n_free]
    u_raw_clip = u_raw[:n_free].astype(np.float32)

    pair_a = []
    pair_b = []
    pair_w = []
    for e1 in range(n_free):
        for e2 in range(e1 + 1, n_free):
            shared = 0.0
            if src[e1] == src[e2]:
                shared += 1.0
            if dst[e1] == dst[e2]:
                shared += 1.0
            if src[e1] == dst[e2]:
                shared -= 1.0
            if dst[e1] == src[e2]:
                shared -= 1.0
            if shared != 0.0:
                pair_a.append(e1)
                pair_b.append(e2)
                pair_w.append(float(lambda_div) * shared)

    diag_prec = float(precision) + 2.0 * float(lambda_div)
    prec_diag = jnp.full((n_free,), diag_prec, dtype=jnp.float32)
    bias = -float(precision) * jnp.asarray(u_raw_clip, dtype=jnp.float32)

    u_nodes = [ContinuousNode() for _ in range(n_free)]
    u_block = Block(u_nodes)
    quad_fac = QuadraticFactor(1.0 / prec_diag, u_block)
    lin_fac = LinearFactor(bias, u_block)
    factors = [quad_fac, lin_fac]
    if pair_a:
        view_a = Block([u_nodes[i] for i in pair_a])
        view_b = Block([u_nodes[i] for i in pair_b])
        coup = CouplingFactor(
            jnp.asarray(pair_w, dtype=jnp.float32),
            (view_a, view_b),
        )
        factors.append(coup)

    sampler = GaussianSampler()
    spec = BlockGibbsSpec(
        [u_block], [],
        {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)},
    )
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=factors,
        other_interaction_groups=[],
    )

    sub_topo = FluidGraphTopology(
        S=topology.S, D=topology.D, n_levels=topology.n_levels,
        band_sizes=topology.band_sizes, band_offsets=topology.band_offsets,
        n_nodes=topology.n_nodes,
        edge_src=tuple(int(s) for s in src),
        edge_dst=tuple(int(d) for d in dst),
        n_lateral=min(topology.n_lateral, n_free),
        n_cross=max(0, n_free - topology.n_lateral),
    )
    target = host_leray_projection(sub_topo, u_raw_clip,
                                   lambda_div=lambda_div)
    return {
        "program": program,
        "spec": spec,
        "u_block": u_block,
        "n_nodes": n_free,
        "target": np.asarray(target, dtype=np.float32),
        "lambda": lambda_div,
        "precision": precision,
    }


def run_velocity_solve_verification(
    graph: dict,
    seed: int = 0,
    n_batches: int = DEFAULT_N_BATCHES,
    schedule: SamplingSchedule = DEFAULT_SCHEDULE,
) -> dict:
    """Sample the velocity-solve graph and report MSE versus the analytic
    posterior mean.
    """
    spec = graph["spec"]
    prog = graph["program"]
    target = graph["target"]
    n_free = graph["n_nodes"]

    key = jax.random.key(seed)
    init_state = []
    for block in spec.free_blocks:
        key, sk = jax.random.split(key)
        init_state.append(
            0.1 * jax.random.normal(sk, (n_batches, len(block.nodes))))
    keys = jax.random.split(key, n_batches)
    observe_blocks = list(spec.free_blocks)

    samples = jax.jit(jax.vmap(
        lambda s, c, k_: sample_states(k_, prog, schedule, s, c, observe_blocks)
    ))(init_state, [], keys)

    sampled = []
    for i in range(n_free):
        sampled.append(float(jnp.mean(samples[0][:, :, i])))
    sampled = jnp.asarray(sampled, dtype=jnp.float32)
    target_jax = jnp.asarray(target, dtype=jnp.float32)
    mse = float(jnp.mean((sampled - target_jax) ** 2))
    return {
        "sampled_values": [float(v) for v in sampled],
        "target_values": [float(v) for v in target],
        "mse": mse,
        "n_nodes": n_free,
    }


def compile_fluid_pc_iteration(
    topology: FluidGraphTopology,
    rho: np.ndarray,
    u_raw: np.ndarray,
    dt: float,
    kappa: float,
    lambda_div: float = 100.0,
    precision: float = 1.0,
    max_nodes: int = MAX_NODES,
    n_batches: int = DEFAULT_N_BATCHES,
    seed: int = 0,
    schedule: SamplingSchedule = DEFAULT_SCHEDULE,
) -> dict:
    """One full FluidPC inner iteration end-to-end on TSU:
        Step 1: Leray velocity-solve  -> u (divergence-free)
        Step 2: density update        -> rho_new

    Returns the rho_new posterior mean (truncated to `max_nodes`) plus per
    -stage MSE diagnostics. The returned `rho_new` can be fed into the next
    call to chain K iterations.
    """
    v_graph = build_velocity_solve_graph(
        topology, u_raw, lambda_div=lambda_div,
        precision=precision, max_nodes=max_nodes)
    v_res = run_velocity_solve_verification(
        v_graph, seed=seed, n_batches=n_batches, schedule=schedule)
    u_proj = np.asarray(v_res["sampled_values"], dtype=np.float32)

    n_e = len(topology.edge_src)
    if u_proj.shape[0] < n_e:
        u_full = np.zeros(n_e, dtype=np.float32)
        u_full[: u_proj.shape[0]] = u_proj
    else:
        u_full = u_proj[:n_e]

    d_graph = build_density_update_graph(
        topology, rho, u_full, dt=dt, kappa=kappa,
        precision=precision, max_nodes=max_nodes)
    d_res = run_density_update_verification(
        d_graph, seed=seed + 1, n_batches=n_batches, schedule=schedule)

    return {
        "u_proj": u_proj,
        "rho_new": np.asarray(d_res["sampled_values"], dtype=np.float32),
        "velocity_mse": v_res["mse"],
        "density_mse": d_res["mse"],
    }


__all__ = [
    "build_density_update_graph",
    "build_velocity_solve_graph",
    "compile_fluid_pc_iteration",
    "run_density_update_verification",
    "run_velocity_solve_verification",
    "host_density_update",
    "host_leray_projection",
    "DEFAULT_PRECISION",
    "DEFAULT_SCHEDULE",
    "DEFAULT_N_BATCHES",
]
