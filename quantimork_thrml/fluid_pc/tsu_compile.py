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


__all__ = [
    "build_density_update_graph",
    "run_density_update_verification",
    "host_density_update",
    "DEFAULT_PRECISION",
    "DEFAULT_SCHEDULE",
    "DEFAULT_N_BATCHES",
]
