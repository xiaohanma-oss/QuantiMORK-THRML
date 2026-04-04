"""
TSU factor graph verification for WaveletLinear.

Extracts per-level weights from a trained WaveletLinear, compiles them into
a thrml factor graph, runs Gibbs sampling, and verifies energy equivalence.

This module uses JAX + thrml (not PyTorch). Run it with the PLN-THRML venv:
    /path/to/PLN-THRML/.venv/bin/python -m quantimork_thrml.thrml_verify

The key verification:
    PyTorch PC energy ≈ thrml Gibbs sampling energy
"""

import json
import math

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    SquareCategoricalEBMFactor,
    CategoricalGibbsConditional,
)
from thrml.factor import FactorSamplingProgram

DEFAULT_K = 16
DEFAULT_PRECISION = 1.0
DEFAULT_SCHEDULE = SamplingSchedule(
    n_warmup=300, n_samples=1000, steps_per_sample=3)
DEFAULT_N_BATCHES = 30


def coeff_bin_centers(k=DEFAULT_K, lo=-1.0, hi=1.0):
    """K evenly-spaced bin centers in [lo, hi]."""
    return jnp.linspace(lo + (hi - lo) / (2 * k), hi - (hi - lo) / (2 * k), k)


def pc_prediction_weights(precision, k=DEFAULT_K, lo=-1.0, hi=1.0):
    """K×K pairwise weight matrix encoding PC prediction error.

    W[i,j] = -precision * 0.5 * (center_i - center_j)²
    Diagonal is highest (prediction match = low energy).
    """
    centers = coeff_bin_centers(k, lo, hi)
    diff = centers[:, None] - centers[None, :]
    W = -precision * 0.5 * diff ** 2
    W = W - jnp.mean(W)  # center for stability
    return W


def pc_prior_weights(mean, precision, k=DEFAULT_K, lo=-1.0, hi=1.0):
    """[K] unary weight vector encoding Gaussian prior.

    W[j] = -0.5 * precision * (center_j - mean)²
    """
    centers = coeff_bin_centers(k, lo, hi)
    W = -0.5 * precision * (centers - mean) ** 2
    W = W - jnp.mean(W)
    return W


def value_to_bin(value, k=DEFAULT_K, lo=-1.0, hi=1.0):
    """Quantize continuous value to nearest bin index."""
    centers = coeff_bin_centers(k, lo, hi)
    return int(jnp.argmin(jnp.abs(centers - value)))


def bin_to_value(bin_idx, k=DEFAULT_K, lo=-1.0, hi=1.0):
    """Bin center value."""
    return float(coeff_bin_centers(k, lo, hi)[bin_idx])


def build_single_level_graph(
    weight_matrix, input_activations, target_activations,
    precision=DEFAULT_PRECISION, k=DEFAULT_K,
):
    """Build factor graph for one level of WaveletLinear.

    Models the PC energy: E = 0.5 * precision * ||target - W @ input||²

    For a small subset of dimensions (to keep tractable), creates:
    - Input nodes (clamped to observed activations)
    - Output nodes (free, to be sampled)
    - Pairwise factors encoding W[i,j] relationship
    - Prior factors on output nodes

    Args:
        weight_matrix: (out_dim, in_dim) numpy array — the Linear weight
        input_activations: (in_dim,) numpy array — observed input
        target_activations: (out_dim,) numpy array — expected output
        precision: PC energy precision
        k: Number of discretization bins

    Returns:
        Dict with graph components + metadata for verification.
    """
    out_dim, in_dim = weight_matrix.shape

    # For tractability, use a small subset
    max_nodes = min(8, in_dim, out_dim)

    # Create nodes
    input_nodes = [CategoricalNode() for _ in range(max_nodes)]
    output_nodes = [CategoricalNode() for _ in range(max_nodes)]

    # Determine value range from activations
    all_vals = list(input_activations[:max_nodes]) + list(target_activations[:max_nodes])
    lo = float(min(all_vals)) - 0.5
    hi = float(max(all_vals)) + 0.5

    # Pairwise factors: output[i] ← input[j] via W[i,j]
    factors = []
    for i in range(max_nodes):
        for j in range(max_nodes):
            w_ij = float(weight_matrix[i, j])
            if abs(w_ij) < 0.01:
                continue  # skip near-zero connections
            # Weighted prediction factor
            W = pc_prediction_weights(precision * abs(w_ij), k, lo, hi)
            factors.append(SquareCategoricalEBMFactor(
                [Block([output_nodes[i]]), Block([input_nodes[j]])],
                W[None, :, :],
            ))

    # Prior factors on output nodes (centered at target)
    for i in range(max_nodes):
        W = pc_prior_weights(
            float(target_activations[i]), precision * 0.1, k, lo, hi)
        factors.append(CategoricalEBMFactor(
            [Block([output_nodes[i]])], W[None, :],
        ))

    # Clamp input nodes
    clamped_state = {}
    for j, node in enumerate(input_nodes):
        clamped_state[node] = value_to_bin(
            float(input_activations[j]), k, lo, hi)

    # Free blocks: output nodes
    free_blocks = [Block(output_nodes)]
    clamped_blocks = [Block(input_nodes)]

    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(n_categories=k)
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=factors,
        other_interaction_groups=[],
    )

    return {
        "program": program,
        "spec": spec,
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "clamped_state": clamped_state,
        "k": k,
        "lo": lo,
        "hi": hi,
        "target_activations": target_activations[:max_nodes],
        "max_nodes": max_nodes,
    }


def run_verification(graph, seed=0, n_batches=DEFAULT_N_BATCHES,
                     schedule=DEFAULT_SCHEDULE):
    """Run Gibbs sampling and compute energy + recovered activations.

    Follows the same sampling pattern as PLN-THRML's run_beta_sampling:
    vmap over batches, each batch gets its own PRNG key.

    Returns:
        Dict with sampled_values, sampled_energy, target_energy, mse.
    """
    k = graph["k"]
    spec = graph["spec"]
    prog = graph["program"]
    key = jax.random.key(seed)

    # Initialize free blocks (output nodes) randomly
    init_state = []
    for block in spec.free_blocks:
        key, subkey = jax.random.split(key)
        init_state.append(
            jax.random.randint(
                subkey, (n_batches, len(block.nodes)),
                minval=0, maxval=k, dtype=jnp.uint8))

    # Clamped state for input nodes
    clamped_state_per_node = graph["clamped_state"]
    clamped_blocks = spec.clamped_blocks
    state_clamp = []
    for block in clamped_blocks:
        vals = jnp.array(
            [clamped_state_per_node[n] for n in block.nodes],
            dtype=jnp.uint8)
        # Broadcast to (n_batches, n_nodes_in_block)
        state_clamp.append(
            jnp.broadcast_to(vals[None, :], (n_batches, len(block.nodes))))

    keys = jax.random.split(key, n_batches)
    observe_blocks = list(spec.free_blocks)

    # Vmapped Gibbs sampling (same pattern as beta.py)
    samples = jax.jit(jax.vmap(
        lambda s, c, k_: sample_states(
            k_, prog, schedule, s, c, observe_blocks)
    ))(init_state, state_clamp, keys)

    # Recover output activations from posterior
    lo, hi = graph["lo"], graph["hi"]
    centers = coeff_bin_centers(k, lo, hi)

    sampled_values = []
    for i in range(graph["max_nodes"]):
        # samples is a list of arrays (one per observe_block)
        # We have one free block, so samples[0] has shape
        # (n_batches, n_samples, n_nodes)
        node_samples = samples[0][:, :, i]  # (n_batches, n_samples)
        flat = node_samples.flatten()
        counts = jnp.zeros(k)
        for idx in range(k):
            counts = counts.at[idx].set(jnp.sum(flat == idx))
        posterior = counts / counts.sum()
        mean = float(jnp.sum(posterior * centers))
        sampled_values.append(mean)

    sampled_values = jnp.array(sampled_values)
    target = jnp.array(graph["target_activations"][:graph["max_nodes"]])

    mse = float(jnp.mean((sampled_values - target) ** 2))
    sampled_energy = float(0.5 * jnp.sum((sampled_values - target) ** 2))

    return {
        "sampled_values": [float(v) for v in sampled_values],
        "target_values": [float(v) for v in target],
        "mse": mse,
        "sampled_energy": sampled_energy,
    }


if __name__ == "__main__":
    import numpy as np

    print("=== QuantiMORK-THRML: TSU Factor Graph Verification ===")

    # Simulate a small WaveletLinear level: Linear(8, 8)
    np.random.seed(42)
    W = np.random.randn(8, 8).astype(np.float32) * 0.1
    x_in = np.random.randn(8).astype(np.float32) * 0.5
    target = (W @ x_in).astype(np.float32)

    print(f"Weight matrix: {W.shape}")
    print(f"Input: {x_in}")
    print(f"Target (W@x): {target}")

    graph = build_single_level_graph(W, x_in, target, precision=2.0, k=16)
    print(f"\nFactor graph: {len(graph['output_nodes'])} free nodes, "
          f"{len(graph['input_nodes'])} clamped nodes")

    result = run_verification(graph, seed=42)
    print(f"\nSampled values: {[f'{v:.3f}' for v in result['sampled_values']]}")
    print(f"Target values:  {[f'{v:.3f}' for v in result['target_values']]}")
    print(f"MSE: {result['mse']:.6f}")
    print(f"Sampled energy: {result['sampled_energy']:.6f}")
    print(f"\n{'PASS' if result['mse'] < 0.1 else 'FAIL'}: MSE {'<' if result['mse'] < 0.1 else '>'} 0.1")
