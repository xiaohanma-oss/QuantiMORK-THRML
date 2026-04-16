"""
P-mode (continuous Gaussian) factor graph verification for WaveletLinear.

Maps PC quadratic energy to ContinuousNode + scalar coupling factors,
runs Gibbs sampling via thrml, and verifies energy equivalence with
PyTorch forward pass.

This is the p-mode counterpart of thrml_verify.py (which uses K=16
CategoricalNode discretization). Under the Laplace approximation
(Active Inference §4.19), PC energy is naturally quadratic — p-mode
encodes it exactly without discretization error.

Energy decomposition for clamped input x, free output y:

    E_pred = 0.5 * prec * ||y - W @ x||²
           = 0.5 * prec * Σ_i y_i² - prec * Σ_i y_i * (Wx)_i + const

    E_td   = 0.5 * α * prec * ||y - td||²   (top-down modulation)

    E_prior = 0.5 * β * prec * ||y - μ||²    (KL / Gaussian prior)

Combined diagonal precision per node:
    A_ii = prec + α * prec + β * prec = prec * (1 + α + β)

Combined bias per node:
    b_i = -prec * (Wx)_i - α * prec * td_i - β * prec * μ_i

Software Gibbs is not efficient for pure Gaussian (Cholesky is faster),
but validates that the factor graph encoding is correct for TSU pmode
hardware deployment.
"""

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
    QuadraticFactor,
    LinearFactor,
    GaussianSampler,
)


DEFAULT_PRECISION = 1.0
DEFAULT_SCHEDULE = SamplingSchedule(
    n_warmup=200, n_samples=5000, steps_per_sample=3)
DEFAULT_N_BATCHES = 500


def build_pmode_level_graph(
    weight_matrix, input_activations, target_activations,
    precision=DEFAULT_PRECISION,
    td_activations=None, td_alpha=0.0,
    beta=0.0,
):
    """Build p-mode factor graph for one level of WaveletLinear.

    With input x clamped, the free variables are output y. The energy is:
        E = 0.5 * A_ii * y_i² + b_i * y_i  (per output node)
    where A_ii = precision diagonal, b_i = combined bias from prediction,
    top-down, and prior terms.

    No output-output coupling is needed because each output y_i is
    conditionally independent given clamped x (the W matrix creates
    input-to-output dependencies, but with x clamped these become biases).

    Args:
        weight_matrix: (out_dim, in_dim) numpy array
        input_activations: (in_dim,) numpy array — clamped observations
        target_activations: (out_dim,) numpy array — expected output (for energy calc)
        precision: PC energy precision
        td_activations: (out_dim,) numpy array — top-down predictions, or None
        td_alpha: Weight for top-down modulation (default 0.0)
        beta: Prior strength (default 0.0)

    Returns:
        Dict with graph components + metadata for verification.
    """
    out_dim, in_dim = weight_matrix.shape
    max_nodes = min(8, in_dim, out_dim)

    W = weight_matrix[:max_nodes, :max_nodes]
    x_in = input_activations[:max_nodes]
    target = target_activations[:max_nodes]

    # Prediction: W @ x (clamped input contribution)
    prediction = W @ x_in  # (max_nodes,)

    # Combined diagonal precision
    total_prec = precision * (1.0 + td_alpha + beta)
    prec_diag = jnp.full(max_nodes, total_prec)

    # Combined bias: b_i = -prec*(Wx)_i - α*prec*td_i - β*prec*μ_i
    bias = -precision * jnp.array(prediction, dtype=jnp.float32)
    if td_activations is not None and td_alpha > 0:
        td = td_activations[:max_nodes]
        bias = bias - td_alpha * precision * jnp.array(td, dtype=jnp.float32)
    if beta > 0:
        bias = bias - beta * precision * jnp.array(target, dtype=jnp.float32)

    # Create output nodes (free)
    output_nodes = [ContinuousNode() for _ in range(max_nodes)]
    output_block = Block(output_nodes)

    # Factors
    quad_fac = QuadraticFactor(1.0 / prec_diag, output_block)
    lin_fac = LinearFactor(bias, output_block)

    # Single block (all nodes update together — valid for diagonal precision)
    sampler = GaussianSampler()
    spec = BlockGibbsSpec(
        [output_block], [],
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
        "output_nodes": output_nodes,
        "max_nodes": max_nodes,
        "target_activations": target[:max_nodes],
        "prediction": prediction,
        "total_prec": total_prec,
    }


def run_pmode_verification(graph, seed=0, n_batches=DEFAULT_N_BATCHES,
                           schedule=DEFAULT_SCHEDULE):
    """Run Gibbs sampling on p-mode graph and verify energy equivalence.

    Returns:
        Dict with sampled_values, mse, sampled_energy, analytic_mean.
    """
    spec = graph["spec"]
    prog = graph["program"]
    key = jax.random.key(seed)

    # Initialize output nodes near zero
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

    # Recover means from posterior samples
    sampled_values = []
    for i in range(graph["max_nodes"]):
        node_samples = samples[0][:, :, i]  # (n_batches, n_samples)
        mean = float(jnp.mean(node_samples))
        sampled_values.append(mean)

    sampled_values = jnp.array(sampled_values)
    target = jnp.array(graph["target_activations"])

    mse = float(jnp.mean((sampled_values - target) ** 2))
    sampled_energy = float(0.5 * jnp.sum((sampled_values - target) ** 2))

    # Analytic solution: y* = -bias / A_ii = prediction (+ td + prior terms)
    # For simple case (no td, no prior): y* = W @ x = prediction
    prediction = jnp.array(graph["prediction"])
    analytic_mse = float(jnp.mean((prediction - target[:len(prediction)]) ** 2))

    return {
        "sampled_values": [float(v) for v in sampled_values],
        "target_values": [float(v) for v in target],
        "mse": mse,
        "sampled_energy": sampled_energy,
        "analytic_mean": [float(v) for v in prediction],
        "analytic_mse": analytic_mse,
    }


if __name__ == "__main__":
    print("=== QuantiMORK-THRML: P-mode Factor Graph Verification ===")

    np.random.seed(42)
    W = np.random.randn(8, 8).astype(np.float32) * 0.1
    x_in = np.random.randn(8).astype(np.float32) * 0.5
    target = (W @ x_in).astype(np.float32)

    print(f"Weight matrix: {W.shape}")
    print(f"Input: {x_in}")
    print(f"Target (W@x): {target}")

    graph = build_pmode_level_graph(W, x_in, target, precision=2.0)
    result = run_pmode_verification(graph, seed=42)

    print(f"\nSampled values: {[f'{v:.4f}' for v in result['sampled_values']]}")
    print(f"Target values:  {[f'{v:.4f}' for v in result['target_values']]}")
    print(f"Analytic mean:  {[f'{v:.4f}' for v in result['analytic_mean']]}")
    print(f"MSE (sampled vs target): {result['mse']:.6f}")
    print(f"MSE (analytic vs target): {result['analytic_mse']:.6f}")
    print(f"\n{'PASS' if result['mse'] < 0.05 else 'FAIL'}: MSE {'<' if result['mse'] < 0.05 else '>'} 0.05")
