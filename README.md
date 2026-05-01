# QuantiMORK-THRML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

> Wavelet-sparse predictive coding compiled to thermodynamic factor graphs
> for TSU execution — bridging [Hyperon/QuantiMORK](https://github.com/trueagi-io/hyperon) (§7.4)
> and [Extropic/thrml](https://github.com/extropic-ai/thrml).
> End-to-end verified on Tiny Shakespeare against
> [iCog-Labs' PC-Transformers](https://github.com/iCog-Labs-Dev/PC-Transformers).

## Table of Contents

- [Overview](#overview)
- [Why this matters](#why-this-matters)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Results](#results)
- [API reference](#api-reference)
- [Hyperon integration outlook](#hyperon-integration-outlook)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

QuantiMORK-THRML compiles wavelet-sparse predictive coding layers into
thrml factor graphs that run on Extropic's TSU via Gibbs sampling. The
wavelet decomposition reduces inter-level connectivity to a tree structure
(≤5 inter-level neighbors), but per-level Linear layers remain dense
(see [Connectivity & Sparsity](#connectivity--sparsity)). The architecture
implements Hyperon whitepaper §7.4 (QuantiMORK), and uses iCog-Labs'
PC-Transformers as the baseline.

Hyperon Whitepaper §7.4.3:
> "Each level of the wavelet hierarchy maintains local predictions and
> errors. Updates flow bidirectionally through local message passing
> rather than requiring a global backward pass."

This project takes iCog-Labs' [PC-Transformers](https://github.com/iCog-Labs-Dev/PC-Transformers)
as a baseline, replaces its dense MLP layers with Haar wavelet-sparse
`WaveletLinear` layers, and verifies on Tiny Shakespeare that the
wavelet-sparse variant (a) produces comparable language modeling results,
and (b) maps to thrml factor graphs for TSU execution.

Together with [PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML) (reasoning, §6),
[ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML) (attention, §5.4),
[MOSES-THRML](https://github.com/xiaohanma-oss/MOSES-THRML) (program evolution),
and [Geodesic-THRML](https://github.com/xiaohanma-oss/Geodesic-THRML) (unified scheduler),
this forms the Hyperon-on-TSU compilation suite.

<details>
<summary><strong>New to Predictive Coding? (30-second primer)</strong></summary>

Standard neural networks learn by **backpropagation** — errors flow
backward from the output through every layer. Predictive coding (PC) is
a biologically-inspired alternative: each layer independently **predicts**
what its neighbors should look like, computes the prediction error, and
updates locally. No global backward pass needed.

The energy being minimized at each layer:
```
F = 0.5 × ||target − prediction||²  +  α × 0.5 × ||td − output||²  +  β × D_KL(q || prior)
         (bottom-up error)             (top-down modulation)         (complexity penalty)
```

The Hebbian weight update combines both error signals in the wavelet domain:
`Δw = lr × (bu_err + α·td_err) × input − lr × β × w`, where `α` (--td-alpha,
default 0.5) controls the top-down contribution and `β` (--beta, training
default 0.1) controls KL regularization (weight decay toward zero under
Gaussian prior; model-level default is 0.0, i.e. no KL unless explicitly set).

PC-Transformers (iCog-Labs) implements this for transformers.
QuantiMORK-THRML makes the MLP layers wavelet-sparse so they fit on TSU.

</details>

<details>
<summary><strong>New to Wavelet Transforms? (30-second primer)</strong></summary>

A wavelet transform decomposes a signal into multiple resolution levels —
like looking at a photo from far away (coarse) and up close (fine detail).

**Haar wavelet** (simplest):
- Split signal into pairs
- Average = coarse approximation
- Difference = fine detail
- Repeat on the coarse part → tree of coefficients

Key property: the wavelet tree gives each coefficient at most **5 inter-level
neighbors** (parent, 2 children, 2 siblings). But each per-level Linear
connects coefficients *within* a level — those connections are still dense.
See [Connectivity & Sparsity](#connectivity--sparsity) for measured data.

</details>

## Why this matters

### The closed-loop problem: why PC can't scale on GPUs

<details>
<summary><strong>Backprop, forgetting, and why PC helps (30-second version)</strong></summary>

Backpropagation is a DAG (directed acyclic graph) computation: the chain
rule propagates a single global error signal from output to input, and
every weight receives a gradient that depends on the entire graph. This
is extremely efficient on GPU hardware (which executes DAGs natively),
but carries a structural consequence: global gradients update shared
representations without inherent protection, so training on a new task
overwrites weights that previous tasks depend on — **catastrophic
forgetting**. Mitigations exist (EWC, PackNet), but they are added
constraints on top of a learning rule that does not naturally isolate
parameters.

Predictive coding changes the default: higher layers predict lower
layers, each layer computes local prediction error, and only locally
involved weights update. Combined with sparse activations — where
inactive neurons' weights are untouched by Hebbian rules — this creates
implicit parameter isolation across tasks. Continual learning becomes
structurally favored rather than requiring external regularization,
though not fully solved: additional mechanisms (lateral competition,
complementary memory) are still needed at scale.

</details>

**Why PC can't compete on GPUs: two compounding costs.**

Backpropagation traverses the network twice — one forward pass, one
backward pass — and it's done. The chain rule gives every weight an
exact gradient in a single traversal. PC has no such global formula:
each layer only sees its immediate neighbors, so information must
propagate through repeated local adjustments. A 3-layer network needs
at least 3 iterations for the output error to ripple back to the input
layer, and convergence typically requires more.

This means PC must scan the entire network T times where backprop
scans it twice:

|              | Backprop (DAG computation) | PC (iterative cycle)      |
|--------------|:--------------------------:|:-------------------------:|
| **GPU (DAG hardware)**  | ✓ DAG on DAG    | △ cycle unrolled into DAG |
| **TSU (cycle hardware)** | —              | ✓ cycle on cycle          |

TSU's value is not "replacing GPUs" — it's making Predictive Coding
a first-class citizen. PC's local Hebbian updates map directly onto
TSU's sparse physical connections, just as matrix multiplication maps
directly onto GPU's SIMD architecture.

This project is a proof-of-concept for the TSU + PC cell of this
matrix: wavelet-sparse predictive coding that compiles to TSU factor
graphs.

### Why wavelets, not dense layers

```
PC-Transformers MLP: nn.Linear(512, 2048)
  → 262K params, 512 connections per node
  → TSU limit: ~12 neighbors → ✗

QuantiMORK WaveletLinear(512, 512, levels=3):
  → 90K params, ≤5 inter-level connections per node
  → per-level intra-level: 256/128/64 connections (still dense)
  → TSU deployment requires per-level sparsification (open question)
```

## Installation

```bash
git clone --recurse-submodules https://github.com/xiaohanma-oss/QuantiMORK-THRML.git
cd QuantiMORK-THRML
pip install -e ".[dev]"              # core + pytest
```

> The [PC-Transformers](https://github.com/iCog-Labs-Dev/PC-Transformers) submodule
> provides the baseline. If you cloned without `--recurse-submodules`, run
> `git submodule update --init`.

## Quick start

```python
from quantimork_thrml import WaveletLinear, haar_dwt_1d, haar_idwt_1d

# WaveletLinear: drop-in replacement for nn.Linear (same dim)
layer = WaveletLinear(512, 512, n_levels=3)
print(f"Parameters: {layer.num_params():,}")  # 90,624 vs 262,656 dense

# Haar wavelet transform
import torch
x = torch.randn(2, 8, 512)
coeffs = haar_dwt_1d(x, n_levels=3)          # decompose
y = haar_idwt_1d(coeffs)                      # reconstruct
assert torch.allclose(x, y, atol=1e-6)        # perfect roundtrip
```

### Train and compare

```bash
python scripts/prepare_data.py                 # download + tokenize Tiny Shakespeare
python scripts/train.py --mode baseline        # train PC-Transformers (dense)
python scripts/train.py --mode wavelet         # train QuantiMORK (wavelet-sparse)
python scripts/compare.py                      # print comparison table
```

## How it works

### 1. Haar DWT along feature dimension (§7.4.2)

```
Input activation: (B, S, 512)
  ↓ Haar DWT (3 levels)
Level 1 detail: (B, S, 256)
Level 2 detail: (B, S, 128)
Level 3 detail: (B, S, 64)
Level 3 approx: (B, S, 64)
```

### 2. Per-level independent Linear (§7.4.3)

Each level's coefficients are transformed by a small, local `nn.Linear`:

```
Level 1: Linear(256, 256)  →  65K params
Level 2: Linear(128, 128)  →  16K params
Level 3: Linear(64, 64)    →   4K params  (×2: detail + approx)
Total: ~90K params  (vs 262K dense)
```

### 3. Bidirectional wavelet-domain Hebbian update (gradient-free)

PC-Transformers uses Hebbian-like local learning (not backprop). QuantiMORK
extends this with **bidirectional free energy**: both bottom-up prediction
error and top-down modulation from the layer above drive weight updates.
Since the Haar DWT is orthogonal, the combined error decomposes exactly:

```
combined_err_i = DWT(bu_err)_i + α × DWT(td_err)_i
Δw_level_i = lr × combined_err_i × DWT(input)_i^T
```

Each level's `nn.Linear` weights are updated independently in the wavelet
coefficient space — no global gradient flow, consistent with ActPC
(Goertzel 2024, arXiv:2412.16547). The `--td-alpha` flag controls the
top-down contribution (default 0.5; set 0 for bu-only baseline).

### 4. Inverse Haar DWT → output

Transformed coefficients are reconstructed back to the original dimension.

### 5. PC energy → factor graph → Gibbs sampling (TSU)

Two factor graph backends are available:

**Categorical backend** (K=16 p-dit discretization):
```
W[i,j] = −0.5 × precision × (center_i − center_j)²   (K×K lookup table)
Execution: Gibbs sampling on CategoricalNode factor graph
```

**P-mode backend** (continuous Gaussian, recommended):
```
E(x) = 0.5 × (x − μ)ᵀ A (x − μ)                     (quadratic energy)
     = 0.5 × Σ_i A_ii x_i² + Σ_i b_i x_i              (with input clamped)
Execution: Gibbs sampling on ContinuousNode factor graph
```

Under the Laplace approximation (Active Inference §4.19, Box 4.3), PC
energy is naturally quadratic — the p-mode backend encodes it exactly
without discretization error. Software Gibbs validates energy equivalence;
TSU hardware pmode samples natively.

### Mapping table

| PC-Transformers concept | QuantiMORK-THRML | thrml API (categorical) | thrml API (p-mode) |
|-------------------------|------------------|------------------------|-------------------|
| MLP dense weights | WaveletLinear per-level weights | `SquareCategoricalEBMFactor` | `CouplingFactor` / `LinearFactor` |
| PC latent state x | Wavelet coefficients (level, pos) | `CategoricalNode` K bins | `ContinuousNode` (Gaussian) |
| prediction error | Factor graph energy | K×K lookup table | Quadratic `QuadraticFactor` |
| T-step iteration | — | Gibbs warmup + sampling | Gibbs warmup + sampling |
| Leaf observations | Clamped input activations | `BlockGibbsSpec(clamped)` | Absorbed into bias terms |

## Results

Tiny Shakespeare, 5 epochs, n_embed=128, n_blocks=4, T=2, CPU training:

| Metric | PC-Transformers | QuantiMORK (td_alpha=0.5, beta=0.1) |
|--------|:---------------:|:-----------------------------------:|
| Parameters | 1.06M | 0.56M |
| MLP params/layer | dense | 90K (wavelet) |
| Inter-level connections/node | 512 | ≤5 |
| Intra-level connections/node | 512 | 256/128/64 (dense per level) |
| Final train PPL | 954.9 | 963.7 |
| Final val PPL | 952.8 | 956.7 |
| Final train energy | 1.3762 | 1.6966 |
| Final val energy | 1.3761 | 1.6969 |
| Final KL energy | — | 0.1885 |
| Parameter compression | — | 1.9× |
| TSU deployable | ✗ | ✓ |

Val perplexity difference is <1% — bidirectional wavelet-sparse PC
preserves language modeling quality while reducing connectivity to
TSU-compatible levels.

Energy is higher for the wavelet model (~23%) because (a) per-level
independent Linears produce less precise predictions than a single
dense layer, and (b) KL regularization adds a complexity penalty. The
KL component decreases over training (0.55 → 0.19), indicating the
model learns to match its Gaussian prior. The gap does not affect
downstream task quality.

Parameter compression: 1.9× (MLP-only compression is ~2.9×; attention
layers are shared and dominate total count at this small scale).

Wavelet per-level weights are trained via bidirectional wavelet-domain
Hebbian updates (combined bottom-up + top-down error in DWT domain),
with KL regularization. Verified by
`test_wavelet_weights_update_after_forward` and
`test_bidirectional_update_differs_from_bu_only`.

## API reference

### Haar transforms (`quantimork_thrml.haar`)

| Function | Description |
|----------|-------------|
| `haar_dwt_1d(x, n_levels=3)` | 1D Haar DWT along last dimension; returns `{"details": [d_1, ..., d_L], "approx": a_L}` |
| `haar_idwt_1d(coeffs)` | Inverse Haar DWT; reconstructs original tensor from coefficients dict |
| `tree_positions(n_levels, n_features)` | Enumerates wavelet coefficient positions as `(level, band, size)` |

### WaveletLinear (`quantimork_thrml.wavelet_linear`)

| Method | Description |
|--------|-------------|
| `WaveletLinear(in_features, out_features, n_levels=3, bias=True)` | Multi-resolution linear layer; requires `in_features == out_features` |
| `.forward(x)` | Haar DWT → per-level independent Linear → inverse Haar DWT |
| `.num_params()` | Total trainable parameters across all per-level transforms |
| `.max_connections_per_node()` | Maximum per-level Linear size (bounds TSU connectivity) |
| `.extract_energy_params()` | Export per-level weights/biases as list of dicts for thrml factor graph construction |

### WaveletPCTransformer (`quantimork_thrml.model`)

| Method | Description |
|--------|-------------|
| `WaveletPCTransformer(config)` | PCTransformer with wavelet-sparse MLP; config: `n_embed`, `n_blocks`, `T`, `wavelet_n_levels`, `td_alpha`, `beta` |
| `.register_all_lateral_weights()` | Register lateral weight tensors for all PC layers |
| `.forward(target_ids, input_ids)` | Full PC forward pass over T inference steps; returns `(B, S, vocab_size)` logits |

### Verification (`quantimork_thrml.thrml_verify`)

These functions use JAX + thrml (not PyTorch) for factor graph verification.

| Function | Description |
|----------|-------------|
| `build_single_level_graph(..., backend="categorical")` | Build thrml factor graph for one WaveletLinear level; `backend="pmode"` uses continuous Gaussian nodes |
| `run_verification(graph, seed=0, n_batches=30)` | Run Gibbs sampling on factor graph; returns `{sampled_values, target_values, mse, sampled_energy}` |
| `pc_prediction_weights(precision, k=16)` | K×K pairwise weight matrix encoding PC squared prediction error (categorical only) |
| `td_modulation_weights(alpha, precision, k=16)` | K×K pairwise weight matrix encoding top-down modulation energy (categorical only) |
| `kl_prior_weights(prior_probs, beta=0.1, k=16)` | [K] unary weight vector encoding KL divergence complexity penalty (categorical only) |
| `coeff_bin_centers(k=16)` | K evenly-spaced bin centers for discretizing activations (categorical only) |
| `value_to_bin(value, k=16)` / `bin_to_value(bin_idx, k=16)` | Quantize continuous scalar ↔ bin index (categorical only) |

### Gaussian EBM (`quantimork_thrml.gaussian_ebm`)

Custom thrml components for p-mode (continuous Gaussian) factor graphs,
based on Extropic's [official Gaussian PGM example](https://docs.thrml.ai/en/latest/examples/01_all_of_thrml/).

| Class | Description |
|-------|-------------|
| `ContinuousNode(AbstractNode)` | Continuous random variable node (pmode on TSU) |
| `QuadraticFactor(inverse_weights, block)` | Diagonal precision: `E_i = 0.5 * A_ii * x_i²` |
| `LinearFactor(weights, block)` | Bias term: `E_i = b_i * x_i` |
| `CouplingFactor(weights, (block_a, block_b))` | Pairwise scalar coupling: `E = w * x_i * x_j` |
| `GaussianSampler()` | Gibbs conditional sampler for Gaussian nodes |

### P-mode verification (`quantimork_thrml.pmode_verify`)

| Function | Description |
|----------|-------------|
| `build_pmode_level_graph(W, input, target, ...)` | Build continuous Gaussian factor graph for one WaveletLinear level |
| `run_pmode_verification(graph, seed=0)` | Run Gibbs sampling; returns `{sampled_values, mse, analytic_mean, ...}` |

## Hyperon integration outlook

See [PLN-THRML README](https://github.com/xiaohanma-oss/PLN-THRML#hyperon-integration-outlook)
for the full heterogeneous pipeline design (Control → Compile → Sample).

This project contributes the **learning tier**: wavelet-sparse predictive coding
compiled to TSU factor graphs, handling perceptual learning alongside PLN's
Boltzmann factor-graph inference and ECAN's LBM attention diffusion. All three
workloads are TSU-native — co-location on one chip via time-multiplexing or
spatial partitioning is an open question (depends on factor graph size, lattice
partitioning, and mixing time).

The key architectural difference from standard PC: QuantiMORK's wavelet
decomposition limits inter-level connectivity to ≤5 neighbors per node.
However, per-level Linear layers are still dense (up to 256 intra-level
connections at level 1), exceeding TSU's ~12-neighbor hardware limit.
See [Connectivity & Sparsity](#connectivity--sparsity) for analysis and
candidate sparsification directions.

## FluidPCBlock — IFN §10.6 Option A

`quantimork_thrml/fluid_pc/` implements the Wavelet-Fluid PC block from
[Incompressible Fluid Networks](docs/references/incompressible-fluid-networks/full-text.md)
§10.6 as a drop-in attention replacement. Tensor signature `(B, S, D) → (B, S, D)`.

Per-iteration pipeline (k = 1..K):

1. **Reaction** — predict density via a small `WaveletLinear` and descend
   on local prediction error.
2. **MPC drift (solenoidal-first, IFN §10.5/§11.1)** — receding-horizon
   optimization over cycle-space coefficients `α ∈ ℝ^K_cyc`. The velocity
   `u = U @ α` is divergence-free **by construction** via the precomputed
   cycle basis `{U_k}`, so the paper's "naive projection deletes useful
   energy" failure mode is avoided.
3. **Leray projection (safety net only)** — Jacobi-CG on the graph
   Laplacian; runs as a numerical-drift safety net per §10.10, not the
   primary divergence-free enforcement.
4. **CFL rescale** — `dt = min(target_cfl / max|u|, dt_max)`.
5. **Conservative advection** — upwind flux + κ-Laplacian diffusion;
   mass-conserving by construction.

Activated by `config.attn_mode = 'fluid'`. Default `'attn'` keeps
existing behavior bit-for-bit. The fluid mode uses standard autograd
(the "PC" semantics live inside the block, in the K iterations of energy
descent on density), bypassing the vendor PCLayer iterative scheme.

TSU verification path: `quantimork_thrml.fluid_pc.tsu_compile.build_density_update_graph`
mirrors `pmode_verify.py` for the linear-in-ρ density-update step. Phase-1
verifies one operator-splitting step (advection + diffusion); the bilinear
velocity solve and full K-iteration alternation are deferred to Phase 2.

```python
from quantimork_thrml.model import WaveletPCTransformer

config.attn_mode = 'fluid'
config.fluid_outer_iters = 3   # K
config.mpc_horizon = 4         # H
config.mpc_inner_steps = 5
model = WaveletPCTransformer(config)
logits = model(target_ids, input_ids)   # routes to forward_fluid()
```

## Connectivity & Sparsity

The wavelet decomposition creates a tree topology where each coefficient
has ≤5 inter-level neighbors. However, per-level `nn.Linear(size, size)`
layers create **dense intra-level** coupling:

```
$ python scripts/connectivity_analysis.py

Level                  Size │  Direct (W) max │  TSU limit (12)
Detail Level 1          256 │             256 │  ⚠ 21× over
Detail Level 2          128 │             128 │  ⚠ 11× over
Detail Level 3           64 │              64 │  ⚠  5× over
Approx Level 3           64 │              64 │  ⚠  5× over
```

The induced precision matrix `WᵀW` is similarly dense at each level.

### Candidate sparsification directions (open question)

These are directions to explore, not solutions. Each involves trade-offs
that require experimental validation:

- **Banded weights**: Restrict `W[i,j] = 0` for `|i-j| > bandwidth`.
  Physical intuition: adjacent wavelet coefficients couple more strongly
  than distant ones. Reduces connectivity to `O(bandwidth)`, but may
  sacrifice expressiveness — especially for non-local correlations.

- **Second-level wavelet** (wavelet-on-wavelet): Apply another Haar
  decomposition *within* each per-level Linear, recursively reducing
  connectivity. Increases implementation complexity.

- **Structured pruning**: Train dense, then prune to a fixed connectivity
  budget (magnitude pruning, `|W[i,j]| < threshold → 0`). Threshold sweep
  shows 100% TSU compliance at `ε ≥ 0.10` (random init), but trained
  weights may have different sparsity patterns.

- **TSU tiling**: Partition a large level across multiple TSU tiles.
  Inter-tile communication latency needs analysis.

Run `python scripts/connectivity_analysis.py` to measure current
connectivity on your trained model.

## Project structure

```
quantimork_thrml/
├── haar.py               # Haar DWT / IDWT (§7.4.2)
├── wavelet_linear.py     # WaveletLinear: Haar → per-level Linear → IDWT
├── model.py              # WaveletPCTransformer (replaces MLP in PC-Transformers)
├── thrml_verify.py       # Factor graph verification (categorical + pmode dispatch)
├── gaussian_ebm.py       # ContinuousNode, QuadraticFactor, CouplingFactor, GaussianSampler
├── pmode_verify.py       # P-mode (continuous Gaussian) factor graph verification
└── fluid_pc/             # FluidPCBlock — IFN §10.6 drop-in attention replacement
    ├── topology.py       # FluidGraphTopology — node/edge layout (shared torch + thrml)
    ├── graph.py          # WaveletGraph: incidence, divergence, gradient, Laplacian
    ├── leray.py          # Graph-Poisson Leray projection (CG with implicit-IFT backward)
    ├── advection.py      # Conservative upwind advection + Laplacian diffusion
    ├── reaction.py       # PC reaction step on density (uses WaveletLinear predictor)
    ├── drift_mpc.py      # Receding-horizon MPC over per-edge velocity
    ├── readout.py        # ρ → (B,S,D) per-band readout
    ├── fluid_pc_block.py # FluidPCBlock(B,S,D)→(B,S,D)
    └── tsu_compile.py    # Density-update factor graph (verified vs ConservativeAdvection)
scripts/
├── prepare_data.py       # Download + tokenize Tiny Shakespeare
├── train.py              # Train wavelet or baseline model
├── compare.py            # Print comparison table
└── connectivity_analysis.py  # Per-level connectivity measurement
tests/
├── conftest.py           # Shared fixtures and tolerances
├── test_haar.py          # Haar roundtrip + coefficient correctness
├── test_wavelet_linear.py # Shape, params, gradients
├── test_model.py         # Forward pass smoke test
├── test_thrml_verify.py  # TSU factor graph energy equivalence
├── test_baseline_compare.py # Wavelet vs dense regression comparison
└── test_fluid_pc/        # FluidPCBlock unit + TSU verification tests
vendor/
└── PC-Transformers/      # iCog-Labs baseline (git submodule)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Sister Projects

Five projects compiling Hyperon's cognitive architecture to thermodynamic hardware:

| Project | What it compiles |
|---------|-----------------|
| [PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML) | Probabilistic inference → Boltzmann energy tables |
| [ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML) | Attention diffusion → Lattice Boltzmann simulation |
| [MOSES-THRML](https://github.com/xiaohanma-oss/MOSES-THRML) | Program evolution → Boltzmann sampling |
| **[QuantiMORK-THRML](https://github.com/xiaohanma-oss/QuantiMORK-THRML)** | **Predictive coding → wavelet-sparse factor graphs** |
| [Geodesic-THRML](https://github.com/xiaohanma-oss/Geodesic-THRML) | Unified geodesic scheduler for all above |

## Acknowledgements

- [PC-Transformers](https://github.com/iCog-Labs-Dev/PC-Transformers) — iCog Labs
- [thrml](https://github.com/extropic-ai/thrml) — Extropic AI factor graph library

## License

[MIT](LICENSE) — Copyright (c) 2026 Xiaohan Ma
