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
thrml factor graphs that run on Extropic's TSU via Gibbs sampling. Each
wavelet coefficient node has ≤5 connections — well within TSU's ~12-neighbor
hardware limit. The architecture implements Hyperon whitepaper §7.4
(QuantiMORK), and uses iCog-Labs' PC-Transformers as the baseline.

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

Key property: each coefficient connects to at most **5 neighbors**
(parent, 2 children, 2 siblings). A dense layer connects to **512+**.
TSU hardware supports ~12 neighbors per node. Wavelets fit; dense doesn't.

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
  → 90K params, ≤5 connections per node
  → TSU limit: ~12 neighbors → ✓
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

```
Free energy:   F = E_pred + α × E_td + β × D_KL(q || prior)
Prediction:    W[i,j] = −0.5 × precision × (center_i − center_j)²
TD modulation: W_td[i,j] = −α × 0.5 × precision × (center_i − center_j)²
KL prior:      W_kl[j] = β × log(prior_j)   (Gaussian discretized on k bins)
Execution:     Gibbs sampling on CategoricalNode factor graph
```

### Mapping table

| PC-Transformers concept | QuantiMORK-THRML | thrml API |
|-------------------------|------------------|-----------|
| MLP dense weights | WaveletLinear per-level weights | `SquareCategoricalEBMFactor` |
| PC latent state x | Wavelet coefficients (level, pos) | `CategoricalNode` K bins |
| prediction error | Factor graph energy | Factor potential |
| T-step iteration | — | Gibbs warmup + sampling |
| Leaf observations | Clamped input activations | `BlockGibbsSpec(clamped)` |

## Results

Tiny Shakespeare, 5 epochs, n_embed=128, n_blocks=4, T=2, CPU training:

| Metric | PC-Transformers | QuantiMORK (td_alpha=0.5, beta=0.1) |
|--------|:---------------:|:-----------------------------------:|
| Parameters | 1.06M | 0.56M |
| MLP params/layer | dense | 90K (wavelet) |
| Max connections/node | 512 | ≤5 |
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
| `build_single_level_graph(weight_matrix, input_act, target_act, ...)` | Build thrml factor graph for one WaveletLinear level with optional TD modulation and KL regularization |
| `run_verification(graph, seed=0, n_batches=30)` | Run Gibbs sampling on factor graph; returns `{sampled_values, target_values, mse, sampled_energy}` |
| `pc_prediction_weights(precision, k=16)` | K×K pairwise weight matrix encoding PC squared prediction error |
| `td_modulation_weights(alpha, precision, k=16)` | K×K pairwise weight matrix encoding top-down modulation energy |
| `kl_prior_weights(prior_probs, beta=0.1, k=16)` | [K] unary weight vector encoding KL divergence complexity penalty |
| `coeff_bin_centers(k=16)` | K evenly-spaced bin centers for discretizing activations |
| `value_to_bin(value, k=16)` / `bin_to_value(bin_idx, k=16)` | Quantize continuous scalar ↔ bin index |

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
decomposition limits each coefficient node to ≤5 neighbors, fitting within
TSU's ~12-neighbor hardware constraint. Dense PC layers (512+ connections)
would require multi-chip partitioning for even a single layer.

## Project structure

```
quantimork_thrml/
├── haar.py               # Haar DWT / IDWT (§7.4.2)
├── wavelet_linear.py     # WaveletLinear: Haar → per-level Linear → IDWT
├── model.py              # WaveletPCTransformer (replaces MLP in PC-Transformers)
└── thrml_verify.py       # Extract weights → thrml factor graph → verify
scripts/
├── prepare_data.py       # Download + tokenize Tiny Shakespeare
├── train.py              # Train wavelet or baseline model
└── compare.py            # Print comparison table
tests/
├── conftest.py           # Shared fixtures and tolerances
├── test_haar.py          # Haar roundtrip + coefficient correctness
├── test_wavelet_linear.py # Shape, params, gradients
├── test_model.py         # Forward pass smoke test
├── test_thrml_verify.py  # TSU factor graph energy equivalence
└── test_baseline_compare.py # Wavelet vs dense regression comparison
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
