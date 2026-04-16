"""
Gaussian EBM components for thrml p-mode factor graphs.

Implements continuous-variable nodes, factors, and Gibbs sampler for
encoding PC quadratic energy on thrml. Based on Extropic's official
Gaussian PGM example (docs.thrml.ai/en/latest/examples/01_all_of_thrml/).

Energy form:
    E_G(x) = 0.5 * (x - μ)^T A (x - μ)
           = 0.5 * Σ_i A_ii x_i² + Σ_{j>i} A_ij x_i x_j + Σ_i b_i x_i

Under Laplace approximation (Active Inference §4.19, Box 4.3), this is
the natural encoding of PC prediction error energy.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block
from thrml.conditional_samplers import (
    AbstractConditionalSampler,
    _SamplerState,
    _State,
)
from thrml.factor import AbstractFactor
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class ContinuousNode(AbstractNode):
    """Continuous random variable node (p-mode on TSU hardware)."""
    pass


# ---------------------------------------------------------------------------
#  Interactions
# ---------------------------------------------------------------------------

class LinearInteraction(eqx.Module):
    """Linear interaction: contributes Σ_j w_j x_j to the energy gradient."""
    weights: Array


class QuadraticInteraction(eqx.Module):
    """Quadratic self-interaction: contributes 0.5 * (1/σ²) * x² to energy.

    Stores inverse_weights = 1 / A_ii (i.e. variance, not precision),
    so the sampler can read off the conditional variance directly.
    """
    inverse_weights: Array


# ---------------------------------------------------------------------------
#  Factors
# ---------------------------------------------------------------------------

class QuadraticFactor(AbstractFactor):
    """Diagonal precision: E_i = 0.5 * A_ii * x_i².

    Args:
        inverse_weights: 1/A_ii per node (conditional variance).
        block: Block of ContinuousNodes this factor acts on.
    """
    inverse_weights: Array

    def __init__(self, inverse_weights: Array, block: Block):
        super().__init__([block])
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [InteractionGroup(
            interaction=QuadraticInteraction(self.inverse_weights),
            head_nodes=self.node_groups[0],
            tail_nodes=[],
        )]


class LinearFactor(AbstractFactor):
    """Bias / mean shift: E_i = b_i * x_i.

    Used for encoding -A @ μ terms after expanding (x - μ)^T A (x - μ).
    """
    weights: Array

    def __init__(self, weights: Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [InteractionGroup(
            interaction=LinearInteraction(self.weights),
            head_nodes=self.node_groups[0],
            tail_nodes=[],
        )]


class CouplingFactor(AbstractFactor):
    """Pairwise scalar coupling: E_{ij} = w * x_i * x_j.

    Each edge (block_a[i], block_b[i]) has coupling weight weights[i].
    Encodes off-diagonal precision matrix elements A_ij.
    """
    weights: Array

    def __init__(self, weights: Array, blocks: tuple[Block, Block]):
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[0],
                [self.node_groups[1]],
            ),
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[1],
                [self.node_groups[0]],
            ),
        ]


# ---------------------------------------------------------------------------
#  Conditional sampler
# ---------------------------------------------------------------------------

class GaussianSampler(AbstractConditionalSampler):
    """Gibbs conditional sampler for continuous Gaussian nodes.

    For node x_i with precision A_ii and neighbors x_j coupled via A_ij:
        x_i | x_{-i} ~ N(μ_cond, σ²_cond)
    where:
        σ²_cond = 1 / A_ii
        μ_cond  = -σ²_cond * Σ_j A_ij x_j  (sum of linear interactions)

    On TSU hardware this is native pmode sampling; in software we use
    JAX random normal (Gibbs is not efficient for pure Gaussian, but
    verifies energy equivalence).
    """

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[Array, _SamplerState]:
        bias = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)
        var = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)

        for active, interaction, state in zip(active_flags, interactions, states):
            if isinstance(interaction, LinearInteraction):
                state_prod = jnp.array(1.0)
                if len(state) > 0:
                    state_prod = jnp.prod(jnp.stack(state, -1), -1)
                bias -= jnp.sum(
                    interaction.weights * active * state_prod,
                    axis=-1,
                )

            if isinstance(interaction, QuadraticInteraction):
                var = active * interaction.inverse_weights
                var = var[..., 0]

        return (
            jnp.sqrt(var) * jax.random.normal(key, output_sd.shape)
            + (bias * var),
            sampler_state,
        )

    def init(self) -> _SamplerState:
        return None
