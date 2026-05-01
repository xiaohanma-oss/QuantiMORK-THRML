"""
FluidPCBlock — IFN §10.6 drop-in attention replacement on (B, S, D).

Forward pipeline (per outer iteration k=1..K):
    1. ρ ← PCReaction(ρ)                  # local prediction-error descent
    2. u ← MPCDrift(ρ, value_field)       # H-step receding-horizon velocity
    3. u ← Leray(u)                       # safety-net divergence projection
    4. dt ← rescale to CFL ∈ [0.3, 0.45]
    5. ρ ← ConservativeAdvection(ρ, u, dt)

Then `WaveletReadout(ρ_K)` returns (B, S, D).

Initial ρ is built from `haar_dwt_1d(x)` packed via `WaveletGraph.coeffs_to_rho`
followed by abs+L1 normalization (so rho is non-negative and sum-to-one per
batch element). The block is meant to be wrapped in a residual: callers do
`x = x + FluidPCBlock(ln(x))`.

This block uses standard torch autograd. The "PC" semantics are *internal*
(energy descent on density via reaction+advection); weights train end-to-end
through `loss.backward()` rather than through vendor PCLayer's Hebbian path.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.advection import ConservativeAdvection
from quantimork_thrml.fluid_pc.drift_mpc import MPCDrift
from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector
from quantimork_thrml.fluid_pc.reaction import PCReaction
from quantimork_thrml.fluid_pc.readout import WaveletReadout
from quantimork_thrml.haar import haar_dwt_1d


@dataclass
class FluidPCConfig:
    S: int
    D: int
    n_levels: int = 3
    fluid_outer_iters: int = 3
    mpc_horizon: int = 4
    mpc_inner_steps: int = 5
    mpc_inner_lr: float = 0.1
    mpc_differentiable: bool = False
    cfl_target: float = 0.4
    kappa_init: float = 0.01
    leray_tol: float = 1e-4
    leray_max_iter: int = 50
    reaction_steps: int = 1
    reaction_eta: float = 0.1


def _config_from(config) -> FluidPCConfig:
    """Build FluidPCConfig from a generic config that exposes `n_embed`,
    `block_size`, and optionally fluid_* fields.
    """
    return FluidPCConfig(
        S=getattr(config, "block_size"),
        D=getattr(config, "n_embed"),
        n_levels=getattr(config, "wavelet_n_levels", 3),
        fluid_outer_iters=getattr(config, "fluid_outer_iters", 3),
        mpc_horizon=getattr(config, "mpc_horizon", 4),
        mpc_inner_steps=getattr(config, "mpc_inner_steps", 5),
        mpc_inner_lr=getattr(config, "mpc_inner_lr", 0.1),
        mpc_differentiable=getattr(config, "mpc_differentiable", False),
        cfl_target=getattr(config, "cfl_target", 0.4),
        kappa_init=getattr(config, "kappa_init", 0.01),
        leray_tol=getattr(config, "leray_tol", 1e-4),
        leray_max_iter=getattr(config, "leray_max_iter", 50),
        reaction_steps=getattr(config, "reaction_steps", 1),
        reaction_eta=getattr(config, "reaction_eta", 0.1),
    )


class FluidPCBlock(nn.Module):
    """Drop-in attention replacement: (B, S, D) → (B, S, D)."""

    def __init__(self, config):
        super().__init__()
        cfg = _config_from(config)
        self.cfg = cfg

        self.graph = WaveletGraph(cfg.S, cfg.D, cfg.n_levels)
        self.leray = LerayProjector(self.graph,
                                    tol=cfg.leray_tol,
                                    max_iter=cfg.leray_max_iter)
        self.advection = ConservativeAdvection(
            self.graph, kappa_init=cfg.kappa_init)
        self.reaction = PCReaction(
            self.graph, cfg.n_levels, cfg.D,
            reaction_steps=cfg.reaction_steps,
            eta=cfg.reaction_eta)
        self.mpc = MPCDrift(
            self.graph, self.leray, self.advection,
            horizon=cfg.mpc_horizon,
            inner_steps=cfg.mpc_inner_steps,
            inner_lr=cfg.mpc_inner_lr,
            mpc_differentiable=cfg.mpc_differentiable,
            cfl_target=cfg.cfl_target)
        self.readout = WaveletReadout(self.graph, cfg.n_levels, cfg.D)

    @staticmethod
    def _normalize_rho(rho: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rho = rho.abs() + eps
        return rho / rho.sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        coeffs = haar_dwt_1d(x, cfg.n_levels)
        rho = self.graph.coeffs_to_rho(coeffs)
        rho = self._normalize_rho(rho)
        # MPC terminal target: the initial density, frozen. The drift then
        # tries to use advection to counteract diffusion-induced smearing.
        v_field = rho.detach()

        for _ in range(cfg.fluid_outer_iters):
            rho = self.reaction(rho)
            u = self.mpc(rho, v_field)
            u = self.leray(u)
            dt = self.advection.rescale_dt(u, target_cfl=cfg.cfl_target)
            rho = self.advection(rho, u, dt)
            rho = self._normalize_rho(rho)

        return self.readout(rho)
