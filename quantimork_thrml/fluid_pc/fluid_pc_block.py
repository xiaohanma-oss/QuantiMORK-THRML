"""
FluidPCBlock — IFN §10.6 drop-in attention replacement on (B, S, D).

Forward pipeline (per outer iteration k=1..K):
    1. ρ ← PCReaction(ρ)                        # cross-scale PC (§10.4)
    2. u ← MPCDrift(ρ, v_field)                 # solenoidal-first (§10.5)
    3. u ← Leray(u)                             # safety net only (§10.3)
    4. dt ← rescale to CFL ∈ [0.3, 0.45]         # §10.4 (iii)
    5. ρ ← ConservativeAdvection(ρ, u, dt, κ(k)) # κ-annealed (§10.7)

Then `WaveletReadout(ρ_K)` returns (B, S, D).

Per IFN §10.5, MPC's terminal target should be a learned multi-scale value
field W_θ rather than the trivial self-stabilization `ρ.detach()`. We
implement W_θ as a tiny `nn.Linear(D, 1)` head that produces a per-token
scalar broadcast across bands, then DWT-packed. Per §10.7, κ is annealed
κ_init → 0 over outer iters via a cosine schedule. Diagnostics (band mass,
expected distance, divergence norm, achieved CFL) are recorded into
`self.last_diagnostics: list[dict]` for training-loop logging.

This block uses standard torch autograd. The "PC" semantics are *internal*
(energy descent on density via reaction+advection); weights train end-to-
end through `loss.backward()` rather than through vendor PCLayer's Hebbian
path.
"""

import math
from dataclasses import dataclass

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
    kappa_anneal: bool = True
    leray_tol: float = 1e-4
    leray_max_iter: int = 50
    reaction_steps: int = 1
    reaction_eta: float = 0.1


def _config_from(config) -> FluidPCConfig:
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
        kappa_anneal=getattr(config, "kappa_anneal", True),
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
        # Learned multi-scale value field W_θ per IFN §10.5.
        self.value_head = nn.Linear(cfg.D, 1, bias=False)

        self.last_diagnostics: list = []
        self.last_free_energy: dict = {}

    @staticmethod
    def _normalize_rho(rho: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rho = rho.abs() + eps
        return rho / rho.sum(dim=-1, keepdim=True)

    def _kappa_at(self, k: int) -> float:
        """Cosine anneal κ(k) from κ_init at k=0 to 0 at k=K-1 (§10.7)."""
        cfg = self.cfg
        if not cfg.kappa_anneal or cfg.fluid_outer_iters <= 1:
            return cfg.kappa_init
        return 0.5 * cfg.kappa_init * (
            1.0 + math.cos(math.pi * k / (cfg.fluid_outer_iters - 1)))

    def _value_field(self, x: torch.Tensor) -> torch.Tensor:
        """Build W_θ value field on graph V from input x: (B, S, D) → (B, |V|).

        A single Linear(D, 1) head produces a scalar per (B, s); broadcast
        across feature dim then DWT-packed gives a value field on every
        graph node. value_head receives gradient via this path → readout.
        """
        cfg = self.cfg
        per_token = self.value_head(x).squeeze(-1)
        per_token_signal = per_token.unsqueeze(-1).expand(-1, -1, cfg.D)
        v_coeffs = haar_dwt_1d(per_token_signal, cfg.n_levels)
        v = self.graph.coeffs_to_rho(v_coeffs)
        return self._normalize_rho(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        coeffs = haar_dwt_1d(x, cfg.n_levels)
        rho = self.graph.coeffs_to_rho(coeffs)
        rho = self._normalize_rho(rho)
        v_field = self._value_field(x)

        diagnostics = []
        reaction_energies = []
        v_field_anchor_loss = None
        for k in range(cfg.fluid_outer_iters):
            rho = self.reaction(rho)
            if self.reaction.last_reaction_energy is not None:
                reaction_energies.append(self.reaction.last_reaction_energy)
            u = self.mpc(rho, v_field)
            u = self.leray(u)
            dt = self.advection.rescale_dt(u, target_cfl=cfg.cfl_target)
            kappa_k = self._kappa_at(k)
            rho = self.advection(rho, u, dt, kappa_override=kappa_k)
            rho = self._normalize_rho(rho)

            # Four diagnostic scalars per IFN §10.7.
            with torch.no_grad():
                top_band = rho.argmax(dim=-1, keepdim=True)
                band_mass = rho.gather(-1, top_band).mean().item()
                exp_dist = ((1.0 - v_field) * rho).sum(dim=-1).mean().item()
                div_norm = self.graph.divergence(u).abs().max().item()
                if dt.dim() == 0:
                    cfl_dt = dt
                else:
                    cfl_dt = dt.view(*dt.shape, 1)
                achieved_cfl = (cfl_dt * u.abs()).amax().item()
            diagnostics.append({
                "k": k, "kappa": kappa_k, "band_mass": band_mass,
                "exp_distance": exp_dist, "div_norm": div_norm,
                "achieved_cfl": achieved_cfl,
            })

        self.last_diagnostics = diagnostics
        # §10.7 free-energy training term — sum of reaction prediction
        # errors across outer iters, plus a value-anchor term that ties the
        # learned W_θ to the converged density (gives value_head gradient
        # via a meaningful supervision signal, not just the readout-mix
        # workaround).
        react_total = (sum(reaction_energies)
                       if reaction_energies
                       else torch.zeros((), device=rho.device))
        v_anchor = ((v_field - rho.detach()) ** 2).sum(dim=-1).mean()
        self.last_free_energy = {
            "reaction": react_total,
            "v_anchor": v_anchor,
        }
        # Mix v_field into the final readout so value_head receives gradient
        # via the main forward path as well.
        rho_final = rho + 0.1 * v_field
        rho_final = rho_final / rho_final.sum(dim=-1, keepdim=True)
        return self.readout(rho_final)
