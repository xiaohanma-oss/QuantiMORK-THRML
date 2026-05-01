"""
Conservative upwind advection + Laplacian diffusion on the wavelet graph.

Given density rho on nodes and velocity u on edges, the discrete continuity
equation is
    rho_new = rho - dt * div(phi)  -  dt * kappa * L rho
where
    phi_e = u_e * rho[src(e)] if u_e > 0 else u_e * rho[dst(e)]    (upwind)
    div(phi)[i] = sum_{e: src(e)=i} phi_e - sum_{e: dst(e)=i} phi_e

Mass conservation: div has zero column-sum on the graph (every edge
contributes phi_e to its src and -phi_e to its dst, which cancel in the
total sum), so sum_i rho_new[i] = sum_i rho[i] up to fp32. The Laplacian
diffusion term is also mass-preserving for the same reason.

CFL helper rescales dt so that the maximum |u_e| * dt lands in [0.3, 0.45],
matching the IFN paper's recommended stability target.
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.graph import WaveletGraph


class ConservativeAdvection(nn.Module):
    def __init__(self, graph: WaveletGraph, kappa_init: float = 0.01):
        super().__init__()
        self.graph = graph
        self.register_buffer("kappa",
                             torch.tensor(float(kappa_init)))

    @staticmethod
    def rescale_dt(u: torch.Tensor, target_cfl: float = 0.4,
                   eps: float = 1e-3, dt_max: float = 1.0) -> torch.Tensor:
        """Per-batch dt with two caps:
            1. Stability: dt = target_cfl / max(|u|, eps).  When |u| is very
               small (no flow yet), eps prevents dt from exploding.
            2. Diffusion safety: cap at `dt_max` so the kappa-Laplacian term
               stays in the linear-stability regime.
        """
        u_max = u.abs().amax(dim=-1).clamp_min(eps)
        return (target_cfl / u_max).clamp_max(dt_max)

    def forward(self, rho: torch.Tensor, u: torch.Tensor,
                dt: torch.Tensor) -> torch.Tensor:
        """rho: (B, |V|), u: (B, |E|), dt: scalar or (B,) tensor.

        Returns rho_new with the same shape as rho.
        """
        graph = self.graph
        rho_src = rho.gather(
            -1, graph.edge_src.expand(*rho.shape[:-1], graph.n_edges))
        rho_dst = rho.gather(
            -1, graph.edge_dst.expand(*rho.shape[:-1], graph.n_edges))
        upwind_rho = torch.where(u >= 0, rho_src, rho_dst)
        phi = u * upwind_rho
        div_phi = graph.divergence(phi)

        if dt.dim() == 0:
            dt_b = dt
        else:
            dt_b = dt.view(*dt.shape, 1)

        diffusion = graph.laplacian_apply(rho)
        return rho - dt_b * div_phi - dt_b * self.kappa * diffusion
