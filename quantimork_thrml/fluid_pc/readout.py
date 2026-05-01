"""
ρ → (B, S, D) readout. Per-band learned scale gains broadcast over S.
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.haar import haar_idwt_1d


class WaveletReadout(nn.Module):
    def __init__(self, graph: WaveletGraph, n_levels: int, D: int):
        super().__init__()
        self.graph = graph
        self.n_levels = n_levels
        self.D = D
        self.band_gains = nn.Parameter(
            torch.ones(len(graph.topology.band_sizes)))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        coeffs = self.graph.rho_to_coeffs(rho)
        scaled_details = []
        for i, d in enumerate(coeffs["details"]):
            scaled_details.append(d * self.band_gains[i])
        scaled_approx = coeffs["approx"] * self.band_gains[-1]
        return haar_idwt_1d({
            "details": scaled_details,
            "approx": scaled_approx,
        })
