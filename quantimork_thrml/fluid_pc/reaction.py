"""
PC reaction step on the wavelet density.

Predictor: a `WaveletLinear` operating on the (B, S, D) coefficient stack
(after `rho_to_coeffs`). The prediction error e = a - W(a) drives an
energy-descent update on a:
    a <- a - eta * e
which is the Euler step on F = 0.5 * ||a - W(a)||^2.

The predictor instance here is *separate* from any predictor in `WaveletMLP`
— same class, different parameters.
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.haar import haar_idwt_1d
from quantimork_thrml.wavelet_linear import WaveletLinear


class PCReaction(nn.Module):
    def __init__(self, graph: WaveletGraph, n_levels: int, D: int,
                 reaction_steps: int = 1, eta: float = 0.1):
        super().__init__()
        self.graph = graph
        self.n_levels = n_levels
        self.D = D
        self.reaction_steps = reaction_steps
        self.eta = eta
        self.predictor = WaveletLinear(D, D, n_levels=n_levels)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """One or more PC error-reduction steps on the wavelet density.

        Args:
            rho: (B, |V|) density-like tensor (need not be a strict simplex).
        Returns:
            (B, |V|) updated density.
        """
        for _ in range(self.reaction_steps):
            coeffs = self.graph.rho_to_coeffs(rho)
            coeffs_signal = haar_idwt_1d(coeffs)
            predicted_signal = self.predictor(coeffs_signal)
            from quantimork_thrml.haar import haar_dwt_1d
            predicted_coeffs = haar_dwt_1d(predicted_signal, self.n_levels)
            predicted_rho = self.graph.coeffs_to_rho(predicted_coeffs)
            err = rho - predicted_rho
            rho = rho - self.eta * err
        return rho
