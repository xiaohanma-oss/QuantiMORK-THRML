"""
PC reaction step on the wavelet density (IFN §10.4).

Predictor: `CrossScalePredictor` operating directly on graph nodes. Each
node predicts its activation from intra-scale lateral neighbors plus
cross-scale parents and children, per the paper's mandate (line 1696):
*"Each node predicts its activation from local neighborhood and cross-
scale parents/children"*. The previous `WaveletLinear`-based predictor
was per-level dense and missed the parent/child terms — and also violated
the TSU 12-neighbor connectivity limit. The new predictor uses only the
graph's intrinsic edges, so per-node connectivity ≈ 5.

Energy-descent step: a ← a − η · (a − â).
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.cross_scale_predictor import CrossScalePredictor
from quantimork_thrml.fluid_pc.graph import WaveletGraph


class PCReaction(nn.Module):
    def __init__(self, graph: WaveletGraph, n_levels: int, D: int,
                 reaction_steps: int = 1, eta: float = 0.1):
        super().__init__()
        self.graph = graph
        self.n_levels = n_levels
        self.D = D
        self.reaction_steps = reaction_steps
        self.eta = eta
        self.predictor = CrossScalePredictor(graph)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """One or more PC error-reduction steps on the wavelet density.

        Args:
            rho: (B, |V|) density-like tensor.
        Returns:
            (B, |V|) updated density.
        """
        for _ in range(self.reaction_steps):
            predicted = self.predictor(rho)
            err = rho - predicted
            rho = rho - self.eta * err
        return rho
