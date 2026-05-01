"""
Cross-scale PC predictor (IFN §10.4).

Paper line 1696: *"Each node predicts its activation from local neighborhood
**and cross-scale parents/children**"*. The Phase-1 `WaveletLinear`-based
predictor was per-level only — no parent/child terms. This module emits

    â_i = Θ_lat * Σ_{j ∈ lateral_neighbors(i)} a_j
        + Θ_up  * Σ_{p ∈ parents(i)}            a_p
        + Θ_down * Σ_{c ∈ children(i)}          a_c
        + b

with `Θ_lat`, `Θ_up`, `Θ_down`, `b` as four scalar learnable parameters.
Sparse parameter sharing across the whole graph keeps the predictor cheap
yet introduces the cross-scale coupling the paper mandates.

Implementation uses the existing edge partition: edges in
`[0, n_lateral)` are intra-scale lateral; edges in
`[n_lateral, n_lateral + n_cross)` are cross-scale (src=finer, dst=coarser
per build_topology). Lateral contribution is symmetric (sum over both
endpoints); for cross edges the src endpoint contributes to its parent
(dst gets `Θ_down·a_src`) while the dst endpoint contributes to its
children (src gets `Θ_up·a_dst`).
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.graph import WaveletGraph


class CrossScalePredictor(nn.Module):
    def __init__(self, graph: WaveletGraph):
        super().__init__()
        self.graph = graph
        self.theta_lat = nn.Parameter(torch.tensor(0.5))
        self.theta_up = nn.Parameter(torch.tensor(0.25))
        self.theta_down = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.zeros(1))
        self._n_lat = graph.topology.n_lateral

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Predict â_i for each node from neighborhood + parents + children.

        Args:
            rho: (B, |V|)
        Returns:
            (B, |V|) prediction
        """
        g = self.graph
        n_v = g.n_nodes
        src = g.edge_src
        dst = g.edge_dst
        n_lat = self._n_lat

        a_hat = torch.zeros_like(rho)

        if n_lat > 0:
            lat_src = src[:n_lat]
            lat_dst = dst[:n_lat]
            a_hat.scatter_add_(
                -1, lat_dst.expand(*rho.shape[:-1], n_lat),
                self.theta_lat * rho.index_select(-1, lat_src))
            a_hat.scatter_add_(
                -1, lat_src.expand(*rho.shape[:-1], n_lat),
                self.theta_lat * rho.index_select(-1, lat_dst))

        n_cross = src.numel() - n_lat
        if n_cross > 0:
            cross_src = src[n_lat:]
            cross_dst = dst[n_lat:]
            a_hat.scatter_add_(
                -1, cross_dst.expand(*rho.shape[:-1], n_cross),
                self.theta_down * rho.index_select(-1, cross_src))
            a_hat.scatter_add_(
                -1, cross_src.expand(*rho.shape[:-1], n_cross),
                self.theta_up * rho.index_select(-1, cross_dst))

        return a_hat + self.bias
