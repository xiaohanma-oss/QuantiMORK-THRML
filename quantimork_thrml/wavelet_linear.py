"""
WaveletLinear — Haar multi-resolution linear layer for predictive coding.

Implements whitepaper §7.4.2 + §7.4.3: replace dense nn.Linear with
Haar DWT → per-level independent small Linear → Inverse Haar.

Each level's Linear only connects coefficients within that level,
giving tree-local inter-level connectivity (max ~5 neighbors per node).
However, per-level Linear layers are dense within each level
(up to 256 intra-level connections at level 1), exceeding TSU's
~12 neighbor limit. See scripts/connectivity_analysis.py for details.

    Dense nn.Linear(512, 512):  262,144 params, 512 connections/node
    WaveletLinear(512, 512, 3): ~85K params, ≤5 inter-level + dense intra-level
"""

import torch
import torch.nn as nn

from quantimork_thrml.haar import haar_dwt_1d, haar_idwt_1d, tree_positions


class WaveletLinear(nn.Module):
    """Multi-resolution linear layer using Haar wavelet decomposition.

    Forward pass:
        1. Haar DWT on input features → per-level coefficients
        2. Independent nn.Linear at each level (small, local)
        3. Inverse Haar DWT → output features

    Args:
        in_features: Input feature dimension (must be divisible by 2^n_levels).
        out_features: Output feature dimension. Must equal in_features
                      (wavelet structure preserves dimensionality).
        n_levels: Number of Haar decomposition levels.
        bias: Whether to include bias in per-level Linears.
    """

    def __init__(self, in_features, out_features, n_levels=3, bias=True):
        super().__init__()
        if in_features != out_features:
            raise ValueError(
                f"WaveletLinear requires in_features == out_features, "
                f"got {in_features} != {out_features}")
        if in_features % (2 ** n_levels) != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by "
                f"2^{n_levels}={2**n_levels}")

        self.in_features = in_features
        self.out_features = out_features
        self.n_levels = n_levels

        # Per-level transforms: detail bands at each level + coarsest approx
        self.detail_transforms = nn.ModuleList()
        size = in_features
        for level in range(n_levels):
            size = size // 2
            self.detail_transforms.append(
                nn.Linear(size, size, bias=bias))

        # Coarsest approximation transform
        self.approx_transform = nn.Linear(size, size, bias=bias)

    def forward(self, x):
        """Forward pass through wavelet decomposition.

        Args:
            x: Tensor of shape (B, S, D) or (B, D).

        Returns:
            Tensor of same shape as input.
        """
        # 1. Haar DWT
        coeffs = haar_dwt_1d(x, self.n_levels)

        # 2. Per-level independent linear transforms
        transformed_details = []
        for i, detail in enumerate(coeffs["details"]):
            transformed_details.append(self.detail_transforms[i](detail))

        transformed_approx = self.approx_transform(coeffs["approx"])

        # 3. Inverse Haar DWT
        result_coeffs = {
            "details": transformed_details,
            "approx": transformed_approx,
        }
        return haar_idwt_1d(result_coeffs)

    def num_params(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def max_connections_per_node(self):
        """Maximum number of connections per coefficient node.

        In the wavelet tree, each coefficient connects to:
        - Its transform weights within the same level (local)
        - Parent/children via Haar reconstruction (implicit, not learned)

        The largest per-level Linear determines max connectivity.
        """
        max_size = 0
        for t in self.detail_transforms:
            max_size = max(max_size, t.in_features)
        max_size = max(max_size, self.approx_transform.in_features)
        return max_size

    def extract_energy_params(self):
        """Export per-level weights for thrml factor graph construction.

        Returns:
            List of dicts, each with:
                "level": int
                "band": "detail" or "approx"
                "weight": Tensor (size, size)
                "bias": Tensor (size,) or None
        """
        params = []
        positions = tree_positions(self.n_levels, self.in_features)

        for i, pos in enumerate(positions):
            if pos["band"] == "detail":
                layer = self.detail_transforms[i]
            else:
                layer = self.approx_transform
            params.append({
                "level": pos["level"],
                "band": pos["band"],
                "weight": layer.weight.detach().clone(),
                "bias": layer.bias.detach().clone()
                       if layer.bias is not None else None,
            })
        return params
