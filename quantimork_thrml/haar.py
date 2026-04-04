"""
Haar wavelet transform along the feature dimension.

Implements 1D Haar DWT/IDWT operating on the last dimension of tensors,
suitable for decomposing neural network activations into multi-resolution
coefficients as described in Hyperon whitepaper §7.4.2:

    "QuantiMORK encodes tensors using discrete wavelet transforms,
     storing coefficients as Atoms keyed by (level, band, position)."
"""

import torch
from math import sqrt

SQRT2_INV = 1.0 / sqrt(2.0)


def haar_dwt_1d(x, n_levels=3):
    """1D Haar DWT along the last dimension.

    Args:
        x: Tensor of shape (..., D) where D must be divisible by 2^n_levels.
        n_levels: Number of decomposition levels.

    Returns:
        List of (approx, detail) tuples per level, from coarsest to finest.
        Level 0 is the coarsest (smallest). The final approx at level 0
        is the overall average.

        coeffs[i] = detail coefficients at level i  (shape (..., D // 2^(i+1)))
        coeffs is ordered [finest_detail, ..., coarsest_detail]
        plus the final coarsest approximation as the last element.

        Concretely, returns a dict:
            {"details": [d_1, d_2, ..., d_L], "approx": a_L}
        where d_1 has shape (..., D//2), d_2 has (..., D//4), etc.
        and a_L has shape (..., D // 2^L).
    """
    D = x.shape[-1]
    if D % (2 ** n_levels) != 0:
        raise ValueError(
            f"Feature dim {D} not divisible by 2^{n_levels}={2**n_levels}")

    details = []
    current = x
    for _ in range(n_levels):
        even = current[..., 0::2]
        odd = current[..., 1::2]
        approx = (even + odd) * SQRT2_INV
        detail = (even - odd) * SQRT2_INV
        details.append(detail)
        current = approx

    return {"details": details, "approx": current}


def haar_idwt_1d(coeffs):
    """Inverse 1D Haar DWT — reconstruct from coefficients.

    Args:
        coeffs: Dict with "details" (list of detail tensors, finest first)
                and "approx" (coarsest approximation tensor).

    Returns:
        Reconstructed tensor with shape matching the original input.
    """
    details = coeffs["details"]
    current = coeffs["approx"]

    for detail in reversed(details):
        even = (current + detail) * SQRT2_INV
        odd = (current - detail) * SQRT2_INV
        # Interleave even and odd using stack (preserves autograd graph)
        current = torch.stack([even, odd], dim=-1).flatten(start_dim=-2)

    return current


def tree_positions(n_levels, n_features):
    """Enumerate all wavelet coefficient positions as (level, band, size).

    Returns:
        List of dicts with keys: level, band ('approx' or 'detail'), size.
        Ordered from finest detail to coarsest, then coarsest approx.
    """
    positions = []
    size = n_features
    for level in range(1, n_levels + 1):
        size = size // 2
        positions.append({
            "level": level,
            "band": "detail",
            "size": size,
        })
    positions.append({
        "level": n_levels,
        "band": "approx",
        "size": size,
    })
    return positions
