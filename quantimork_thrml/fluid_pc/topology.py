"""
Topology shared between torch-side `WaveletGraph` and TSU-side factor builders.

A `FluidGraphTopology` is a frozen description of the wavelet graph V and its
edges E that does not depend on torch device or dtype. Building it once means
the same node/edge layout drives both the torch forward pass and the thrml
factor graph compilation.

Conventions:
    Linear node index:  idx(s, band, b) = s * D + band_offset[band] + b
    Bands ordered:      [detail_1, detail_2, ..., detail_L, approx_L]
    Band sizes:         [D/2, D/4, ..., D/2^L, D/2^L]   (sum = D)
    |V| = S * D
    Edges, in order:    lateral first (along S, within each (band, b))
                        then cross-level (Haar tree, parent-child)
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class FluidGraphTopology:
    S: int
    D: int
    n_levels: int
    band_sizes: Tuple[int, ...]
    band_offsets: Tuple[int, ...]
    n_nodes: int
    edge_src: Tuple[int, ...]
    edge_dst: Tuple[int, ...]
    n_lateral: int
    n_cross: int

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    def node_index(self, s: int, band: int, b: int) -> int:
        return s * self.D + self.band_offsets[band] + b


def build_topology(S: int, D: int, n_levels: int) -> FluidGraphTopology:
    """Compute node/edge layout for a wavelet graph on (S, D, n_levels).

    Edges:
      Lateral (path along S):  (s, band, b) <-> (s+1, band, b),
                               for s in [0..S-2] and every (band, b).
      Cross-level (Haar tree):
          For ell in [1..L-1]:  detail_ell[b] <-> detail_{ell+1}[b // 2]
          For ell = L:          detail_L[b] <-> approx_L[b]   (sibling pair)
    """
    if D % (2 ** n_levels) != 0:
        raise ValueError(
            f"D={D} not divisible by 2^{n_levels}={2 ** n_levels}")

    band_sizes: List[int] = [D // (2 ** ell) for ell in range(1, n_levels + 1)]
    band_sizes.append(D // (2 ** n_levels))
    band_offsets: List[int] = [0]
    for sz in band_sizes[:-1]:
        band_offsets.append(band_offsets[-1] + sz)
    assert sum(band_sizes) == D, (sum(band_sizes), D)

    n_bands = len(band_sizes)
    detail_band_idx = list(range(n_levels))
    approx_band_idx = n_levels

    n_nodes = S * D

    edge_src: List[int] = []
    edge_dst: List[int] = []

    def idx(s: int, band: int, b: int) -> int:
        return s * D + band_offsets[band] + b

    for s in range(S - 1):
        for band, sz in enumerate(band_sizes):
            for b in range(sz):
                edge_src.append(idx(s, band, b))
                edge_dst.append(idx(s + 1, band, b))
    n_lateral = len(edge_src)

    for s in range(S):
        for ell in range(n_levels - 1):
            child_band = detail_band_idx[ell]
            parent_band = detail_band_idx[ell + 1]
            for b in range(band_sizes[child_band]):
                edge_src.append(idx(s, child_band, b))
                edge_dst.append(idx(s, parent_band, b // 2))
        last_detail = detail_band_idx[-1]
        for b in range(band_sizes[last_detail]):
            edge_src.append(idx(s, last_detail, b))
            edge_dst.append(idx(s, approx_band_idx, b))
    n_cross = len(edge_src) - n_lateral

    return FluidGraphTopology(
        S=S, D=D, n_levels=n_levels,
        band_sizes=tuple(band_sizes),
        band_offsets=tuple(band_offsets),
        n_nodes=n_nodes,
        edge_src=tuple(edge_src),
        edge_dst=tuple(edge_dst),
        n_lateral=n_lateral,
        n_cross=n_cross,
    )
