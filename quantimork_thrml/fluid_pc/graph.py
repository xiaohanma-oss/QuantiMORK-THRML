"""
Wavelet graph V for FluidPCBlock. Builds the static (S, D, n_levels) topology
once and exposes it as torch buffers usable in forward.

A node corresponds to a wavelet coefficient at (s, band, b). An edge is
either lateral (same band, adjacent s) or cross-level (Haar parent/child or
detail-L<->approx-L sibling pair). See topology.py for the linear ordering.

Operators (all stored as `torch.sparse_coo_tensor`):
    B^T   : (|E|, |V|) signed incidence (+1 at src, -1 at dst).
    L_lap : (|V|, |V|) graph Laplacian B^T B (SPD up to constant null space).

`coeffs_to_rho` / `rho_to_coeffs` are pure tensor reshape/concat ops, so they
preserve autograd and are exact roundtrip (matches haar_dwt_1d ordering).
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.topology import FluidGraphTopology, build_topology


class WaveletGraph(nn.Module):
    """Static wavelet graph topology cached as buffers."""

    def __init__(self, S: int, D: int, n_levels: int):
        super().__init__()
        topo = build_topology(S, D, n_levels)
        self.S = S
        self.D = D
        self.n_levels = n_levels
        self.topology = topo

        edge_src = torch.tensor(topo.edge_src, dtype=torch.long)
        edge_dst = torch.tensor(topo.edge_dst, dtype=torch.long)
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)

        n_edges = edge_src.numel()
        n_nodes = topo.n_nodes

        # B^T: rows = edges, cols = nodes. +1 at src, -1 at dst.
        e_idx = torch.arange(n_edges, dtype=torch.long)
        bt_indices = torch.stack([
            torch.cat([e_idx, e_idx]),
            torch.cat([edge_src, edge_dst]),
        ])
        bt_values = torch.cat([
            torch.ones(n_edges),
            -torch.ones(n_edges),
        ])
        self._bt_indices = bt_indices
        self._bt_values = bt_values
        # NOTE: torch sparse tensors can't be registered as buffers cleanly;
        # we materialize on-demand via .B_T()/.L_lap()/.B().

        band_sizes = torch.tensor(topo.band_sizes, dtype=torch.long)
        self.register_buffer("band_sizes", band_sizes)

        diag = torch.zeros(n_nodes)
        diag.scatter_add_(0, edge_src, torch.ones(n_edges))
        diag.scatter_add_(0, edge_dst, torch.ones(n_edges))
        self.register_buffer("laplacian_diag", diag)

    @property
    def n_nodes(self) -> int:
        return self.topology.n_nodes

    @property
    def n_edges(self) -> int:
        return self.topology.n_edges

    def _move(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(device=self.laplacian_diag.device,
                    dtype=self.laplacian_diag.dtype)

    def B_T(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._bt_indices.to(self.laplacian_diag.device),
            self._move(self._bt_values),
            size=(self.n_edges, self.n_nodes),
        ).coalesce()

    def divergence(self, u: torch.Tensor) -> torch.Tensor:
        """B^T @ u_signed -> per-node divergence.

        Convention: signed incidence has +1 at src, -1 at dst, so the per-node
        divergence is the sum of u_e at edges where the node is the *src*
        minus the sum at edges where it is the *dst*.

        Args:
            u: (..., |E|) edge field.
        Returns:
            (..., |V|) node field.
        """
        out = torch.zeros(*u.shape[:-1], self.n_nodes,
                          device=u.device, dtype=u.dtype)
        out.scatter_add_(-1, self.edge_src.expand_as(u), u)
        out.scatter_add_(-1, self.edge_dst.expand_as(u), -u)
        return out

    def gradient(self, p: torch.Tensor) -> torch.Tensor:
        """B @ p -> edge differences. (Adjoint of -divergence.)

        For each edge e = (src, dst): grad(p)_e = p[src] - p[dst].
        """
        return p.gather(-1, self.edge_src.expand(*p.shape[:-1], self.n_edges)) \
             - p.gather(-1, self.edge_dst.expand(*p.shape[:-1], self.n_edges))

    def laplacian_apply(self, p: torch.Tensor) -> torch.Tensor:
        """L @ p where L = B^T B, applied without materializing L.

        L p [i] = deg(i) * p[i] - sum_{j ~ i} p[j].
        Computed via grad(p) (per-edge differences) then divergence.
        """
        return self.divergence(self.gradient(p))

    def coeffs_to_rho(self, coeffs: dict) -> torch.Tensor:
        """Pack haar_dwt_1d output into (B, |V|) using the topology ordering.

        Concat detail bands (finest→coarsest) then approx along last dim,
        then reshape (B, S, D) -> (B, S*D). Linear index of (s, band, b) is
        s * D + band_offsets[band] + b.
        """
        bands = list(coeffs["details"]) + [coeffs["approx"]]
        cat = torch.cat(bands, dim=-1)
        return cat.reshape(cat.shape[0], -1)

    def rho_to_coeffs(self, rho: torch.Tensor) -> dict:
        """Inverse of coeffs_to_rho: (B, |V|) -> haar_dwt_1d-shaped dict."""
        cat = rho.reshape(rho.shape[0], self.S, self.D)
        sizes = list(self.topology.band_sizes)
        chunks = torch.split(cat, sizes, dim=-1)
        return {"details": list(chunks[:-1]), "approx": chunks[-1]}
