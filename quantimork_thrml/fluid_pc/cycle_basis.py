"""
Cycle-space basis {U_k} for the wavelet graph, per IFN §10.10.

Each fundamental cycle of the graph yields one column of `U`. By
construction every column lies in the kernel of the divergence operator
B^T, so any `u = U @ alpha` is divergence-free without need for explicit
projection. This is the "solenoidal parameterization first" path that
§11.1 of IFN flags as critical: *"Naively projecting a gradient field can
delete almost all useful energy."*

Algorithm
=========
1. BFS spanning tree from node 0 over the (undirected) graph induced by
   edges; record `parent[v] = (parent_node, edge_idx, sign)` where `sign`
   is the contribution of that tree edge when traversed *child → parent*.
2. For each non-tree edge e = (s, d) (with src=s, dst=d):
       Fundamental cycle = e (s→d, sign=+1)
                         + tree_path(d → LCA, walking up)
                         + reverse(tree_path(s → LCA))
   Reverse traversal flips the sign.
3. Stack columns into a `torch.sparse_coo_tensor` of shape (|E|, K_cyc)
   where K_cyc = |E| − |V| + 1 for a connected graph.

Cycle vectors have only ±1 entries (no scaling), so storage is
O(|E| · avg_cycle_len) rather than O(|E| · K_cyc).
"""

from collections import deque
from typing import Tuple

import torch

from quantimork_thrml.fluid_pc.topology import FluidGraphTopology


def build_cycle_basis(
    topology: FluidGraphTopology,
) -> Tuple[torch.Tensor, int]:
    """Return (U_sparse, n_cycles).

    U_sparse: torch.sparse_coo_tensor of shape (|E|, K_cyc) with ±1 entries.
    n_cycles: K_cyc = number of fundamental cycles.
    """
    n_v = topology.n_nodes
    n_e = len(topology.edge_src)
    src = topology.edge_src
    dst = topology.edge_dst

    adj = [[] for _ in range(n_v)]
    for e in range(n_e):
        s, d = src[e], dst[e]
        adj[s].append((d, e, +1))
        adj[d].append((s, e, -1))

    parent = [(-1, -1, 0)] * n_v
    visited = [False] * n_v
    tree_edges = set()
    component_root = [-1] * n_v
    for start in range(n_v):
        if visited[start]:
            continue
        visited[start] = True
        component_root[start] = start
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v, e, sign_u_to_v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = (u, e, -sign_u_to_v)
                    component_root[v] = start
                    tree_edges.add(e)
                    queue.append(v)

    rows = []
    cols = []
    vals = []
    cycle_idx = 0
    for e in range(n_e):
        if e in tree_edges:
            continue
        s, d = src[e], dst[e]
        if component_root[s] != component_root[d]:
            # Non-tree bridging edge — shouldn't happen for an undirected
            # BFS forest where every edge has both endpoints visited; skip
            # defensively.
            continue
        rows.append(e); cols.append(cycle_idx); vals.append(1.0)

        root = component_root[s]
        ancestors_s = set()
        u = s
        while True:
            ancestors_s.add(u)
            if u == root:
                break
            u = parent[u][0]
            if u == -1:
                break

        u = d
        d_path = []
        while u not in ancestors_s and u != -1:
            pu, pe, psign = parent[u]
            d_path.append((pe, psign))
            u = pu
        lca = u

        s_path = []
        u = s
        while u != lca and u != -1:
            pu, pe, psign = parent[u]
            s_path.append((pe, psign))
            u = pu

        for pe, psign in d_path:
            rows.append(pe); cols.append(cycle_idx); vals.append(float(psign))
        for pe, psign in reversed(s_path):
            rows.append(pe); cols.append(cycle_idx); vals.append(float(-psign))

        cycle_idx += 1

    n_cyc = cycle_idx
    if n_cyc == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
    else:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
    U = torch.sparse_coo_tensor(indices, values, size=(n_e, max(n_cyc, 1))).coalesce()
    return U, n_cyc


__all__ = ["build_cycle_basis"]
