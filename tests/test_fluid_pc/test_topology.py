"""Wavelet graph topology — node/edge counts and band layout."""

from quantimork_thrml.fluid_pc.topology import build_topology


def test_node_count_equals_S_times_D():
    topo = build_topology(S=4, D=8, n_levels=2)
    assert topo.n_nodes == 4 * 8


def test_band_sizes_sum_to_D():
    topo = build_topology(S=4, D=16, n_levels=3)
    assert sum(topo.band_sizes) == 16


def test_band_sizes_follow_haar_pyramid():
    topo = build_topology(S=2, D=16, n_levels=3)
    assert topo.band_sizes == (8, 4, 2, 2)


def test_lateral_edges_are_S_minus_1_per_node_column():
    topo = build_topology(S=4, D=8, n_levels=2)
    assert topo.n_lateral == (topo.S - 1) * topo.D


def test_cross_edges_match_haar_tree():
    topo = build_topology(S=4, D=8, n_levels=2)
    cross_per_s = topo.D - topo.D // 2 + topo.D // 4
    assert topo.n_cross == cross_per_s * topo.S


def test_edge_endpoints_are_distinct():
    topo = build_topology(S=4, D=8, n_levels=2)
    for u, v in zip(topo.edge_src, topo.edge_dst):
        assert u != v


def test_node_indices_in_range():
    topo = build_topology(S=4, D=8, n_levels=2)
    for u, v in zip(topo.edge_src, topo.edge_dst):
        assert 0 <= u < topo.n_nodes
        assert 0 <= v < topo.n_nodes
