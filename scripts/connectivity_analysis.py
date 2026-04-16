#!/usr/bin/env python
"""
Connectivity analysis for WaveletLinear per-level weight matrices.

Measures the actual connection degree distribution induced by per-level
Linear layers, both directly (W row non-zeros) and through the induced
precision matrix (W^T W row non-zeros).

Usage:
    python scripts/connectivity_analysis.py                    # random init
    python scripts/connectivity_analysis.py --checkpoint path  # trained weights
"""

import argparse
import sys

import numpy as np
import torch

from quantimork_thrml.wavelet_linear import WaveletLinear

TSU_NEIGHBOR_LIMIT = 12


def analyze_connectivity(weight_matrix, label, threshold=1e-3):
    """Analyze connectivity of a single weight matrix.

    Returns dict with stats for both W (direct) and W^T W (induced coupling).
    """
    W = weight_matrix.detach().cpu().numpy()
    n = W.shape[0]

    # Direct connectivity: non-zero entries per row of W
    direct_nnz = np.array([(np.abs(W[i]) > threshold).sum() for i in range(n)])

    # Induced coupling: W^T W (precision matrix off-diagonal)
    WtW = W.T @ W
    # Zero out diagonal for coupling analysis
    np.fill_diagonal(WtW, 0)
    induced_nnz = np.array([(np.abs(WtW[i]) > threshold).sum() for i in range(n)])

    return {
        "label": label,
        "size": n,
        "direct": {
            "max": int(direct_nnz.max()),
            "median": float(np.median(direct_nnz)),
            "mean": float(direct_nnz.mean()),
            "min": int(direct_nnz.min()),
        },
        "induced": {
            "max": int(induced_nnz.max()),
            "median": float(np.median(induced_nnz)),
            "mean": float(induced_nnz.mean()),
            "min": int(induced_nnz.min()),
        },
    }


def sweep_thresholds(weight_matrix, thresholds):
    """Sweep sparsity thresholds and report max direct connectivity."""
    W = weight_matrix.detach().cpu().numpy()
    n = W.shape[0]
    results = []
    for t in thresholds:
        nnz_per_row = np.array([(np.abs(W[i]) > t).sum() for i in range(n)])
        results.append({
            "threshold": t,
            "max_degree": int(nnz_per_row.max()),
            "median_degree": float(np.median(nnz_per_row)),
            "pct_under_tsu": float((nnz_per_row <= TSU_NEIGHBOR_LIMIT).mean() * 100),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="WaveletLinear connectivity analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dim", type=int, default=512,
                        help="Feature dimension (default: 512)")
    parser.add_argument("--n-levels", type=int, default=3,
                        help="Wavelet decomposition levels (default: 3)")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Sparsity threshold for counting non-zeros")
    args = parser.parse_args()

    # Load or create WaveletLinear
    wl = WaveletLinear(args.dim, args.dim, n_levels=args.n_levels)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        # Try to extract WaveletLinear weights from model checkpoint
        wl_keys = {k: v for k, v in state.items() if "wavelet" in k.lower()}
        if wl_keys:
            wl.load_state_dict(wl_keys, strict=False)
            print(f"Loaded weights from {args.checkpoint}")
        else:
            print(f"Warning: no wavelet keys found in {args.checkpoint}, using random init")
    else:
        print(f"Using random initialization (dim={args.dim}, n_levels={args.n_levels})")

    print(f"\n{'='*70}")
    print(f"CONNECTIVITY ANALYSIS — WaveletLinear({args.dim}, {args.dim}, n_levels={args.n_levels})")
    print(f"TSU neighbor limit: {TSU_NEIGHBOR_LIMIT}")
    print(f"Sparsity threshold: {args.threshold}")
    print(f"{'='*70}\n")

    # Analyze each level
    all_stats = []
    for i, dt in enumerate(wl.detail_transforms):
        stats = analyze_connectivity(dt.weight, f"Detail Level {i+1}", args.threshold)
        all_stats.append(stats)

    stats = analyze_connectivity(wl.approx_transform.weight,
                                 f"Approx Level {args.n_levels}", args.threshold)
    all_stats.append(stats)

    # Print results table
    print(f"{'Level':<20} {'Size':>6} │ {'Direct (W)':^25} │ {'Induced (WᵀW)':^25}")
    print(f"{'':20} {'':>6} │ {'max':>6} {'med':>6} {'mean':>6} │ {'max':>6} {'med':>6} {'mean':>6}")
    print(f"{'─'*20} {'─'*6}─┼─{'─'*25}─┼─{'─'*25}")

    for s in all_stats:
        d, ind = s["direct"], s["induced"]
        exceeds_w = " ⚠" if d["max"] > TSU_NEIGHBOR_LIMIT else " ✓"
        exceeds_i = " ⚠" if ind["max"] > TSU_NEIGHBOR_LIMIT else " ✓"
        print(f"{s['label']:<20} {s['size']:>6} │ "
              f"{d['max']:>5}{exceeds_w} {d['median']:>5.0f} {d['mean']:>6.1f} │ "
              f"{ind['max']:>5}{exceeds_i} {ind['median']:>5.0f} {ind['mean']:>6.1f}")

    print(f"\n⚠ = exceeds TSU ~{TSU_NEIGHBOR_LIMIT} neighbor limit")

    # Threshold sweep on the largest level
    largest = wl.detail_transforms[0]
    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP — Detail Level 1 (size={largest.in_features})")
    print(f"{'='*70}\n")

    thresholds = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5]
    sweep = sweep_thresholds(largest.weight, thresholds)

    print(f"{'Threshold':>10} │ {'Max Degree':>10} {'Median':>8} {'% ≤ TSU':>8}")
    print(f"{'─'*10}─┼─{'─'*10} {'─'*8} {'─'*8}")
    for r in sweep:
        marker = "✓" if r["max_degree"] <= TSU_NEIGHBOR_LIMIT else "⚠"
        print(f"{r['threshold']:>10.4f} │ {r['max_degree']:>9} {marker} "
              f"{r['median_degree']:>7.0f} {r['pct_under_tsu']:>7.1f}%")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for s in all_stats:
        d = s["direct"]
        status = "EXCEEDS" if d["max"] > TSU_NEIGHBOR_LIMIT else "OK"
        print(f"  {s['label']}: {d['max']} direct connections/node "
              f"(TSU limit {TSU_NEIGHBOR_LIMIT}) → {status}")


if __name__ == "__main__":
    main()
