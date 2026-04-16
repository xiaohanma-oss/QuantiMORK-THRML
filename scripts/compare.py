#!/usr/bin/env python3
"""Compare wavelet vs baseline training results."""

import json
import os


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    wavelet_path = os.path.join(results_dir, "wavelet_results.json")
    baseline_path = os.path.join(results_dir, "baseline_results.json")

    have_wavelet = os.path.exists(wavelet_path)
    have_baseline = os.path.exists(baseline_path)

    if not have_wavelet and not have_baseline:
        print("No results found. Run training first:")
        print("  python scripts/train.py --mode wavelet")
        print("  python scripts/train.py --mode baseline")
        return

    print("=" * 70)
    print("QuantiMORK-THRML vs PC-Transformers — Tiny Shakespeare Results")
    print("=" * 70)

    header = f"{'Metric':<25}"
    if have_baseline:
        header += f"{'PC-Transformers':>20}"
    if have_wavelet:
        header += f"{'QuantiMORK-THRML':>20}"
    print(header)
    print("-" * len(header))

    def final_epoch(results):
        return results["epochs"][-1]

    rows = [
        ("Parameters", lambda r: f"{r['config']['n_params']/1e6:.2f}M"),
        ("n_embed", lambda r: str(r["config"]["n_embed"])),
        ("n_blocks", lambda r: str(r["config"]["n_blocks"])),
        ("Final train PPL", lambda r: f"{final_epoch(r)['train_ppl']:.1f}"),
        ("Final val PPL", lambda r: f"{final_epoch(r)['val_ppl']:.1f}"),
        ("Final train energy", lambda r: f"{final_epoch(r)['train_energy']:.4f}"),
        ("Final val energy", lambda r: f"{final_epoch(r)['val_energy']:.4f}"),
    ]

    for name, fn in rows:
        line = f"{name:<25}"
        if have_baseline:
            b = load_results(baseline_path)
            line += f"{fn(b):>20}"
        if have_wavelet:
            w = load_results(wavelet_path)
            line += f"{fn(w):>20}"
        print(line)

    print()

    if have_wavelet:
        w = load_results(wavelet_path)
        print("TSU deployability:")
        print(f"  Inter-level connections/node: ≤5  (wavelet tree)")
        print(f"  Intra-level connections/node: 256/128/64  (dense per level, exceeds TSU ~12 limit)")
        if have_baseline:
            b = load_results(baseline_path)
            ratio = b["config"]["n_params"] / w["config"]["n_params"]
            print(f"  Parameter compression: {ratio:.1f}×")


if __name__ == "__main__":
    main()
