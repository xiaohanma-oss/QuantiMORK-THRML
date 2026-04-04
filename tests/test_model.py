"""Smoke test for WaveletPCTransformer forward pass."""

import sys
import os
import pytest
import torch

# Ensure vendor is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "PC-Transformers"))


class _MinimalConfig:
    """Minimal config matching PC-Transformers' expected attributes."""
    vocab_size = 256
    block_size = 32
    n_embed = 128       # small for testing (must be divisible by 2^3=8)
    num_heads = 4
    n_blocks = 2
    T = 2               # PC inference iterations
    dropout = 0.0
    lr = 0.001
    update_bias = False
    internal_energy_fn_name = "pc_e"
    output_energy_fn_name = "kld"
    wavelet_n_levels = 3
    use_flash_attention = False


@pytest.fixture
def config():
    return _MinimalConfig()


class TestWaveletPCTransformerForward:
    def test_forward_runs(self, config):
        from quantimork_thrml.model import WaveletPCTransformer

        model = WaveletPCTransformer(config)
        B, S = 2, config.block_size
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        target_ids = torch.randint(0, config.vocab_size, (B, S))

        logits = model(target_ids, input_ids)
        assert logits.shape == (B, S, config.vocab_size)

    def test_output_finite(self, config):
        from quantimork_thrml.model import WaveletPCTransformer

        model = WaveletPCTransformer(config)
        B, S = 2, config.block_size
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        target_ids = torch.randint(0, config.vocab_size, (B, S))

        logits = model(target_ids, input_ids)
        assert torch.isfinite(logits).all()

    def test_wavelet_mlp_fewer_params(self, config):
        """Wavelet model should have fewer MLP params than dense."""
        from quantimork_thrml.model import WaveletPCTransformer, WaveletMLP
        from quantimork_thrml.wavelet_linear import WaveletLinear

        model = WaveletPCTransformer(config)
        wavelet_params = sum(
            p.numel() for block in model.blocks
            for p in block.mlp.wavelet.parameters())

        # Dense equivalent: fc1(128→512) + fc2(512→128)
        dense_params = 128 * 512 + 512 + 512 * 128 + 128
        assert wavelet_params < dense_params

    def test_wavelet_weights_update_after_forward(self, config):
        """Wavelet per-level weights must change after a forward pass (PC Hebbian update)."""
        from quantimork_thrml.model import WaveletPCTransformer

        model = WaveletPCTransformer(config)
        model.register_all_lateral_weights()
        B, S = 2, config.block_size
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        target_ids = torch.randint(0, config.vocab_size, (B, S))

        # Snapshot weights before forward
        before = {}
        for idx, block in enumerate(model.blocks):
            wl = block.mlp.wavelet
            for i, dt in enumerate(wl.detail_transforms):
                before[f"block{idx}_detail{i}"] = dt.weight.data.clone()
            before[f"block{idx}_approx"] = wl.approx_transform.weight.data.clone()

        # Run forward (PC inference + Hebbian updates)
        model(target_ids, input_ids)

        # At least some wavelet weights must have changed
        any_changed = False
        for idx, block in enumerate(model.blocks):
            wl = block.mlp.wavelet
            for i, dt in enumerate(wl.detail_transforms):
                key = f"block{idx}_detail{i}"
                if not torch.equal(before[key], dt.weight.data):
                    any_changed = True
                    break
            if not any_changed:
                key = f"block{idx}_approx"
                if not torch.equal(before[key], wl.approx_transform.weight.data):
                    any_changed = True
            if any_changed:
                break

        assert any_changed, (
            "No wavelet weights changed after forward pass — "
            "Hebbian PC update is not reaching per-level weights"
        )
