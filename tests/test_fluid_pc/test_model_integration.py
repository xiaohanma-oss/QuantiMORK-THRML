"""WaveletPCTransformer with attn_mode='fluid' — drop-in compatibility."""

import torch
import torch.nn.functional as F

from quantimork_thrml.model import WaveletPCTransformer


class _ModelConfig:
    vocab_size = 32
    block_size = 8
    n_embed = 16
    num_heads = 2
    n_blocks = 1
    dropout = 0.0
    T = 2
    lr = 0.01
    update_bias = False
    internal_energy_fn_name = "mse"
    output_energy_fn_name = "cross_entropy"
    use_flash_attention = False
    wavelet_n_levels = 2
    attn_mode = "fluid"
    fluid_outer_iters = 2
    mpc_horizon = 2
    mpc_inner_steps = 3


def test_fluid_mode_forward_returns_logits():
    torch.manual_seed(0)
    model = WaveletPCTransformer(_ModelConfig())
    input_ids = torch.randint(0, 32, (2, 8))
    target_ids = torch.randint(0, 32, (2, 8))
    logits = model(target_ids, input_ids)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all().item()


def test_fluid_mode_loss_decreases_in_one_step():
    torch.manual_seed(0)
    model = WaveletPCTransformer(_ModelConfig())
    input_ids = torch.randint(0, 32, (2, 8))
    target_ids = torch.randint(0, 32, (2, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    logits = model(target_ids, input_ids)
    loss_before = F.cross_entropy(
        logits.reshape(-1, 32), target_ids.reshape(-1)).item()
    loss = F.cross_entropy(logits.reshape(-1, 32), target_ids.reshape(-1))
    loss.backward()
    opt.step()

    logits = model(target_ids, input_ids)
    loss_after = F.cross_entropy(
        logits.reshape(-1, 32), target_ids.reshape(-1)).item()
    assert loss_after < loss_before


def test_register_lateral_skips_fluid_blocks():
    torch.manual_seed(0)
    model = WaveletPCTransformer(_ModelConfig())
    model.register_all_lateral_weights()
