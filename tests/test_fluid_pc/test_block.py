"""FluidPCBlock + FluidWaveletTransformerBlock: shape, finiteness, autograd."""

import torch

from quantimork_thrml.fluid_pc import FluidPCBlock


class _BlockConfig:
    block_size = 8
    n_embed = 16
    wavelet_n_levels = 2
    fluid_outer_iters = 2
    mpc_horizon = 2
    mpc_inner_steps = 3


def test_shape_preserved():
    torch.manual_seed(0)
    block = FluidPCBlock(_BlockConfig())
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_output_is_finite():
    torch.manual_seed(0)
    block = FluidPCBlock(_BlockConfig())
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert torch.isfinite(y).all().item()


def test_grad_flows_to_trainable_params():
    torch.manual_seed(0)
    block = FluidPCBlock(_BlockConfig())
    x = torch.randn(2, 8, 16)
    loss = (block(x) ** 2).sum()
    loss.backward()
    n_with_grad = 0
    n_total = 0
    for p in block.parameters():
        n_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    assert n_with_grad == n_total


def test_solenoidal_first_means_leray_is_minor_correction():
    """§11.1 conformance: u from MPC is solenoidal-by-construction, so the
    safety-net Leray pass should change it by < 5% in norm."""
    torch.manual_seed(0)
    block = FluidPCBlock(_BlockConfig())
    g = block.graph
    rho = torch.rand(2, g.n_nodes)
    rho = rho / rho.sum(-1, keepdim=True)
    u_pre = block.mpc(rho, rho.detach())
    u_post = block.leray(u_pre)
    rel = (u_post - u_pre).norm(dim=-1) / u_pre.norm(dim=-1).clamp_min(1e-8)
    assert rel.max().item() < 0.05


def test_one_optimizer_step_changes_params():
    torch.manual_seed(0)
    block = FluidPCBlock(_BlockConfig())
    x = torch.randn(2, 8, 16)
    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in block.named_parameters()}
    loss = (block(x) ** 2).sum()
    loss.backward()
    opt.step()
    changed = sum(
        not torch.allclose(before[n], p)
        for n, p in block.named_parameters()
    )
    assert changed > 0
