"""κ annealing (§10.7) + four diagnostic scalars (§10.7)."""

import torch

from quantimork_thrml.fluid_pc import FluidPCBlock


class _Cfg:
    block_size = 8
    n_embed = 16
    wavelet_n_levels = 2
    fluid_outer_iters = 4
    mpc_horizon = 2
    mpc_inner_steps = 3


def _new_block():
    torch.manual_seed(0)
    return FluidPCBlock(_Cfg())


def test_kappa_anneals_cosine_to_zero():
    b = _new_block()
    x = torch.randn(2, 8, 16)
    _ = b(x)
    kappas = [d["kappa"] for d in b.last_diagnostics]
    assert len(kappas) == _Cfg.fluid_outer_iters
    assert kappas[0] == _Cfg().__class__.__dict__.get("kappa_init", 0.01) or \
           kappas[0] == b.cfg.kappa_init
    assert abs(kappas[-1]) < 1e-9
    for a, b_ in zip(kappas, kappas[1:]):
        assert b_ <= a + 1e-9


def test_diagnostics_record_four_scalars_per_iteration():
    b = _new_block()
    x = torch.randn(2, 8, 16)
    _ = b(x)
    assert len(b.last_diagnostics) == _Cfg.fluid_outer_iters
    for d in b.last_diagnostics:
        for key in ("k", "kappa", "band_mass", "exp_distance",
                    "div_norm", "achieved_cfl"):
            assert key in d
        assert isinstance(d["band_mass"], float)
        assert isinstance(d["div_norm"], float)
        assert d["achieved_cfl"] <= 0.45 + 1e-3


def test_three_part_free_energy_is_populated():
    b = _new_block()
    x = torch.randn(2, 8, 16)
    _ = b(x)
    fe = b.last_free_energy
    assert "reaction" in fe and "v_anchor" in fe
    assert torch.isfinite(fe["reaction"]).all().item()
    assert torch.isfinite(fe["v_anchor"]).all().item()
    assert fe["reaction"].requires_grad
    assert fe["v_anchor"].requires_grad


def test_value_head_receives_gradient():
    b = _new_block()
    x = torch.randn(2, 8, 16)
    y = b(x)
    (y ** 2).sum().backward()
    assert b.value_head.weight.grad is not None
    assert b.value_head.weight.grad.abs().sum().item() > 0
