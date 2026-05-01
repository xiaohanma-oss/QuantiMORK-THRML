"""ConservativeAdvection: mass conservation, CFL rescale, diffusion."""

import torch

from quantimork_thrml.fluid_pc.advection import ConservativeAdvection
from quantimork_thrml.fluid_pc.graph import WaveletGraph


def test_mass_conservation_under_advection_only():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    adv = ConservativeAdvection(g, kappa_init=0.0)
    rho = torch.rand(2, g.n_nodes)
    u = 0.1 * torch.randn(2, g.n_edges)
    dt = torch.tensor(0.1)
    rho_new = adv(rho, u, dt)
    delta = (rho_new.sum(-1) - rho.sum(-1)).abs()
    assert delta.max().item() < 1e-4


def test_mass_conservation_under_diffusion_only():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    adv = ConservativeAdvection(g, kappa_init=0.05)
    rho = torch.rand(2, g.n_nodes)
    u = torch.zeros(2, g.n_edges)
    dt = torch.tensor(0.1)
    rho_new = adv(rho, u, dt)
    delta = (rho_new.sum(-1) - rho.sum(-1)).abs()
    assert delta.max().item() < 1e-4


def test_cfl_rescale_stays_in_target_band():
    target = 0.4
    u = torch.randn(2, 32) * 5.0
    dt = ConservativeAdvection.rescale_dt(u, target_cfl=target)
    cfl = (dt.unsqueeze(-1) * u.abs()).amax(dim=-1)
    assert (cfl <= target + 1e-5).all().item()


def test_cfl_rescale_caps_dt_when_u_is_zero():
    u = torch.zeros(2, 32)
    dt = ConservativeAdvection.rescale_dt(u, target_cfl=0.4, dt_max=1.0)
    assert (dt <= 1.0 + 1e-5).all().item()


def test_diffusion_smoothes_dirichlet_impulse():
    g = WaveletGraph(S=4, D=8, n_levels=2)
    adv = ConservativeAdvection(g, kappa_init=0.5)
    rho = torch.zeros(1, g.n_nodes)
    rho[0, 0] = 1.0
    u = torch.zeros(1, g.n_edges)
    dt = torch.tensor(0.5)
    rho_new = adv(rho, u, dt)
    assert rho_new[0, 0].item() < 1.0
    assert (rho_new[0, 1:] != 0).any().item()
