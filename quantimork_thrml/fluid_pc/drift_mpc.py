"""
Receding-horizon MPC drift solver.

We optimize per-edge raw velocity u_raw ∈ ℝ^|E| (no cycle-space basis in
Phase 1 — the divergence-free constraint is enforced downstream by Leray
projection) over a horizon of H rollout steps. The control cost penalizes
||u||^2 and the terminal cost shapes rho_H toward a learned `value_field`
proxy via squared distance.

Cost (per batch element):
    J(u_raw) = sum_{t=0..H-1} 0.5 * control_weight * ||u_t||^2
             + 0.5 * terminal_weight * ||rho_H - value_field||^2

`u_t` is held constant across t in this Phase-1 implementation (constant
control), so only one |E|-dim vector is optimized per batch. This is the
simplest receding-horizon variant; richer per-step parameterization is
straightforward to add later.

By default `mpc_differentiable=False`: the optimizer runs inside a no_grad
context and only the final `u_raw` is returned to the outer pass as a
detached constant. Set True to backprop through inner Adam steps.
"""

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.advection import ConservativeAdvection
from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector


class MPCDrift(nn.Module):
    def __init__(self, graph: WaveletGraph, leray: LerayProjector,
                 advection: ConservativeAdvection,
                 horizon: int = 4, inner_steps: int = 5,
                 inner_lr: float = 0.1,
                 control_weight: float = 1.0,
                 terminal_weight: float = 1.0,
                 mpc_differentiable: bool = False,
                 cfl_target: float = 0.4):
        super().__init__()
        self.graph = graph
        self.leray = leray
        self.advection = advection
        self.horizon = horizon
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.control_weight = control_weight
        self.terminal_weight = terminal_weight
        self.mpc_differentiable = mpc_differentiable
        self.cfl_target = cfl_target

    def _rollout_cost(self, u_raw: torch.Tensor, rho0: torch.Tensor,
                      value_field: torch.Tensor) -> torch.Tensor:
        u = self.leray(u_raw)
        dt = self.advection.rescale_dt(u, target_cfl=self.cfl_target)
        rho = rho0
        for _ in range(self.horizon):
            rho = self.advection(rho, u, dt)
        control = 0.5 * self.control_weight * (u_raw ** 2).sum(dim=-1)
        terminal = 0.5 * self.terminal_weight * \
            ((rho - value_field) ** 2).sum(dim=-1)
        return control + terminal

    def _solve(self, rho0: torch.Tensor,
               value_field: torch.Tensor) -> torch.Tensor:
        B = rho0.shape[0]
        u_raw = torch.zeros(B, self.graph.n_edges,
                            device=rho0.device, dtype=rho0.dtype,
                            requires_grad=True)
        opt = torch.optim.Adam([u_raw], lr=self.inner_lr)
        for _ in range(self.inner_steps):
            opt.zero_grad()
            cost = self._rollout_cost(u_raw, rho0, value_field).sum()
            cost.backward()
            opt.step()
        return u_raw.detach()

    def forward(self, rho: torch.Tensor,
                value_field: torch.Tensor) -> torch.Tensor:
        """Return the projected velocity `u = Leray(u_raw_optimal)`."""
        if self.mpc_differentiable:
            u_raw = self._solve_differentiable(rho, value_field)
        else:
            with torch.enable_grad():
                rho_d = rho.detach()
                value_d = value_field.detach()
                u_raw = self._solve(rho_d, value_d)
        return self.leray(u_raw)

    def _solve_differentiable(self, rho0: torch.Tensor,
                              value_field: torch.Tensor) -> torch.Tensor:
        B = rho0.shape[0]
        u_raw = torch.zeros(B, self.graph.n_edges,
                            device=rho0.device, dtype=rho0.dtype)
        for _ in range(self.inner_steps):
            u_raw = u_raw.detach().requires_grad_(True)
            cost = self._rollout_cost(u_raw, rho0, value_field).sum()
            grad, = torch.autograd.grad(cost, u_raw, create_graph=True)
            u_raw = u_raw - self.inner_lr * grad
        return u_raw
