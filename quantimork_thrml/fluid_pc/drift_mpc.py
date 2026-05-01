"""
Receding-horizon MPC drift solver — solenoidal-first (IFN §10.5 / §11.1).

Per IFN §11.1 ("Solenoidal parameterization first, projection second"), we
optimize coefficients α ∈ ℝ^{B, K_cyc} in the cycle-space basis {U_k}
rather than raw per-edge velocity. The velocity u = U @ α is divergence-
free **by construction**, so no Leray projection is needed during the
inner cost rollout. A single Leray pass downstream in `FluidPCBlock` is
kept as the §10.10 safety net (numerical drift through fp32 advection).

Cost:
    J(α) = sum_{t=0..H-1} 0.5 * control_weight * ||α||^2
         + 0.5 * terminal_weight * ||rho_H - value_field||^2
α is held constant across t (constant-control variant); H-step structure
preserved by repeatedly applying the same u over the rollout.

By default `mpc_differentiable=False`: the inner Adam optimization runs
under a torch.enable_grad scope on a fresh α with stop-grad on rho/value;
final u is returned as a detached constant to the outer forward.
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

    def _rollout_cost(self, alpha: torch.Tensor, rho0: torch.Tensor,
                      value_field: torch.Tensor) -> torch.Tensor:
        u = self.graph.apply_U(alpha)
        dt = self.advection.rescale_dt(u, target_cfl=self.cfl_target)
        rho = rho0
        for _ in range(self.horizon):
            rho = self.advection(rho, u, dt)
        control = 0.5 * self.control_weight * (alpha ** 2).sum(dim=-1)
        terminal = 0.5 * self.terminal_weight * \
            ((rho - value_field) ** 2).sum(dim=-1)
        return control + terminal

    def _solve(self, rho0: torch.Tensor,
               value_field: torch.Tensor) -> torch.Tensor:
        B = rho0.shape[0]
        K_cyc = max(self.graph.n_cycles, 1)
        alpha = torch.zeros(B, K_cyc,
                            device=rho0.device, dtype=rho0.dtype,
                            requires_grad=True)
        opt = torch.optim.Adam([alpha], lr=self.inner_lr)
        for _ in range(self.inner_steps):
            opt.zero_grad()
            cost = self._rollout_cost(alpha, rho0, value_field).sum()
            cost.backward()
            opt.step()
        return alpha.detach()

    def forward(self, rho: torch.Tensor,
                value_field: torch.Tensor) -> torch.Tensor:
        """Return the divergence-free velocity `u = U @ alpha*`.

        No Leray projection here — `u` is solenoidal by construction. The
        `FluidPCBlock.forward` calls Leray once afterwards as a safety net.
        """
        if self.mpc_differentiable:
            alpha = self._solve_differentiable(rho, value_field)
        else:
            with torch.enable_grad():
                rho_d = rho.detach()
                value_d = value_field.detach()
                alpha = self._solve(rho_d, value_d)
        return self.graph.apply_U(alpha)

    def _solve_differentiable(self, rho0: torch.Tensor,
                              value_field: torch.Tensor) -> torch.Tensor:
        B = rho0.shape[0]
        K_cyc = max(self.graph.n_cycles, 1)
        alpha = torch.zeros(B, K_cyc,
                            device=rho0.device, dtype=rho0.dtype)
        for _ in range(self.inner_steps):
            alpha = alpha.detach().requires_grad_(True)
            cost = self._rollout_cost(alpha, rho0, value_field).sum()
            grad, = torch.autograd.grad(cost, alpha, create_graph=True)
            alpha = alpha - self.inner_lr * grad
        return alpha
