"""
Leray projection: enforce ∇·u = 0 via graph-Poisson solve.

Given a raw edge field u_raw on the wavelet graph, find pressure p so that
    L p = B^T u_raw      (graph Poisson, L = B^T B)
then project
    u = u_raw - B p     (which has B^T u = 0).

The Laplacian L is sparse SPD up to its 1-D null space (constants), so we
solve with conjugate gradient. The null-space is removed by recentering p
and the right-hand side at each iteration.

Backward:
    For training stability we expose `LerayProjector(...)` returning
    `u_proj = u_raw - B p(u_raw)`. p depends linearly on u_raw, so the
    derivative of `u_proj` with respect to u_raw is the same projection
    operator. We use a custom `torch.autograd.Function` that runs CG in
    forward and re-runs CG in backward (implicit differentiation), which is
    cheaper and more stable than unrolling the iterations.
"""

from typing import Optional

import torch
import torch.nn as nn

from quantimork_thrml.fluid_pc.graph import WaveletGraph


def _zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _cg(graph: WaveletGraph, rhs: torch.Tensor, tol: float, max_iter: int,
        x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Solve L x = rhs with conjugate gradient, projecting out the constant
    null-space of L at each step.

    Numerical safeguards:
      - rs_old, rs_new are clamped above a tiny floor so 0/0 cannot arise.
      - alpha is set to zero on already-converged batches (rs_old < eps).
      - We use both a relative (rs / rhs_norm) and an absolute (rs) check
        so per-batch elements with near-zero rhs converge instantly.
    """
    eps = 1e-12
    rhs = _zero_mean(rhs)
    x = torch.zeros_like(rhs) if x0 is None else _zero_mean(x0)
    r = rhs - graph.laplacian_apply(x)
    r = _zero_mean(r)
    p = r.clone()
    rs_old = (r * r).sum(dim=-1, keepdim=True)
    rhs_norm = (rhs * rhs).sum(dim=-1, keepdim=True)
    abs_tol_sq = tol * tol
    for _ in range(max_iter):
        Ap = _zero_mean(graph.laplacian_apply(p))
        denom = (p * Ap).sum(dim=-1, keepdim=True)
        active = (rs_old > eps) & (denom.abs() > eps)
        safe_denom = torch.where(active, denom, torch.ones_like(denom))
        alpha = torch.where(active, rs_old / safe_denom,
                            torch.zeros_like(rs_old))
        x = x + alpha * p
        r = r - alpha * Ap
        r = _zero_mean(r)
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        rel = rs_new / rhs_norm.clamp_min(eps)
        if (rs_new.max().item() < abs_tol_sq
                and rel.max().item() < abs_tol_sq):
            break
        safe_rs_old = torch.where(active, rs_old, torch.ones_like(rs_old))
        beta = torch.where(active, rs_new / safe_rs_old,
                           torch.zeros_like(rs_new))
        p = r + beta * p
        rs_old = rs_new
    return _zero_mean(x)


class _LerayFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_raw: torch.Tensor,
                graph: WaveletGraph, tol: float, max_iter: int) -> torch.Tensor:
        rhs = graph.divergence(u_raw)
        p = _cg(graph, rhs, tol=tol, max_iter=max_iter)
        u_proj = u_raw - graph.gradient(p)
        ctx.graph = graph
        ctx.tol = tol
        ctx.max_iter = max_iter
        return u_proj

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        graph: WaveletGraph = ctx.graph
        tol: float = ctx.tol
        max_iter: int = ctx.max_iter
        rhs = graph.divergence(grad_out)
        q = _cg(graph, rhs, tol=tol, max_iter=max_iter)
        grad_u_raw = grad_out - graph.gradient(q)
        return grad_u_raw, None, None, None


class LerayProjector(nn.Module):
    """Differentiable Leray projection on a `WaveletGraph`."""

    def __init__(self, graph: WaveletGraph, tol: float = 1e-4,
                 max_iter: int = 50):
        super().__init__()
        self.graph = graph
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, u_raw: torch.Tensor) -> torch.Tensor:
        return _LerayFn.apply(u_raw, self.graph, self.tol, self.max_iter)
