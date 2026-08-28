# =============================================================================
# norm.py — 归一化层
#
# 实现 RMS 归一化：
#   - RMSNorm      : 标准 RMS 归一化（封装 flashinfer 的 rmsnorm）
#   - RMSNormFused : 融合"残差加法 + RMSNorm"（封装 flashinfer 的 fused_add_rmsnorm）
#
# RMSNormFused 是"残差流"设计的关键：forward(x, residual) 原地执行
# x += residual 后归一化，返回 (x, residual)，省去单独的残差张量。
# =============================================================================
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """RMS 归一化：x / sqrt(mean(x^2) + eps) * weight。"""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """原地归一化（out=x），避免额外分配输出张量。"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """融合残差 + RMSNorm：原地执行 x += residual 后归一化。

    forward(x, residual) 返回 (x, residual)：
        - residual 为 None 时：直接归一化，返回 (归一化结果, x)
        - residual 非 None 时：原地 x += residual 后归一化，返回 (x, residual)
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
