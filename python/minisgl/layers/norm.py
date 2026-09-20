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


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm：x / sqrt(mean(x^2) + eps) * weight，内部用 fp32 累加，输出保持输入 dtype。"""
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps)).to(x.dtype) * weight


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        x.copy_(_rmsnorm(x, self.weight, self.eps))


class RMSNormFused(BaseOP):
    """residual += x 后归一化：返回 (归一化结果, 更新后的 residual)。

    等价于 flashinfer 的 fused_add_rmsnorm：
        out       = rmsnorm(x + residual)
        residual' = x + residual
    """

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rmsnorm(x, self.weight, self.eps), x
        residual.add_(x)
        x.copy_(_rmsnorm(residual, self.weight, self.eps))
        return x, residual
