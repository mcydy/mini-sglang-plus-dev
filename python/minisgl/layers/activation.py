# =============================================================================
# activation.py — 门控激活函数（SwiGLU / GeGLU）
#
# 纯 PyTorch 实现，不再依赖 flashinfer。
# 输入是 gate_up 拼接张量：最后一维为 2 * intermediate_size，
# 前半是 gate，后半是 up；输出最后一维为 intermediate_size。
# =============================================================================
from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up，输入最后一维为 2 * d，输出为 d。"""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """GeGLU: gelu(gate) * up，输入最后一维为 2 * d，输出为 d。"""
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


__all__ = ["silu_and_mul", "gelu_and_mul"]
