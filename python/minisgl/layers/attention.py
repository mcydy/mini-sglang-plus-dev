# =============================================================================
# attention.py — 注意力层
#
# 实现无状态注意力层 AttentionLayer：
#   1. 把 QKV 合并投影的输出切分为 Q/K/V
#   2. 可选 QK 归一化（Qwen3 使用，对 Q 和 K 分别做 RMSNorm）
#   3. 施加 RoPE 旋转（使用 ctx.batch.positions）
#   4. 调用注意力后端（FlashAttention / FlashInfer）计算注意力
#   5. 恢复输出形状
#
# 头数按 TP 切分：num_qo_heads / num_kv_heads 除以 TP size。
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """无状态注意力层：切分 QKV -> QK 归一化 -> RoPE -> 注意力后端 -> 恢复形状。"""

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        # 头数按 TP 切分（KV 头数小于 TP 数时允许复制）
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """前向：输入 QKV 合并投影输出，返回注意力输出（o_proj 的输入）。"""
        ctx = get_global_ctx()
        # 按 [qo_attn_dim, kv_attn_dim, kv_attn_dim] 切分 Q/K/V
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 施加 RoPE（使用当前 batch 的位置编码）
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        # 调用注意力后端（FlashAttention / FlashInfer）
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
