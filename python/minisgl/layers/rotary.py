# =============================================================================
# rotary.py — 旋转位置编码（RoPE）
#
# 实现 RoPE（Rotary Position Embedding）：
#   - apply_rope_with_cos_sin_cache_inplace : flashinfer 同名算子的纯 PyTorch 替代
#   - RotaryEmbedding : 预计算 cos/sin 缓存，forward 原地施加旋转
#   - _get_rope       : 按 rope_scaling 的 rope_type 分派（default/llama3/yarn）
#   - get_rope        : 全局缓存 RoPE 实例（functools.cache）
#
# 本版本不再依赖 flashinfer，行为与 flashinfer 的
# `apply_rope_with_cos_sin_cache_inplace(..., is_neox=True)` 保持一致：
#   * is_neox=True  : 前后半配对（split-half），等价于 HF 的 rotate_half
#   * is_neox=False : 相邻奇偶配对（interleave），即 GPT-J 风格
#   * cos_sin_cache : [max_pos, rotary_dim]，前 rotary_dim/2 是 cos，后一半是 sin
#
# RoPE 通过把位置信息编码进 Q/K 的旋转角度，使注意力能感知 token 的相对位置。
# =============================================================================
from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    """对最后一维施加旋转。

    Args:
        x:    [..., rotary_dim]（已切成只含参与旋转的维度）
        cos:  [..., 1, rotary_dim // 2] 或可广播到该形状
        sin:  同 cos
        is_neox: True 用前后半配对，False 用相邻奇偶配对
    """
    if is_neox:
        # (x_i, x_{i + d/2}) 成对旋转
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    # (x_{2i}, x_{2i+1}) 成对旋转
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fp32_compute: bool = True,
) -> None:
    """纯 PyTorch 版的 flashinfer `apply_rope_with_cos_sin_cache_inplace`。

    原地修改 query / key，无返回值。

    Args:
        positions:     [nnz] 位置索引（int32/int64）
        query:         [nnz, num_q_heads * head_size]，也可直接传 [nnz, H, head_size]
        key:           [nnz, num_kv_heads * head_size]，同上
        head_size:     每个 head 的维度
        cos_sin_cache: [max_seq_len, rotary_dim]，float32，前半 cos、后半 sin
        is_neox:       True=前后半配对（默认，Llama/Qwen 等），False=奇偶交错（GPT-J）
        fp32_compute:  是否在 fp32 下做旋转运算（与 flashinfer kernel 一致，精度略高）

    注意:
        query/key 必须是连续（contiguous）张量，否则 `.view` 会失败。
        参与旋转的维度由 cos_sin_cache 决定，支持 rotary_dim < head_size（部分旋转）。
    """
    num_tokens = positions.shape[0]
    # [nnz, num_heads * head_size] -> [nnz, num_heads, head_size]，零拷贝 view
    q = query.view(num_tokens, -1, head_size)
    k = key.view(num_tokens, -1, head_size)

    # [nnz, rotary_dim] -> cos/sin 各 [nnz, rotary_dim // 2] -> [nnz, 1, rotary_dim // 2]
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1] * 2
    q_rot, k_rot = q, k
    if rotary_dim < head_size:
        # 部分旋转：只有前 rotary_dim 维参与，其余维原样透传
        q_rot = q[..., :rotary_dim]
        k_rot = k[..., :rotary_dim]

    if fp32_compute:
        q_rot.copy_(_apply_rotary_emb(q_rot.float(), cos, sin, is_neox).to(q.dtype))
        k_rot.copy_(_apply_rotary_emb(k_rot.float(), cos, sin, is_neox).to(k.dtype))
    else:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_rot.copy_(_apply_rotary_emb(q_rot, cos, sin, is_neox))
        k_rot.copy_(_apply_rotary_emb(k_rot, cos, sin, is_neox))


class RotaryEmbedding(StateLessOP):
    """旋转位置编码层（无状态）。

    预计算 cos/sin 缓存（拼接为 _cos_sin_cache），forward 时按 positions
    对 query/key 原地施加旋转。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        is_neox_style: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style
        assert rotary_dim <= head_size, (rotary_dim, head_size)
        assert rotary_dim % 2 == 0 and head_size % 2 == 0

        # 计算逆频率：inv_freq[i] = 1 / base^(2i/rotary_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        # 作为 buffer 存储，不参与权重存取
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1).contiguous()

    def _cache(self, like: torch.Tensor) -> torch.Tensor:
        """取出 cos/sin 缓存，必要时搬到输入所在设备（只搬一次）。"""
        cache = self._cos_sin_cache
        if cache.device != like.device:
            cache = cache.to(device=like.device, non_blocking=True)
            self._cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 positions 对 query/key 原地施加 RoPE 旋转，返回 (query, key)。"""
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cache(query),
            is_neox=self.is_neox_style,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox_style: bool = True,
) -> RotaryEmbedding:
    """按 rope_scaling 的 rope_type 创建 RoPE 实例。

    支持的类型：
        - default : 原样（无缩放）
        - llama3  : 低/高频因子分段缩放（Llama-3 的长上下文扩展）
        - yarn    : YaRN 插值（通过 ramp 在低频/高频之间平滑过渡）
    """
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox_style=is_neox_style)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox_style=is_neox_style)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                # 在低/高频之间平滑过渡缩放因子
                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(
                head_dim, rotary_dim, max_position, base, post_process, is_neox_style=is_neox_style
            )

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 在 [low, high] 维度范围内用 ramp 平滑过渡缩放
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32, device=inv_freq.device) - low)
                    / max(high - low, 1),
                    0,
                    1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim, rotary_dim, max_position, base, post_process, is_neox_style=is_neox_style
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """设置 RoPE 实例所在的设备（meta device 下创建 RoPE 前必须先调用）。"""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox_style: bool = True,
) -> RotaryEmbedding:
    """获取（缓存的）RoPE 实例。

    由于 RoPE 需要真实设备（不能是 meta device），当当前设备为 meta 时，
    会切换到 _ROPE_DEVICE 上创建。
    """
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox_style)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox_style)


__all__ = [
    "get_rope",
    "RotaryEmbedding",
    "set_rope_device",
    "apply_rope_with_cos_sin_cache_inplace",
]