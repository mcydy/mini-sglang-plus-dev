# =============================================================================
# core.py — mini-sglang 核心数据结构
#
# 本模块定义了贯穿整个推理框架的基础数据结构，是引擎(Engine)与调度器
# (Scheduler)之间共享的"通用语言"：
#   - SamplingParams : 用户提供的采样参数（温度、top-k、top-p 等）
#   - Req            : 单个请求的状态（输入 token、缓存长度、输出长度等）
#   - Batch          : 一批请求（prefill 或 decode 阶段）
#   - Context        : 全局推理上下文（KV 缓存、页表、注意力后端等）
#
# 其中 Context 通过 set_global_ctx / get_global_ctx 以全局单例方式访问，
# 模型前向时从全局上下文取出当前正在执行的 Batch。
# =============================================================================
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    """采样参数：描述如何从 logits 中挑选下一个 token。

    属性：
        temperature: 采样温度，<=0 表示贪心（取 argmax）
        top_k: 只从概率最高的 top_k 个 token 中采样，-1 表示不限制
        top_p: 核采样阈值，只从累计概率达到 top_p 的最小集合中采样
        ignore_eos: 是否忽略结束符（用于强制生成固定长度）
        max_tokens: 最多生成的 token 数
    """

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """是否为贪心解码：温度<=0 或 top_k==1，且 top_p==1.0 时退化为 argmax。"""
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """单个请求的状态。

    关键长度概念：
        cached_len  : 已缓存到 KV cache 中的 token 数（前缀可被复用）
        device_len  : 已送入设备计算的 token 数
        max_device_len: 允许的最大 token 数（输入 + 输出）
        extend_len  = device_len - cached_len，即本次前向需要计算的 token 数
        remain_len  = max_device_len - device_len，即还能生成的 token 数

    属性：
        input_ids: 输入 token（CPU 张量，int32）
        table_idx: 该请求在页表(page_table)中占用的行号
        cache_handle: 指向前缀缓存(KV cache)的句柄，用于前缀复用
    """

    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """剩余可生成的 token 数。"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次前向需要计算的 token 数（尚未缓存的部分）。"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一步 decode：把已计算部分标记为已缓存，并推进一个 token。"""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """把新生成的 token 追加到 CPU 侧的输入序列末尾。"""
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """是否还能继续 decode（还有剩余生成额度）。"""
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """一批请求，是引擎前向计算的基本单位。

    phase 为 "prefill"（预填充，一次处理多个 token）或 "decode"（逐 token 生成）。
    以下字段由调度器(Scheduler)在准备批次时填充：
        input_ids   : 本批所有 token 的 id（从 token_pool 中按映射取出）
        positions   : 每个 token 的位置编码（用于 RoPE）
        out_loc     : 每个 token 的输出应写入的 KV cache 位置
        padded_reqs : 填充后的请求列表（为匹配 CUDA graph 的 batch size）
    attn_metadata 由注意力后端在 prepare_metadata 时填充。
    """

    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """是否为 prefill 阶段。"""
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """是否为 decode 阶段。"""
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """真实请求数。"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """填充后的请求数（可能大于 size，用于 CUDA graph）。"""
        return len(self.padded_reqs)


@dataclass
class Context:
    """全局推理上下文，持有引擎运行所需的共享资源。

    属性：
        page_size: 分页大小（KV cache 每页的 token 数）
        page_table: 页表，记录每个请求槽位每个位置对应的 KV cache 位置
                    （注意：此表始终按 page_size=1 处理，即存的是原始位置而非页号）
        attn_backend: 注意力后端（FlashAttention / FlashInfer 等）
        moe_backend: MoE 后端（仅 MoE 模型使用）
        kv_cache: KV cache 池
    """

    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """获取当前正在前向的 Batch（在 forward_batch 上下文内有效）。"""
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """上下文管理器：把 batch 挂载为当前批次，供模型前向时通过 ctx.batch 访问。"""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """设置全局上下文单例（只能在启动时调用一次）。"""
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """获取全局上下文单例，模型前向时据此访问当前 batch 与各类后端。"""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
