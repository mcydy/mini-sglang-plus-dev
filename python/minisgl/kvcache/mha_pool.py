from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    MHA（多头注意力）KV 缓存池：物理分页存储实现。

    缓存张量形状为 (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)：
    - 第 0 维：2，分别对应 K 缓存和 V 缓存
    - 第 1 维：层数
    - 第 2 维：物理页数（分页管理）
    - 第 3 维：每页的 token 数（page_size）
    - 第 4 维：本进程负责的 KV 头数（张量并行切分后）
    - 第 5 维：每个头的维度

    张量并行：KV 头按 TP 大小切分，每个进程只存自己负责的 KV 头；
    当 KV 头数小于 TP 数时允许复制（allow_replicate=True）。
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        # 计算本进程负责的 KV 头数（TP 切分，允许复制）
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 一次性分配整块缓存：2 表示 [K, V]
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]  # K 缓存视图
        self._v_buffer = self._kv_buffer[1]  # V 缓存视图
        self._device = device
        # 展平后的存储形状：把 (num_pages, page_size) 合并为 (num_pages * page_size)
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 K 缓存。"""
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 V 缓存。"""
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把本层新算出的 k/v 按 out_loc 指定的物理位置写入缓存（调用自定义 CUDA 内核）。"""
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,  # 每个 token 对应的物理位置
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
