from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """
    KV 缓存池基类（物理存储层）。

    负责 KV 缓存的物理存储与读写，是"物理分页存储"的抽象接口。
    与 BasePrefixCache（逻辑前缀管理）分离：本类只管"把 KV 写到哪、从哪读"，
    不管"哪些前缀可以复用"。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 K 缓存张量。"""

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 V 缓存张量。"""

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把本层新算出的 k/v 按 out_loc 指定的位置写入缓存（分页写入）。"""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """缓存所在的设备。"""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """缓存的数据类型。"""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """缓存的总层数。"""


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """缓存句柄：代表一次前缀匹配/插入的结果，持有匹配到的缓存节点信息。"""

    cached_len: int  # 已缓存的长度（token 数）

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """返回匹配到的前缀在缓存中的物理索引序列。"""


class SizeInfo(NamedTuple):
    """缓存容量信息：可驱逐大小 + 受保护大小。"""

    evictable_size: int  # 可被 LRU 驱逐的 token 数
    protected_size: int  # 被引用保护、不可驱逐的 token 数

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    """插入前缀的结果。"""

    cached_len: int  # 插入前已存在于缓存中的长度（这部分应被释放/复用）
    handle: BaseCacheHandle  # 插入后返回的缓存句柄


class MatchResult(NamedTuple):
    """前缀匹配的结果。"""

    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


class BasePrefixCache(ABC):
    """前缀缓存基类（逻辑管理层）。

    负责前缀的匹配、插入、驱逐与引用计数保护，是 Radix Cache 的抽象接口。
    与 BaseKVCachePool（物理存储）分离：本类只管理"逻辑前缀"，
    通过句柄把逻辑前缀映射到物理索引。
    """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁一个缓存句柄。
        该操作不会修改缓存内容，只改变容量统计信息。
        句柄被锁定时，其对应的缓存不可被驱逐。
        在使用 `match_prefix` 返回的张量之前，必须先锁定句柄，
        否则该缓存可能被 evict 驱逐。

        Args:
            handle (BaseCacheHandle): 要锁定/解锁的缓存句柄。
            unlock (bool): 是否解锁。默认为 False（即锁定）。
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        匹配前缀，返回匹配到的前缀在缓存中的索引。
        该操作不会修改缓存。
        返回的索引只有在句柄被锁定时才安全可用。

        Args:
            input_ids (torch.Tensor): 要匹配的输入 id。形状: (seq_len,)
        Returns:
            MatchResult: 包含缓存句柄的匹配结果。
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        向缓存中插入一个新的前缀。
        该操作会修改缓存。
        Args:
            input_ids (torch.Tensor): 要插入的输入 id。形状: (seq_len,)
            indices (torch.Tensor): 存储新前缀的物理索引。形状: (seq_len,)

        Returns:
            InsertResult: 插入的结果。
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        从缓存中驱逐一些前缀以释放空间。
        该操作会修改缓存。
        注意：驱逐 0 总是安全的，且不做任何事。
        注意：实际驱逐的大小可能大于请求的大小。
        Args:
            size (int): 要驱逐的大小。

        Returns:
            torch.Tensor: 被驱逐的物理索引。形状: (evict_size,)
        Raises:
            RuntimeError: 如果请求的大小大于可驱逐的大小。
        """

    @abstractmethod
    def reset(self) -> None:
        """重置缓存管理器及其底层缓存。"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """获取缓存的容量信息。"""

    @abstractmethod
    def check_integrity(self) -> None:
        """检查缓存的完整性，若缓存损坏则抛出异常。"""
