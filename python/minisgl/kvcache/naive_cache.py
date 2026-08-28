import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """朴素缓存句柄：永远匹配不到任何前缀（cached_len=0）。"""

    empty_tensor: torch.Tensor  # 由 NaivePrefixCache 设置的空张量

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        """返回空索引，表示没有任何前缀命中。"""
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    """朴素前缀缓存：不做任何前缀复用，作为 RadixPrefixCache 的占位实现。

    所有匹配都返回空结果，插入不实际存储，驱逐直接抛异常。
    用于关闭前缀缓存功能时的兜底实现。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """无操作：朴素缓存没有可保护的内容。"""
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """总是返回空匹配（无前缀命中）。"""
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """不实际插入，返回 cached_len=0。"""
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        """朴素缓存不支持驱逐。"""
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        """容量恒为 0。"""
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
