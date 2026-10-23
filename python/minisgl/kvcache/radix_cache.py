from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]




_CACHE_STAT: Dict[str, int] = {"req": 0, "hit": 0, "total": 0}


def _log_cache_hit(prefix_len: int, total: int) -> None:
    """打印本次请求的缓存命中情况 + 累计命中率。"""
    s = _CACHE_STAT
    s["req"] += 1
    s["hit"] += prefix_len
    s["total"] += total
    rate = 100.0 * prefix_len / total if total > 0 else 0.0
    cum = 100.0 * s["hit"] / s["total"] if s["total"] > 0 else 0.0
    print(
        f"[CACHE] req#{s['req']:4d}  hit={prefix_len:6d}/{total:<6d} "
        f"{rate:6.2f}%  | cum: {s['hit']}/{s['total']} = {cum:6.2f}%",
        flush=True,
    )


class RadixTreeNode:
    """Radix 树节点：代表一段连续的前缀 token 序列。

    每个节点保存一段 key（token id 序列）和对应的 value（物理索引序列）。
    子节点按 key_fn(key) 作为字典键组织，形成共享前缀的树结构。
    ref_count 表示当前有多少请求正在引用该节点（引用中的节点不可被驱逐）。
    timestamp 用于 LRU 驱逐（最近访问的节点时间戳更新）。
    """

    counter: int = 0  # 全局节点计数器，用于生成唯一 uuid

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}  # 子节点字典
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # 引用计数，>0 表示被保护
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()  # 最近访问时间（LRU 用）

        # 这些字段稍后通过 set_key_value 设置
        self._key: torch.Tensor  # 本节点代表的 token id 序列
        self._value: torch.Tensor  # 对应的物理索引序列
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置节点的 key（token 序列）与 value（物理索引序列）。"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """把本节点挂到 parent 下，并按 key_fn 注册到父节点的 children 字典。"""
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        """本节点覆盖的 token 数。"""
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        """本节点对应的物理索引序列。"""
        return self._value

    def is_root(self) -> bool:
        """是否为根节点。"""
        return self._parent is None

    def is_leaf(self) -> bool:
        """是否为叶子节点（无子节点）。"""
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """计算本节点 key 与 input_ids 的公共前缀长度（调用自定义 CUDA 内核快速比较）。"""
        from minisgl.kernel import fast_compare_key

        # 比较 key 和 input_ids，找到第一个不同的位置
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在 pos 处把本节点拆成两个节点，返回新生成的前半段节点。

        拆分后：新节点保存 [0:pos) 段，本节点变为保存 [pos:) 段。
        用于"部分匹配"场景：请求只匹配了节点的一部分，需要把未匹配部分拆出来。
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 新节点保存前半段，继承本节点的时间戳与引用计数
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        # 本节点改为保存后半段，并挂到新节点下
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        """按时间戳比较，供 heapq 做 LRU 排序（时间戳越小越旧，越先被驱逐）。"""
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """Radix 缓存句柄：指向匹配到的树节点。"""

    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        """从匹配节点沿父链向上收集所有 value，拼接成完整的物理索引序列。"""
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()  # 从根到叶的顺序
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    """Radix 前缀缓存：用 Radix 树实现前缀复用。

    核心思想：把公共前缀在树中只存一份，多个请求共享；
    新请求到来时沿树匹配最长公共前缀，命中部分直接复用缓存中的 KV，
    只需计算未命中的部分。支持按页对齐（align_down）、LRU 驱逐（ref_count 保护）。
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size  # 页大小，前缀按页对齐
        self.key_fn = _get_key_fn(self.page_size)  # 子节点字典的键函数
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0  # 可驱逐的 token 数
        self.protected_size = 0  # 被引用保护的 token 数
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # 根节点始终受保护，不会被驱逐

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁定/解锁句柄：沿父链更新每个节点的引用计数与容量统计。

        锁定（unlock=False）：ref_count 从 0 变 1 时，把该段从可驱逐区移到受保护区。
        解锁（unlock=True）：ref_count 减到 0 时，把该段从受保护区移回可驱逐区。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    # 引用归零，重新变为可驱逐
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    # 首次被引用，从可驱逐区移入受保护区
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """沿树匹配最长公共前缀，返回句柄（不修改缓存）。"""
        node, prefix_len = self._tree_walk(input_ids)
        _log_cache_hit(prefix_len, len(input_ids))
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把新计算出的前缀插入缓存。

        先按页对齐截断，再沿树匹配；若匹配长度不足，则把未匹配部分作为新节点挂到树上。
        """
        insert_len = align_down(len(input_ids), self.page_size)  # 按页对齐
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len，有未匹配部分
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """按 LRU 驱逐可驱逐的叶子节点，释放空间，返回被驱逐的物理索引。

        用最小堆按时间戳排序叶子节点，优先驱逐最久未访问的；
        驱逐叶子后若其父节点变为可驱逐叶子，则继续加入堆。
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()  # 收集可驱逐的叶子
        heapq.heapify(leave_nodes)  # 按时间戳建最小堆
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)  # 弹出最旧的叶子
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]  # 从父节点移除该子节点
            # NOTE: root is always protected, so won't be evicted
            # 若父节点因此变成可驱逐叶子，加入堆继续驱逐
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """DFS 遍历整棵树，收集所有"可驱逐的叶子节点"（ref_count==0 且无子节点）。"""
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """沿树匹配 input_ids，返回 (匹配到的节点, 匹配长度)。

        逐页向下走：每到一个子节点，用 fast_compare_key 计算匹配长度，
        按页对齐后累加；若只部分匹配，则拆分节点并返回。
        同时更新访问节点的时间戳（LRU 用）。
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            # 用当前剩余序列的 key_fn 值查找子节点
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len  # 无匹配子节点，返回当前节点
            node = child_node  # 走到子节点

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])  # 计算公共前缀长度
            match_len = align_down(match_len, self.page_size)  # 按页对齐
            prefix_len += match_len

            # 若未完全匹配，需要拆分节点（把未匹配部分拆出来）
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # 更新被访问节点的时间戳（LRU 最近使用）
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    """根据页大小生成子节点字典的键函数。

    page_size=1 时用单个 token id 作为键；否则用一页的 token 序列元组作为键。
    """
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
