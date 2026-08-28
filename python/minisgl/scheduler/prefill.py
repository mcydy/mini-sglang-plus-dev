# =============================================================================
# prefill.py — PrefillManager（预填充管理器）
#
# 负责 prefill 阶段：接收新请求、分块（chunked prefill）、生成 prefill 批次。
#
# 核心概念：
#   - ChunkedReq : 超长请求被分块后的"块请求"，禁止采样，剩余部分留在
#                  pending_list 中等待下一轮继续处理（chunked prefill）
#   - PrefillAdder: 尝试为一个请求分配资源（页表槽位 + KV cache 页），
#                  受 token 预算与在飞 decode 预留空间约束
#   - PrefillManager: 维护待处理请求列表，按预算逐个加入批次
#
# Chunked Prefill 的意义：把长 prompt 拆成小块处理，降低峰值显存占用，
# 避免长上下文服务时的 OOM（源自 Sarathi-Serve 论文）。
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """分块请求：超长请求被切分后的一块。

    与普通 Req 的区别：
        - append_host 抛错：分块请求不允许被采样（它只是 prefill 的一部分）
        - can_decode 恒为 False：避免被加入 decode 管理器
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """预填充添加器：尝试把待处理请求加入当前 prefill 批次。

    属性：
        token_budget: 本批次剩余的 token 预算（prefill 最多处理的 token 数）
        reserved_size: 为在飞 decode 请求预留的 token 数
        cache_manager: KV cache 管理器（用于前缀匹配与页分配）
        table_manager: 页表管理器（用于分配请求槽位）
    """

    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """尝试为一个请求分配资源（页表槽位 + KV cache 页）。

        若资源不足返回 None；成功则返回 (cache_handle, table_idx)。
        命中前缀缓存时，会把缓存部分拷入 token_pool 和 page_table。
        """
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """把一个待处理请求转换为 Req（或 ChunkedReq），并扣减 token 预算。

        分块逻辑：chunk_size = min(token_budget, remain_len)，
        若 chunk_size < remain_len 则创建 ChunkedReq，剩余部分下次继续。
        """
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试把一个待处理请求加入批次；成功返回 Req，失败（预算/资源不足）返回 None。"""
        if self.token_budget <= 0:
            return None

        # 若该请求之前已被分块，则继续处理剩余部分（复用已分配的槽位与缓存句柄）
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """预填充管理器：维护待处理请求列表，按预算调度 prefill 批次。

    属性：
        cache_manager: KV cache 管理器
        table_manager: 页表管理器
        decode_manager: decode 管理器（用于估算在飞 decode 占用的 token）
        pending_list: 待处理请求列表
    """

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """把一条用户消息加入待处理列表。"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """调度下一批 prefill：在预算内逐个加入请求，超预算即停止。"""
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """取消一个待处理请求，返回其已分配的 Req（若有）以便释放资源。"""
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """是否有待处理的请求。"""
        return len(self.pending_list) > 0
