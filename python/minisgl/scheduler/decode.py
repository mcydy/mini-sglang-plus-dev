# =============================================================================
# decode.py — DecodeManager（解码管理器）
#
# 负责 decode 阶段：管理正在运行（逐 token 生成）的请求集合。
#
# 核心职责：
#   - filter_reqs: 每步前向后过滤掉已完成的请求（can_decode=False）
#   - schedule_next_batch: 生成 decode 批次（按 uid 排序保证确定性）
#   - inflight_tokens: 估算在飞 decode 占用的 token 数，供 prefill 预算参考
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """解码管理器：维护 decode 阶段的运行中请求集合。

    属性：
        page_size: 分页大小（用于估算在飞 token 的预留页）
        running_reqs: 正在 decode 的请求集合
    """

    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """过滤运行中请求：只保留还能继续 decode 的请求（can_decode=True）。"""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """从运行集合中移除一个请求（请求已完成）。"""
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """按 uid 取消一个运行中的请求，返回该请求以便释放资源。"""
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """估算在飞 decode 占用的 token 数。

        每个请求：剩余可生成 token 数 + 预留 1 页（page_size 个 token）。
        该值用于 prefill 调度时预留空间，避免抢占 decode 的 KV cache。
        """
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """生成 decode 批次：按 uid 排序保证各 rank 调度一致。"""
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """是否有正在 decode 的请求。"""
        return len(self.running_reqs) > 0
