from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """注意力元数据基类：描述一次注意力计算所需的辅助信息（长度、页表等）。"""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个序列最后一个 token 的位置索引（decode 阶段取 logits 用）。"""


class BaseAttnBackend(ABC):
    """注意力后端抽象基类：定义可插拔的注意力计算接口。

    具体实现（FlashAttentionBackend / FlashInferBackend）负责：
    - forward：执行一次注意力计算（含 KV 写入）
    - prepare_metadata：根据 batch 生成注意力元数据
    - CUDA Graph 捕获/回放相关接口
    """

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行注意力前向计算，返回注意力输出。"""

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """根据 batch 计算并填充 batch.attn_metadata。"""

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的缓冲。"""

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """在 CUDA Graph 捕获前准备 batch 的元数据。"""

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """在 CUDA Graph 回放前把真实元数据拷贝到捕获缓冲。"""


class HybridBackend(BaseAttnBackend):
    """混合注意力后端：按 prefill/decode 阶段自动切换底层实现。

    prefill 阶段（长序列、计算密集）与 decode 阶段（batch 大、访存密集）
    的最优实现往往不同，因此把两者组合起来，按 batch.is_prefill 分发。
    """

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # prefill 用 prefill 后端，decode 用 decode 后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # CUDA Graph 只用于 decode 阶段
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
