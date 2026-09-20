# =============================================================================
# embedding.py — 词嵌入与 LM Head
#
# 实现词表并行（vocab parallel）的嵌入层与 LM Head：
#   - VocabParallelEmbedding : 词表并行嵌入，按 TP 切分词表
#   - ParallelLMHead         : 并行 LM Head，支持 tie_word_embeddings
#
# 词表并行：把词表按 TP size 切分，每卡持有部分词表。
# 嵌入时用自定义 kernel（minisgl.kernel.indexing）查表，TP>1 时 all_reduce；
# LM Head 计算 logits 时 TP>1 用 all_gather 拼接各卡词表分片。
# =============================================================================
from __future__ import annotations
from minisgl.utils import div_even, init_logger

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


logger = init_logger(__name__)

class VocabParallelEmbedding(BaseOP):
    """词表并行嵌入层：按 TP 切分词表，每卡持有部分词表。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        # 本卡负责的词表范围 (起始索引, 长度)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        # 申请的这个显存存词向量表，初始化就申请
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：查表得到嵌入向量；TP>1 时 all_reduce 汇总各卡分片。"""
        from minisgl.kernel import indexing

        logger.debug(f"嵌入查找几个token: {x.shape}")
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )
        # y是形状[num_tokens, hidden] 的每张卡的完整嵌入矩阵
        logger.debug(f"执行all_reduce向量加，每个查找到向量矩阵y形状 {y.shape}")
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """并行 LM Head：把隐藏状态映射为词表 logits。

    支持 tie_word_embeddings（与词嵌入共享权重，加载/导出时跳过自身权重）。
    prefill 阶段只取每个序列最后一个 token 计算 logits（节省计算）。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            # 权重共享时跳过 lm_head 自身的权重（与 embed_tokens 共享）
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：计算 logits。

        prefill 阶段用 attn_metadata.get_last_indices(bs) 只取每个序列最后一个 token；
        TP>1 时 all_gather 拼接各卡词表分片，再截取有效词表范围。
        """
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            # prefill 阶段只需每个序列最后一个 token 的 logits
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        
        # x @ W.T + b
        logits = F.linear(x, module.weight, self.bias)

        # [4, 151936] 151936每个词的概率
        logger.debug(f"LM head 计算的logits {logits.shape}")

        if self.tp_size == 1:
            return logits
        # 如果多卡，进行拼接
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # 把各卡词表分片拼接回完整词表顺序
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
