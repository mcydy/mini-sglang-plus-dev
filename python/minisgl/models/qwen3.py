# =============================================================================
# qwen3.py — Qwen3 模型实现
#
# 实现 Qwen3 系列因果语言模型。结构与 Llama 几乎一致，唯一区别是
# Qwen3 的注意力带 QK 归一化（has_qk_norm=True），即对 Q 和 K 分别做 RMSNorm。
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
from minisgl.utils import div_even, init_logger

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig

logger = init_logger(__name__)


class Qwen3DecoderLayer(BaseOP):
    """Qwen3 单层解码器。

    与 Llama 的区别：self_attn 使用带 QK 归一化的 Qwen3Attn。
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向：残差流方式依次经过归一化、注意力、归一化、MLP。"""
        logger.debug(f"attention前归一化")
        x, residual = self.input_layernorm.forward(x, residual)
        logger.debug(f"开始计算attention")
        x = self.self_attn.forward(x)
        logger.debug(f"开始残差连接")
        x, residual = self.post_attention_layernorm.forward(x, residual)
        logger.debug(f"开始MLP处理")
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    """Qwen3 主干模型：词嵌入 + 多层解码器 + 末尾归一化。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向：嵌入 -> 逐层残差流 -> 末尾归一化，返回隐藏状态。"""
        logger.debug(f"模型开始查找词嵌表")
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        logger.debug(f"每层处理 进行最终归一化")
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    """Qwen3 因果语言模型：主干 + LM Head，输出 logits。"""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """前向：从全局上下文取当前 batch 的 input_ids，输出 logits。"""
        logger.debug(f"模型开始前向---")
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logger.debug(f"DecoderLayer结束，开始LM head ")
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
