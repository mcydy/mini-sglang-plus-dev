# =============================================================================
# llama.py — Llama 模型实现
#
# 实现 Llama 系列因果语言模型，采用"残差流"设计：
#   - LlamaDecoderLayer : 单层解码器（注意力 + MLP + 两个 RMSNorm）
#   - LlamaModel        : 嵌入 + 多层解码器 + 末尾归一化
#   - LlamaForCausalLM  : 组合模型 + LM Head，输出 logits
#
# 残差流：RMSNormFused 把"残差加法 + 归一化"融合为原地操作，
# forward 返回 (x, residual) 贯穿所有层，省去单独的残差张量。
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """Llama 单层解码器。

    结构：input_layernorm -> self_attn -> post_attention_layernorm -> mlp
    采用残差流：x, residual = norm(x, residual)，返回 (x, residual)。
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
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
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向：残差流方式依次经过归一化、注意力、归一化、MLP。"""
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    """Llama 主干模型：词嵌入 + 多层解码器 + 末尾归一化。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向：嵌入 -> 逐层残差流 -> 末尾归一化，返回隐藏状态。"""
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """Llama 因果语言模型：主干 + LM Head，输出 logits。

    支持 tie_word_embeddings（LM Head 与词嵌入共享权重）。
    """

    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """前向：从全局上下文取当前 batch 的 input_ids，输出 logits。"""
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
