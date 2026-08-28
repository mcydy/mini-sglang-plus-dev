# =============================================================================
# base.py — 模型基类
#
# 定义所有因果语言模型（Causal LM）的抽象基类 BaseLLMModel。
# 它同时继承 BaseOP（获得递归的 state_dict / load_state_dict 能力），
# 只声明抽象方法 forward()，具体实现由各模型（Llama/Qwen3 等）提供。
# =============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """所有因果语言模型的抽象基类。

    继承 BaseOP 以获得统一的权重存取（state_dict / load_state_dict）能力。
    子类需实现 forward()，返回 logits 张量。
    """

    @abstractmethod
    def forward(self) -> torch.Tensor: ...
