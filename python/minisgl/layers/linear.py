# =============================================================================
# linear.py — 张量并行线性层
#
# 实现支持张量并行(TP)的各类线性层：
#   - LinearReplicated      : 权重全量复制（每卡持有完整矩阵，用于 MoE router gate）
#   - LinearColParallelMerged: 列并行合并（多个输出维度各自按 TP 切分后拼接，用于 gate_up_proj）
#   - LinearQKVMerged       : QKV 合并投影（Q/KV 头按 TP 切分，用于 qkv_proj）
#   - LinearOProj           : 输出投影（行并行，forward 后 all_reduce，用于 o_proj）
#   - LinearRowParallel     : 行并行（输入按 TP 切分，输出后 all_reduce，用于 down_proj）
#
# 张量并行约定：
#   - 列并行（qkv/gate_up）：切分输出维度，各卡计算不同输出分片
#   - 行并行（o/down）：切分输入维度，各卡计算部分和，再 all_reduce 汇总
# =============================================================================
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism.

    张量并行线性层的实现基类：记录 full/local 输入输出维度，
    weight 形状为 (local_osize, local_isize)，forward 用 F.linear。
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.

    权重全量复制的线性层：每张卡持有完整权重矩阵（不切分）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """列并行合并线性层：多个输出维度各自按 TP 切分后拼接。

    用于 gate_up_proj（输出 [intermediate, intermediate] 两个维度）。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """QKV 合并投影：Q 头按 TP 切分、KV 头按 TP 切分。

    allow_replicate=True 表示 KV 头数小于 TP 数时允许复制（每卡持有完整 KV 头）。
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """输出投影（行并行）：输入按 TP 切分，输出后 all_reduce 汇总。

    用于 o_proj（注意力输出投影）。
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """行并行线性层：输入按 TP 切分，输出后 all_reduce 汇总。

    用于 down_proj（MLP 输出投影）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
