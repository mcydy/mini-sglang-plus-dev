from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """分布式通信实现抽象基类：定义张量并行所需的集合通信接口。"""

    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """对张量 x 做全归约（求和），返回归约后的张量。"""

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """沿第 0 维对张量 x 做全收集，返回拼接后的张量。"""


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """基于 torch.distributed 的默认实现。"""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x  # 单卡无需通信
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size  # 第 0 维扩大 tp_size 倍
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """基于 PyNCCL（自定义 CUDA 内核）的实现，用于覆盖默认 torch 实现。"""

    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """分布式通信器：插件式设计，默认使用 torch 实现。

    通过 plugins 列表管理多个实现，实际调用时使用列表末尾的插件
    （后追加的插件优先级更高，用于覆盖默认实现）。
    """

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    启用基于 PyNCCL 的分布式通信（用于张量并行）。

    把 PyNCCL 实现追加到插件列表末尾，从而覆盖默认的 torch 实现。
    """
    if tp_info.size == 1:
        return  # 单卡无需启用
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    销毁所有分布式通信插件。
    """
    DistributedCommunicator.plugins = []
