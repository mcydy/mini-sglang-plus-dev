from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import torch
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | float | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _temperature_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """按行做温度缩放的 softmax（fp32 计算，等价于 flashinfer 的 sampling.softmax）。"""
    return torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)


def _filter_top_k(probs: torch.Tensor, top_k: Union[torch.Tensor, int]) -> torch.Tensor:
    """按行保留概率最高的 top_k 个 token，其余置 0；top_k 支持标量或 [batch] 张量。"""
    V = probs.shape[-1]
    k = torch.as_tensor(top_k, device=probs.device).to(torch.int64)

    if k.dim() == 0:
        k_scalar = int(k.item())
        if k_scalar >= V:  # 不裁剪
            return probs
        threshold = torch.topk(probs, k_scalar, dim=-1).values[..., -1:]
    else:
        k = k.clamp_(min=1, max=V)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        threshold = sorted_probs.gather(1, (k - 1).unsqueeze(1))

    return probs.masked_fill(probs < threshold, 0.0)


def _filter_top_p(probs: torch.Tensor, top_p: Union[torch.Tensor, float]) -> torch.Tensor:
    """按行做 nucleus（top-p）裁剪：保留累积概率跨过 p 的那个 token，其余置 0。"""
    p = top_p
    if torch.is_tensor(p):
        p = p.to(probs.dtype).unsqueeze(1)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # (cumsum - p_i) 是该 token 之前的累积概率，< top_p 表示应当保留
    keep_sorted = (cumulative - sorted_probs) < p
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(1, sorted_indices, keep_sorted)
    return probs.masked_fill(~keep, 0.0)


def _renormalize(probs: torch.Tensor) -> torch.Tensor:
    """重新归一化；整行被裁剪为空时退化为取 argmax（避免 multinomial 报错）。"""
    total = probs.sum(dim=-1, keepdim=True)
    fallback = torch.zeros_like(probs)
    fallback.scatter_(1, probs.argmax(dim=-1, keepdim=True), 1.0)
    return torch.where(total > 0, probs / total.clamp_min(torch.finfo(probs.dtype).tiny), fallback)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """纯 PyTorch 的 top-k / top-p 采样，替代 flashinfer.sampling。

    Args:
        logits:       [batch, vocab] float32
        temperatures: [batch] float32
        top_k:        [batch] int32/int64 或标量；None 表示不裁剪
        top_p:        [batch] float32 或标量；None 表示不裁剪

    Returns:
        [batch] int32 的采样结果
    """
    probs = _temperature_softmax(logits, temperatures)

    if top_k is not None:
        probs = _filter_top_k(probs, top_k)
    if top_p is not None:
        probs = _filter_top_p(probs, top_p)
    if top_k is not None or top_p is not None:
        probs = _renormalize(probs)

    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int32)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
