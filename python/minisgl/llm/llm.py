from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    """离线模式专用异常：所有请求都已处理完毕时抛出，用于终止 run_forever。"""

    pass


@dataclass
class RequestStatus:
    """单个请求的状态记录（离线模式用）。"""

    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    """离线推理入口：类似 HuggingFace transformers 的 LLM 接口。

    继承 Scheduler，但以 offline_mode 运行：不启动 ZMQ 消息循环，
    而是通过 offline_receive_msg / offline_send_result 直接与调度器交互，
    在单进程内完成"编码 -> 调度 -> 解码"的完整流程。
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),  # 离线模式固定单卡
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []  # 待处理请求
        self.status_map: Dict[int, RequestStatus] = {}  # uid -> 请求状态
        self.counter = 0  # uid 计数器

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """把单个 prompt 编码为 token 张量（支持字符串或 token 列表）。"""
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式的消息接收：从 pending_requests 中取出请求，构造 UserMsg。

        按 prefill_budget 预算分批取出，避免一次塞入过多请求。
        blocking=True 且无待处理请求时抛出 RequestAllFinished 终止循环。
        """
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break  # 达到预算，停止本批
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            # 记录请求状态，供后续收集输出
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]  # 移除已取出的请求
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式的结果回传：把解码结果追加到对应请求的 output_ids。

        跳过 EOS 结束标记本身（finished 且 next_token == eos 时不记录）。
        """
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        """离线生成：对一批 prompts 执行完整推理，返回文本与 token id。

        流程：清空状态 -> 填充 pending_requests -> 运行调度主循环
        （直到 RequestAllFinished）-> 收集并解码所有输出。
        """
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        # 若只给了一个 SamplingParams，则对所有 prompt 复用
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()  # 运行调度主循环，直到所有请求完成
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
