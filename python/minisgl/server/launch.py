from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """子进程入口：运行一个 TP rank 的 Scheduler 主循环。

    每个 TP rank 一个进程，进程内创建 Scheduler 并运行 run_forever()。
    只有主 rank（rank 0）会向 ack_queue 发送就绪信号。
    """
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()  # 等待所有 TP rank 就绪

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")  # 主 rank 报告就绪

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()  # 进入调度主循环
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    """启动整个服务：解析参数、spawn 所有工作进程、运行 API 服务器。

    进程拓扑：
    - world_size 个 TP scheduler 进程（每个 TP rank 一个）
    - 1 个 detokenizer 进程
    - num_tokenizer 个 tokenizer 进程
    通过 ack_queue 保证所有子进程就绪后才开始对外服务。
    """
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # Windows 上必须用 spawn 启动方式
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # 用于接收子进程就绪信号的 multiprocessing 队列，
        # 保证所有子进程都准备好后才开始服务
        ack_queue: mp.Queue[str] = mp.Queue()

        # 启动 world_size 个 TP scheduler 进程
        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),  # 每个进程分配自己的 TP rank
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        # 启动 1 个 detokenizer 进程（负责把 token 解码为文本）
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # 启动 num_tokenizer 个 tokenizer 进程（负责把文本编码为 token）
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # 等待所有工作进程的就绪信号：
        # - world_size 个 scheduler（但只有主 rank 发送 ack）
        # - num_tokenizers 个 tokenizer
        # - 1 个 detokenizer
        # 总共期望的 ack 数：1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
