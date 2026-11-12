#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_scheduler.py — mini-sglang 调度特性测试工具

纯标准库实现（无第三方依赖），通过 HTTP + SSE 从客户端侧测量调度行为。

可观测的调度特性：
  1) prefix  —— RadixAttention 前缀缓存是否生效（同一 prompt 二次请求的 TTFT 下降）
  2) chunked —— chunked prefill 是否生效（长 prompt 被 --max-prefill-length 切块）
  3) interfere —— prefill 对 decode 的干扰（正是 PD 分离要解决的问题）
  4) concur  —— 不同并发度下的吞吐 / TTFT / TPOT 曲线
  5) graph   —— CUDA graph 生效边界（batch size 超过 --cuda-graph-max-bs 时的延迟跳变）
  6) capacity —— KV cache 容量上限与并发排队行为

基本用法：
    python bench_scheduler.py --model Qwen3-4B-Instruct-2507 prefix
    python bench_scheduler.py --model Qwen3-4B-Instruct-2507 concur --concurrency 1 4 16 64
    python bench_scheduler.py --model Qwen3-4B-Instruct-2507 interfere

注意事项：
  * 服务端返回的 usage 字段恒为 0（mini-sglang 未实现计数），
    因此 TTFT / TPOT / token 数全部由本脚本在客户端侧测量：
      TTFT = 发出请求 → 收到第一个非空 content 的时间
      TPOT = 相邻 token 到达时间的间隔
      token 数 = 收到的非空 content chunk 数（SSE 每步一个，近似等于生成 token 数）
  * 建议先用一次 warmup 请求消除 CUDA graph capture 之外的首次开销。
  * 对比实验需重启服务并修改启动参数（脚本会给出提示）。
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 单次请求的测量结果
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """一次流式请求的测量结果。"""

    ok: bool = True
    error: str = ""

    ttft: float = 0.0            # 首 token 延迟（秒）
    total: float = 0.0           # 端到端总延迟（秒）
    n_tokens: int = 0            # 收到的 token 数
    inter_token: List[float] = field(default_factory=list)  # 相邻 token 间隔
    token_times: List[float] = field(default_factory=list)  # 每个 token 相对 t0 的到达时刻
    text: str = ""

    # 调用方附加的标签，便于汇总
    tag: str = ""
    start_offset: float = 0.0    # 相对本批次起始的发起时刻（用于干扰分析）
    raw_samples: List[str] = field(default_factory=list)  # --verbose 下保留原始 SSE 行

    @property
    def tpot_avg(self) -> float:
        """平均 TPOT（总时长扣除 TTFT 后按 token 均摊）。"""
        if self.n_tokens <= 1:
            return 0.0
        return (self.total - self.ttft) / (self.n_tokens - 1)

    @property
    def tpot_p50(self) -> float:
        return _pct(self.inter_token, 50)

    @property
    def tpot_p99(self) -> float:
        return _pct(self.inter_token, 99)

    @property
    def tpot_max(self) -> float:
        return max(self.inter_token) if self.inter_token else 0.0

    @property
    def tokens_per_sec(self) -> float:
        dur = self.total - self.ttft
        return (self.n_tokens - 1) / dur if dur > 0 and self.n_tokens > 1 else 0.0


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


# ---------------------------------------------------------------------------
# 请求发送
# ---------------------------------------------------------------------------


class Client:
    # 默认 60s 而不是更长：scheduler 进程崩溃后，HTTP 连接不会关闭，
    # 客户端会一直阻塞在 readline() 上。宁可快速失败并给出可诊断的提示。
    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        timeout: float = 60.0,
        verbose: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        self._local = threading.local()

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            conn.connect()
            self._local.conn = conn
        return conn

    def stream(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> Result:
        """发起一次流式 chat completion，测量 TTFT / TPOT。

        使用 temperature=0（greedy）保证结果可复现，同时避免采样耗时干扰。
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        r = Result()
        t0 = time.perf_counter()
        try:
            conn = self._conn()
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                r.ok = False
                r.error = f"HTTP {resp.status}: {resp.read()[:300]!r}"
                return r
            self._consume_sse(resp, r, t0)
        except Exception as e:  # noqa: BLE001
            r.ok = False
            r.error = f"{type(e).__name__}: {e}"
            if "timed out" in str(e).lower() or isinstance(e, TimeoutError):
                r.error += f"  (超过 {self.timeout}s 未收到数据 —— 服务端可能已崩溃或未调度本请求)"
            # 连接可能已坏，下次重建
            self._local.conn = None
        if r.ok and r.n_tokens == 0:
            r.error = "未收到任何 content chunk（流已结束但无内容）"
        if self.verbose and r.raw_samples:
            print("    [verbose] 原始 SSE 前几行:")
            for line in r.raw_samples:
                print(f"      {line[:160]}")
        return r

    def complete(self, prompt: str, max_tokens: int, timeout: float = 10.0) -> Dict[str, Any]:
        """非流式请求，用于快速探测服务端是否健康（不测量时序）。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        body = json.dumps(payload).encode()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read()
            return {
                "status": resp.status,
                "ok": resp.status == 200,
                "body": raw.decode("utf-8", errors="replace"),
            }
        except Exception as e:  # noqa: BLE001
            return {"status": -1, "ok": False, "body": f"{type(e).__name__}: {e}"}
        finally:
            conn.close()

    @staticmethod
    def _consume_sse(resp: Any, r: Result, t0: float) -> None:
        """逐行解析 SSE。chat 接口格式为 'data: {json}\\n\\n'。"""
        last = t0
        chunks: List[str] = []
        finished = False
        while True:
            line = resp.readline()
            if not line:
                break
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            if len(r.raw_samples) < 5:
                r.raw_samples.append(line)
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            # 注意：收到 finish_reason 后不能 break —— 服务端还会接着发 end_chunk 和
            # [DONE]。若提前退出，残留字节会留在 socket 里，使下一次复用该连接时
            # 抛出 ResponseNotReady。必须读到 [DONE] 或 EOF 才算完整消费。
            if finished:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            delta = ch.get("delta") or {}
            content = delta.get("content")
            now = time.perf_counter()
            if content:                       # 只统计非空内容，跳过首个 role chunk
                if r.n_tokens == 0:
                    r.ttft = now - t0
                else:
                    r.inter_token.append(now - last)
                r.token_times.append(now - t0)
                r.n_tokens += 1
                chunks.append(content)
                last = now
            if ch.get("finish_reason"):
                finished = True
        r.total = time.perf_counter() - t0
        r.text = "".join(chunks)
        # drain 剩余字节（含 chunked 的终止块 "0\r\n\r\n"），让 HTTPResponse 进入
        # complete 状态，否则该连接无法复用，下次请求会抛 ResponseNotReady。
        try:
            resp.read()
        except Exception:  # noqa: BLE001, S110
            pass


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def build_prompt(target_tokens: int, seed: str = "") -> str:
    """构造一个大致 target_tokens 个 token 的 prompt。

    中文按 ~1.2 token/字 估算不够准，这里用可重复的长句拼接，
    实际长度以服务端为准；仅用于控制量级（短 / 中 / 长）。
    """
    base = (
        "以下是关于大规模语言模型推理系统调度的一段说明性文字，"
        "用于构造具有指定长度的测试输入。内容包括连续批处理、分页注意力、"
        "前缀缓存、分块预填充以及预填充与解码分离等技术要点。"
    )
    if seed:
        base = seed + base
    repeat = max(1, target_tokens // 60)
    return base * repeat


def run_concurrent(
    client: Client,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float = 0.0,
) -> List[Result]:
    """并发送出所有请求，返回结果列表（顺序与 prompts 一致）。"""
    t_batch = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [
            ex.submit(client.stream, p, max_tokens, temperature) for p in prompts
        ]
        results = [f.result() for f in futs]
    for r in results:
        r.start_offset = 0.0
    return results


def summarize(results: Iterable[Result], label: str) -> Dict[str, float]:
    rs = [r for r in results if r.ok]
    if not rs:
        return {}
    ttft = [r.ttft for r in rs]
    tpot = [r.tpot_avg for r in rs]
    return {
        "label": label,
        "n": len(rs),
        "ttft_p50": _pct(ttft, 50),
        "ttft_p99": _pct(ttft, 99),
        "ttft_max": max(ttft),
        "tpot_p50": _pct(tpot, 50),
        "tpot_p99": _pct(tpot, 99),
        "tpot_max": max(tpot),
        "total_p50": _pct([r.total for r in rs], 50),
        "tok_s": sum(r.tokens_per_sec for r in rs),
    }


def print_table(rows: List[Dict[str, Any]], title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    if not rows:
        print("  (无数据)")
        return
    cols = [
        ("label", "场景", 22),
        ("n", "N", 5),
        ("ttft_p50", "TTFT p50", 10),
        ("ttft_p99", "TTFT p99", 10),
        ("tpot_p50", "TPOT p50", 10),
        ("tpot_p99", "TPOT p99", 10),
        ("tpot_max", "TPOT max", 10),
        ("tok_s", "tok/s", 10),
    ]
    print("".join(f"{h:>{w}}" for _, h, w in cols))
    print("-" * 96)
    for row in rows:
        line = ""
        for key, _, w in cols:
            v = row.get(key)
            if v is None or v == "":
                v = "-"
            elif isinstance(v, float):
                v = f"{v * 1000:.1f}ms" if key.startswith(("ttft", "tpot")) else f"{v:.1f}"
            line += f"{str(v):>{w}}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# 测试场景
# ---------------------------------------------------------------------------


def test_prefix(client: Client, args: argparse.Namespace) -> None:
    """前缀缓存：同一长 prompt 连续多次请求，观察 TTFT 是否逐次下降。"""
    prompt = build_prompt(args.long_prompt, seed="[前缀缓存测试]")
    print(f"prompt 长度 ≈ {len(prompt)} 字符，重复请求 {args.repeat} 次")

    rows = []
    for i in range(args.repeat):
        r = client.stream(prompt, args.max_tokens)
        if not r.ok:
            print(f"  第 {i+1} 次失败: {r.error}")
            continue
        rows.append(
            {
                "label": f"第{i+1}次",
                "n": 1,
                "ttft_p50": r.ttft,
                "ttft_p99": r.ttft,
                "tpot_p50": r.tpot_avg,
                "tpot_p99": r.tpot_avg,
                "tpot_max": r.tpot_max,
                "tok_s": r.tokens_per_sec,
            }
        )
        print(f"  第 {i+1:>2} 次: TTFT={r.ttft*1000:7.1f}ms  TPOT={r.tpot_avg*1000:6.2f}ms")

    print_table(rows, "场景 1：RadixAttention 前缀缓存")

    if len(rows) >= 2:
        first, later = rows[0]["ttft_p50"], statistics.median([r["ttft_p50"] for r in rows[1:]])
        if later < first * 0.8:
            print(f"  ✅ 前缀缓存生效：首次 {first*1000:.1f}ms → 后续中位 {later*1000:.1f}ms "
                  f"（降低 {100*(1-later/first):.0f}%）")
        else:
            print(f"  ⚠️  未见明显缓存收益：首次 {first*1000:.1f}ms，后续中位 {later*1000:.1f}ms")
            print("     可能原因：--cache-type naive / prompt 太短 / 每次请求被 evict / 服务端未启用 radix")
    print("  💡 对比实验：重启服务加 --cache-type naive，同样的测试应看不到 TTFT 下降")


def test_chunked(client: Client, args: argparse.Namespace) -> None:
    """Chunked prefill：长 prompt 是否被切成多块，从而不长时间独占 GPU。"""
    print("原理：长 prompt 若不分块，会在一次 forward 中独占调度预算，")
    print("      导致同批其他请求的 TPOT 出现长尖峰；分块后尖峰被摊薄。\n")

    for length in args.chunked_lengths:
        prompt = build_prompt(length, seed=f"[长度{length}]")
        # 单独一个长请求，测它自己的 TTFT
        single = client.stream(prompt, 8)
        # 长请求 + 若干短 decode 请求混发，看短请求被拖慢多少
        mixed_prompts = [prompt] + [build_prompt(8) for _ in range(args.chunked_siblings)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(mixed_prompts)) as ex:
            futs = [ex.submit(client.stream, p, args.chunked_decode_tokens)
                    for p in mixed_prompts]
            results = [f.result() for f in futs]
        wall = time.perf_counter() - t0

        long_r, sib = results[0], results[1:]
        ok_sib = [r for r in sib if r.ok]
        if not ok_sib:
            print(f"  长度 {length}: 兄弟请求全部失败，跳过")
            continue
        p99 = _pct([r.tpot_p99 for r in ok_sib], 50)
        mx = max(r.tpot_max for r in ok_sib)
        print(f"  prompt≈{length:>5}: 长请求 TTFT={single.ttft*1000:7.1f}ms | "
              f"兄弟请求 TPOT p99={p99*1000:6.2f}ms max={mx*1000:7.2f}ms | 批次耗时={wall:.2f}s")

    print()
    print("  💡 对比实验：分别用 --max-prefill-length 512 和 8192 重启服务，")
    print("     小 chunk 下兄弟请求的 TPOT max 应明显更小（尖峰被切碎），")
    print("     但长请求自身 TTFT 会略增（分块有额外开销）。这就是 chunked prefill 的取舍。")


def test_interference(client: Client, args: argparse.Namespace) -> None:
    """prefill 干扰 decode —— PD 分离要解决的核心问题。"""
    n = args.decode_count
    short = build_prompt(16)

    print(f"步骤 1：{n} 个短 decode 请求并发（max_tokens={args.chunked_decode_tokens}），测量基线 TPOT")
    base = run_concurrent(client, [short] * n, args.chunked_decode_tokens)

    print(f"步骤 2：同样 {n} 个 decode 请求，稳定后插入一个长 prefill 请求")
    long_prompt = build_prompt(args.long_prompt, seed="[干扰源]")
    t_batch = time.perf_counter()

    def long_job() -> Result:
        time.sleep(args.inject_delay)      # 等 decode 进入稳定期
        return client.stream(long_prompt, 8)

    with ThreadPoolExecutor(max_workers=n + 1) as ex:
        futs = [ex.submit(client.stream, short, args.chunked_decode_tokens) for _ in range(n)]
        f_long = ex.submit(long_job)
        results = [f.result() for f in futs]
        r_long = f_long.result()

    wall = time.perf_counter() - t_batch
    ok_b = [r for r in base if r.ok]
    ok_d = [r for r in results if r.ok]

    def agg(rs: List[Result]) -> Dict[str, float]:
        return {
            "tpot_p50": _pct([r.tpot_p50 for r in rs], 50),
            "tpot_p99": _pct([r.tpot_p99 for r in rs], 50),
            "tpot_max": max(r.tpot_max for r in rs),
            "ttft_p50": _pct([r.ttft for r in rs], 50),
        }

    b, d = agg(ok_b), agg(ok_d)
    rows = [
        {"label": "基线(无 prefill 插入)", "n": len(ok_b), "tpot_p50": b["tpot_p50"],
         "tpot_p99": b["tpot_p99"], "tpot_max": b["tpot_max"], "ttft_p50": b["ttft_p50"]},
        {"label": "有 prefill 插入", "n": len(ok_d), "tpot_p50": d["tpot_p50"],
         "tpot_p99": d["tpot_p99"], "tpot_max": d["tpot_max"], "ttft_p50": d["ttft_p50"]},
    ]
    print_table(rows, "场景 3：prefill 对 decode 的干扰（PD 分离的动机）")

    if r_long.ok:
        print(f"  插入的长 prefill 请求：TTFT={r_long.ttft*1000:.1f}ms，总耗时={r_long.total:.2f}s")
    print(f"  批次墙钟：{wall:.2f}s")

    if b["tpot_max"] > 0:
        ratio = d["tpot_max"] / b["tpot_max"]
        print(f"  TPOT max 放大倍数：{ratio:.2f}x  "
              f"({b['tpot_max']*1000:.2f}ms → {d['tpot_max']*1000:.2f}ms)")
        if ratio > 2:
            print("  ⚠️  干扰显著：一个 prefill 就让 decode 的 token 间隔抖动数倍。")
            print("     这正是 PD 分离（--disaggregation-mode prefill/decode）要消除的现象。")
        else:
            print("  ✅ 干扰不明显，可能因为 prefill 被 chunked 切碎了，或 GPU 余量充足。")
    print()
    print("  💡 想放大这个现象：把 --max-prefill-length 调大（如 16384）让 prefill 不分块，")
    print("     或把长 prompt 加长、decode 并发数加大，再跑一次对比。")


def test_concurrency(client: Client, args: argparse.Namespace) -> None:
    """不同并发度下的吞吐与延迟曲线。"""
    prompt = build_prompt(args.prompt_len)
    rows = []
    for c in args.concurrency:
        prompts = [prompt] * c
        t0 = time.perf_counter()
        results = run_concurrent(client, prompts, args.max_tokens)
        wall = time.perf_counter() - t0
        ok = [r for r in results if r.ok]
        if not ok:
            print(f"  并发 {c}: 全部失败")
            continue
        total_tokens = sum(r.n_tokens for r in ok)
        rows.append(
            {
                "label": f"并发={c}",
                "n": len(ok),
                "ttft_p50": _pct([r.ttft for r in ok], 50),
                "ttft_p99": _pct([r.ttft for r in ok], 99),
                "tpot_p50": _pct([r.tpot_avg for r in ok], 50),
                "tpot_p99": _pct([r.tpot_avg for r in ok], 99),
                "tpot_max": max(r.tpot_avg for r in ok),
                "tok_s": total_tokens / wall,
            }
        )
        print(f"  并发 {c:>3}: 墙钟={wall:6.2f}s  聚合吞吐={total_tokens/wall:7.1f} tok/s  "
              f"TTFT p50={_pct([r.ttft for r in ok],50)*1000:7.1f}ms")
    print_table(rows, "场景 4：并发度 vs 吞吐 / 延迟")


def test_graph(client: Client, args: argparse.Namespace) -> None:
    """CUDA graph 生效边界：batch size 超过 --cuda-graph-max-bs 后延迟应跳变。"""
    print("原理：decode 阶段 batch size ≤ --cuda-graph-max-bs 时走 CUDA graph replay，")
    print("      超过后回退到逐 kernel 发射，单步延迟会明显上升。\n")
    prompt = build_prompt(16)
    rows = []
    for c in args.graph_bs:
        results = run_concurrent(client, [prompt] * c, args.max_tokens)
        ok = [r for r in results if r.ok]
        if not ok:
            continue
        rows.append(
            {
                "label": f"bs={c}",
                "n": len(ok),
                "tpot_p50": _pct([r.tpot_avg for r in ok], 50),
                "tpot_p99": _pct([r.tpot_avg for r in ok], 99),
                "tpot_max": max(r.tpot_avg for r in ok),
                "ttft_p50": _pct([r.ttft for r in ok], 50),
                "tok_s": sum(r.tokens_per_sec for r in ok),
            }
        )
        print(f"  bs={c:>3}: TPOT p50={_pct([r.tpot_avg for r in ok],50)*1000:6.2f}ms  "
              f"聚合={sum(r.tokens_per_sec for r in ok):7.1f} tok/s")
    print_table(rows, "场景 5：CUDA graph 生效边界")
    print("  💡 对比实验：--cuda-graph-max-bs 1（关闭）与默认值各跑一次，")
    print("     小 batch 下单步延迟差异最明显。")


def test_capacity(client: Client, args: argparse.Namespace) -> None:
    """KV cache 容量：并发数逼近上限时的排队与失败行为。"""
    print("原理：可用页数 = f(memory_ratio, num_pages, page_size)。")
    print("      并发请求所需页数超过上限时，新请求会在 pending_list 排队（准入控制），")
    print("      表现为 TTFT 随并发数超线性增长，而不是直接报错。\n")
    prompt = build_prompt(args.prompt_len)
    rows = []
    for c in args.capacity_bs:
        t0 = time.perf_counter()
        results = run_concurrent(client, [prompt] * c, args.capacity_tokens)
        wall = time.perf_counter() - t0
        ok = [r for r in results if r.ok]
        fail = len(results) - len(ok)
        if not ok:
            print(f"  并发 {c}: 全部失败")
            continue
        rows.append(
            {
                "label": f"并发={c}",
                "n": len(ok),
                "ttft_p50": _pct([r.ttft for r in ok], 50),
                "ttft_p99": _pct([r.ttft for r in ok], 99),
                "tpot_p50": _pct([r.tpot_avg for r in ok], 50),
                "tpot_max": max(r.tpot_max for r in ok),
                "tok_s": sum(r.tokens_per_sec for r in ok),
            }
        )
        print(f"  并发 {c:>3}: 成功={len(ok):>3} 失败={fail:>3} 墙钟={wall:6.2f}s "
              f"TTFT p50={_pct([r.ttft for r in ok],50)*1000:7.1f}ms")
    print_table(rows, "场景 6：KV cache 容量与准入控制")
    print("  💡 对比实验：--page-size 1 vs 64，同样并发下内存碎片不同，能承载的并发数也不同。")


def test_diag(client: Client, args: argparse.Namespace) -> None:
    """快速诊断：确认服务端健康、SSE 格式符合预期。遇到问题先跑这个。

    按顺序做三件事，任何一步失败都能直接定位问题层次：
      1) HTTP 层：/v1/models 是否有响应
      2) 推理层：非流式请求能否在限定时间内返回
      3) 流式层：SSE 每行的格式是否能被本脚本解析
    """
    print("步骤 1/3  HTTP 连通性：GET /v1/models")
    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        print(f"         HTTP {resp.status}  {body[:200]}")
        if resp.status != 200:
            print("         ❌ 服务未就绪")
            return
    except Exception as e:  # noqa: BLE001
        print(f"         ❌ {type(e).__name__}: {e}")
        print("            服务未启动，或端口不对。启动命令：")
        print("            python -m minisgl --model <path> --attn fa")
        return
    print("         ✅ 服务在监听")

    print("\n步骤 2/3  推理健康：非流式请求（超时 20s）")
    t0 = time.perf_counter()
    res = client.complete("你好", 8, timeout=20.0)
    dt = time.perf_counter() - t0
    if not res["ok"]:
        print(f"         ❌ {res['body'][:300]}")
        print("            服务在监听但不返回推理结果 —— scheduler 进程很可能已崩溃。")
        print("            请回到服务端终端查看是否有 traceback，常见原因：")
        print("              · 自定义算子（rotary / norm / activation / sample）抛异常")
        print("              · KV cache 分配失败（显存不足）")
        print("              · attention 后端与 GPU 架构不匹配（务必显式 --attn fa）")
        return
    print(f"         耗时 {dt:.2f}s  HTTP {res['status']}")
    print(f"         响应: {res['body'][:300]}")
    print("         ✅ 能返回结果")
    if "乱码" or True:  # 提示用户自查内容质量
        print("         ℹ️  请肉眼确认 content 是否为正常文本；若是乱码，")
        print("            说明前向计算有问题，此时测出的延迟数据没有意义。")

    print("\n步骤 3/3  SSE 格式：流式请求（超时 20s）")
    r = client.stream("你好", 8)
    if not r.ok:
        print(f"         ❌ {r.error}")
        return
    print(f"         n_tokens={r.n_tokens}  TTFT={r.ttft*1000:.1f}ms  "
          f"total={r.total*1000:.1f}ms")
    if r.raw_samples:
        print("         原始 SSE 行:")
        for line in r.raw_samples:
            print(f"           {line[:150]}")
    print("         ✅ SSE 可解析" if r.n_tokens > 0 else "         ⚠️  未解析到任何 token")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

TESTS = {
    "diag": (test_diag, "快速诊断（排障先跑这个）"),
    "prefix": (test_prefix, "RadixAttention 前缀缓存"),
    "chunked": (test_chunked, "Chunked Prefill"),
    "interfere": (test_interference, "Prefill 干扰 Decode"),
    "concur": (test_concurrency, "并发吞吐曲线"),
    "graph": (test_graph, "CUDA graph 边界"),
    "capacity": (test_capacity, "KV cache 容量"),
}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="mini-sglang 调度特性测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="可用场景: " + ", ".join(f"{k} ({v[1]})" for k, v in TESTS.items()),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--model", default="Qwen3-4B-Instruct-2507")

    p.add_argument("--max-tokens", type=int, default=64, help="默认生成长度")
    p.add_argument("--prompt-len", type=int, default=256, help="常规 prompt 目标长度")
    p.add_argument("--long-prompt", type=int, default=2048, help="长 prompt 目标长度")
    p.add_argument("--repeat", type=int, default=5, help="prefix 场景重复次数")

    p.add_argument("--chunked-lengths", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--chunked-siblings", type=int, default=4, help="伴随的短 decode 请求数")
    p.add_argument("--chunked-decode-tokens", type=int, default=128)

    p.add_argument("--decode-count", type=int, default=8, help="interfere 场景的 decode 并发数")
    p.add_argument("--inject-delay", type=float, default=1.5, help="插入 prefill 前的等待秒数")

    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 8, 32])
    p.add_argument("--graph-bs", type=int, nargs="+", default=[1, 8, 64, 160, 256])
    p.add_argument("--capacity-bs", type=int, nargs="+", default=[8, 32, 128, 256])
    p.add_argument("--capacity-tokens", type=int, default=32)

    p.add_argument("--timeout", type=float, default=60.0,
                   help="单次请求超时秒数（默认 60；服务端崩溃时会阻塞到超时才报错）")
    p.add_argument("--verbose", action="store_true", help="打印原始 SSE 行，便于排障")
    p.add_argument("--no-warmup", action="store_true", help="跳过 warmup 请求")
    p.add_argument("tests", nargs="+", choices=list(TESTS) + ["all"])

    args = p.parse_args(argv)
    if "all" in args.tests:
        args.tests = [k for k in TESTS if k != "diag"]

    client = Client(args.host, args.port, args.model,
                    timeout=args.timeout, verbose=args.verbose)

    # 连通性检查（只确认端口在监听，不代表能推理）
    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            print(f"❌ 服务端返回 HTTP {resp.status}，请确认 mini-sglang 已启动在 "
                  f"{args.host}:{args.port}")
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法连接 {args.host}:{args.port} —— {type(e).__name__}: {e}")
        print("   请先启动服务：python -m minisgl --model <path> --attn fa")
        print("   （注意：端口在监听 ≠ 能推理。若卡在 warmup，请改跑 diag 场景）")
        return 1

    # diag 自带健康检查，不需要 warmup（warmup 本身可能就是卡住的那一步）
    if not args.no_warmup and args.tests != ["diag"]:
        print(f"warmup ... (超时 {args.timeout:.0f}s，长时间无输出说明服务端未响应)",
              end=" ", flush=True)
        w = client.stream("你好", 8)
        print(f"TTFT={w.ttft*1000:.1f}ms {'OK' if w.ok else 'FAILED: ' + w.error}")
        if not w.ok:
            print()
            print("❌ warmup 失败，后续测量无意义。建议：")
            print(f"   1) 检查 scheduler 进程是否存活: ps aux | grep '[m]inisgl-TP0-scheduler'")
            print(f"   2) 查看服务端终端的 traceback")
            print(f"   3) 跑诊断: {sys.argv[0]} --model {args.model} diag --verbose")
            return 1

    for name in args.tests:
        fn, desc = TESTS[name]
        print()
        print("#" * 96)
        print(f"# {name} —— {desc}")
        print("#" * 96)
        fn(client, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
