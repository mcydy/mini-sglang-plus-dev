#!/usr/bin/env python3
"""
mini-sglang 基准测试脚本 v2 —— 只用 /generate 原生接口

    POST /generate  {"prompt": "...", "max_tokens": N, "temperature": 0.0}

========================================================================
核心原理：两点差分法（非流式接口下唯一可靠的 TTFT/TPOT 测法）
========================================================================
一次性返回的 /generate 拿不到"第一个 token 的时刻"，所以用同一个
prompt 跑两次不同 max_tokens：

    t(1) = TTFT + eps                 只生成 1 个 token
    t(k) = TTFT + (k-1) * TPOT        生成 k 个 token
    =>  TPOT = ( t(k) - t(1) ) / (k-1)
    =>  TTFT_pure = t(1) - TPOT       去掉 t(1) 里包含的那一个 decode step

注意：直接把 t(1) 当 TTFT 会高估（含 1 个 decode step），本脚本已去卷积。

========================================================================
新增指标（v2）
========================================================================
  MFU    Model FLOPs Utilization      算力利用率（★ MoE 下 prefill/decode 口径不同）
  MBU    Model Bandwidth Utilization  带宽利用率（decode 是访存瓶颈，MBU 比 MFU 更有意义）
  P99 / P99.9                         尾部延迟全套分位数
  CV     变异系数 (std/mean)           抖动程度，CUDA graph 是否生效的敏感指标
  Goodput                             满足 SLO 的吞吐（比裸吞吐更实用）

========================================================================
用法示例
========================================================================
  # 单点（推荐带上模型路径，才能算 MFU/MBU）
  python3 bench_ttft_tpot.py --model-path /home/jeff/work/demo/OLMoE-1B-7B-0924 \
      --input-len 128 --output-len 32 --num-requests 8

  # 并发扫描
  python3 bench_ttft_tpot.py --model-path <path> --mode conc \
      --concurrency-list 1,2,4,8 --input-len 128 --output-len 32

  # input_len 扫描
  python3 bench_ttft_tpot.py --mode input \
      --input-len-list 64,128,512,1024 --output-len 8

  # 不想给模型路径，直接指定参数量（单位：B，十亿）
  python3 bench_ttft_tpot.py --total-params 7.0 --active-params 1.0

  # 探活
  python3 bench_ttft_tpot.py --probe
========================================================================
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- 默认配置
DEFAULT_BASE_URL = "http://127.0.0.1:1919"

# 峰值算力口径：BF16/FP16 Tensor Core **dense**（不含 sparsity 翻倍）
# 若你的机器规格口径不同，用 --peak-flops 覆盖
GPU_SPECS = {
    "3090":  {"flops": 71.0e12,  "bw": 936e9},
    "4090":  {"flops": 82.6e12,  "bw": 1008e9},
    "a10":   {"flops": 62.5e12,  "bw": 600e9},
    "l40s":  {"flops": 362e12,   "bw": 864e9},
    "a100":  {"flops": 312e12,   "bw": 2039e9},
    "h100":  {"flops": 989e12,   "bw": 3350e9},
    "h200":  {"flops": 989e12,   "bw": 4800e9},
}

BASE_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The study of logic and formal "
    "reasoning began in the classical world, and the modern field of "
    "artificial intelligence research was founded at a workshop at "
    "Dartmouth College in 1956. The field has since experienced several "
    "waves of optimism, followed by disappointment and the loss of "
    "funding, followed by new approaches, success and renewed funding. "
    "Artificial intelligence research has tried and abandoned many "
    "different approaches during its lifetime, including simulating the "
    "brain, modeling human problem solving, formal logic, large databases "
    "of knowledge and imitating animal behavior. In the first decades of "
    "the 21st century, highly mathematical statistical machine learning "
    "has dominated the field, and this technique has proved highly "
    "successful, helping to solve many challenging problems throughout "
    "industry and academia. "
)


# ---------------------------------------------------------------- 参数量估算
def _get(c: dict, keys, default=None):
    for k in keys:
        if k in c and c[k] is not None:
            return c[k]
    return default


def estimate_params(config_path: str) -> dict | None:
    """从 config.json 估算 total / active 参数量（支持 MoE 与 dense）"""
    try:
        with open(os.path.join(config_path, "config.json"), "r") as f:
            c = json.load(f)
    except Exception as e:
        print(f"[warn] 读取 config.json 失败: {e}")
        return None

    h = _get(c, ["hidden_size", "d_model", "n_embd"])
    L = _get(c, ["num_hidden_layers", "n_layers", "n_layer"])
    vocab = _get(c, ["vocab_size", "n_vocab"], 0)
    n_heads = _get(c, ["num_attention_heads", "n_heads"], 1)
    n_kv = _get(c, ["num_key_value_heads", "n_kv_heads"], n_heads)
    if n_kv is None:
        n_kv = n_heads
    head_dim = _get(c, ["head_dim"], None) or (h // n_heads if h and n_heads else 0)

    # MoE 相关
    n_exp = _get(c, ["num_experts", "n_routed_experts", "num_local_experts"], None)
    top_k = _get(c, ["num_experts_per_tok", "top_k", "n_experts_per_tok"], None)
    moe_inter = _get(c, ["moe_intermediate_size", "moe_ffn_hidden_size"], None)
    inter = _get(c, ["intermediate_size", "ffn_hidden_size"], None)

    if not h or not L:
        print("[warn] config.json 缺少 hidden_size / num_hidden_layers")
        return None

    # ---- 每层 attention（qkv + o）
    attn = (h * n_heads * head_dim          # q_proj
            + h * n_kv * head_dim           # k_proj
            + h * n_kv * head_dim           # v_proj
            + n_heads * head_dim * h)       # o_proj

    # ---- 每层 FFN
    if n_exp and moe_inter:
        # MoE：SwiGLU 三矩阵（gate/up/down）
        expert_ffn = 3 * h * moe_inter
        total_ffn = n_exp * expert_ffn
        k = top_k if top_k else 1
        active_ffn = k * expert_ffn
        is_moe = True
    elif inter:
        expert_ffn = 3 * h * inter          # 多数现代模型是 SwiGLU
        total_ffn = expert_ffn
        active_ffn = expert_ffn
        is_moe = False
        n_exp, k = 1, 1
    else:
        print("[warn] 找不到 FFN 维度配置")
        return None

    # ---- 非 FFN 部分
    embed = vocab * h
    lm_head = vocab * h
    non_ffn = embed + lm_head + L * attn

    total = non_ffn + L * total_ffn
    active = non_ffn + L * active_ffn

    return {
        "total": total,
        "active": active,
        "is_moe": is_moe,
        "num_experts": n_exp if n_exp else 1,
        "top_k": k if is_moe else 1,
        "hidden": h,
        "layers": L,
        "vocab": vocab,
    }


# ---------------------------------------------------------------- HTTP
def post_generate(base_url: str, prompt: str, max_tokens: int,
                  timeout: float) -> tuple[float, dict]:
    """返回 (耗时ms, 响应dict)"""
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        return ms, json.loads(raw.decode())
    except Exception:
        return ms, {}


def make_prompt(target_tokens: int) -> str:
    """按目标 token 数生成 prompt（粗估 1 token ≈ 4 字符）"""
    target_chars = target_tokens * 4
    if target_chars <= len(BASE_TEXT):
        return BASE_TEXT[:max(target_chars, 16)]
    rep = (target_chars // len(BASE_TEXT)) + 1
    return (BASE_TEXT * rep)[:target_chars]


def count_output_tokens(data: dict, fallback: int) -> int:
    """尽量从返回体里取真实生成 token 数"""
    if not isinstance(data, dict):
        return fallback
    mi = data.get("meta_info")
    if isinstance(mi, dict):
        for k in ("completion_tokens", "generated_tokens", "output_tokens"):
            if k in mi:
                return int(mi[k])
    for k in ("completion_tokens", "output_tokens"):
        if k in data:
            return int(data[k])
    return fallback


# ---------------------------------------------------------------- 统计
def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats_line(name: str, vals: list[float], unit: str = "ms",
               width: int = 18) -> str:
    """完整分位数行：mean/p50/p90/p99/p99.9/std/CV/max"""
    if not vals:
        return f"{name:<{width}} (no data)"
    s = sorted(vals)
    mean = statistics.mean(s)
    sd = statistics.pstdev(s) if len(s) > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else float("nan")
    return (f"{name:<{width}} mean={mean:9.2f}  p50={pct(s,.5):8.2f}  "
            f"p90={pct(s,.9):8.2f}  p99={pct(s,.99):8.2f}  "
            f"p99.9={pct(s,.999):8.2f}  std={sd:7.2f}  CV={cv:5.1f}%  "
            f"max={s[-1]:8.2f}{unit}")


def cv_of(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = statistics.mean(vals)
    return statistics.pstdev(vals) / m * 100.0 if m > 0 else float("nan")


# ---------------------------------------------------------------- 核心测量
def run_batch(base_url: str, prompt: str, max_tokens: int, n: int,
              concurrency: int, timeout: float, label: str,
              verbose: bool = True) -> tuple[list[float], int]:
    """并发发 n 个请求，返回 (耗时ms列表, 失败数)"""
    if verbose:
        print(f"[{label}] n={n} concurrency={concurrency} max_tokens={max_tokens}",
              flush=True)

    results: list[float] = []
    lock = threading.Lock()
    errors = [0]

    def one(i: int):
        try:
            ms, _ = post_generate(base_url, prompt, max_tokens, timeout)
            with lock:
                results.append(ms)
            if verbose:
                print(f"    #{i+1:<3} -> {ms:9.2f} ms", flush=True)
        except Exception as e:
            with lock:
                errors[0] += 1
            if verbose:
                print(f"    #{i+1:<3} -> ERROR: {e}", flush=True)

    if concurrency <= 1:
        for i in range(n):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(one, range(n)))
    return results, errors[0]


def measure_point(base_url, input_len, output_len, num_requests, concurrency,
                  warmup, timeout, t1_requests, prompt, hw, params) -> dict:
    """单点完整测量：warmup -> t(1) -> t(k) -> 差分求 TTFT/TPOT"""
    p = prompt if prompt else make_prompt(input_len)
    print(f"[info] prompt 约 {len(p)} 字符 (目标 {input_len} tokens)")
    print(f"[info] prompt 前 80 字符: {p[:80]!r}")
    print(f"[info] 单次超时 {timeout}s", flush=True)

    errs = 0
    if warmup > 0:
        print("\n[warmup]", flush=True)
        _, e = run_batch(base_url, p, max(output_len, 8), warmup, 1, timeout, "warmup")
        errs += e

    print("\n[measure] t(1) —— 近似 TTFT", flush=True)
    t1_list, e = run_batch(base_url, p, 1, t1_requests, 1, timeout, "t(1)")
    errs += e

    print(f"\n[measure] t({output_len}) —— 完整生成", flush=True)
    tk_list, e = run_batch(base_url, p, output_len, num_requests,
                           concurrency, timeout, f"t({output_len})")
    errs += e

    if not t1_list or not tk_list:
        raise RuntimeError("测量失败：没有成功的请求")

    # ---- 差分
    t1_mean = statistics.mean(t1_list)
    tk_mean = statistics.mean(tk_list)
    tpot = (tk_mean - t1_mean) / max(output_len - 1, 1)
    ttft_pure = t1_mean - tpot

    # ---- TPOT 分布（排序配对近似）
    a = sorted(t1_list)
    b = sorted(tk_list)
    n_pair = min(len(a), len(b))
    tpot_dist = [(b[i] - a[i]) / max(output_len - 1, 1) for i in range(n_pair)]
    ttft_dist = [a[i] - tpot_dist[i] for i in range(n_pair)]

    prefill_tps = input_len / (ttft_pure / 1000.0) if ttft_pure > 0 else float("nan")
    decode_tps = 1000.0 / tpot if tpot > 0 else float("nan")

    total_tokens = num_requests * output_len
    wall_s = sum(tk_list) / 1000.0 / max(concurrency, 1)
    agg_tps = total_tokens / wall_s if wall_s > 0 else float("nan")

    # ---- MFU / MBU
    r = {
        "input_len": input_len, "output_len": output_len,
        "concurrency": concurrency, "n": num_requests, "errors": errs,
        "t1_mean": t1_mean, "t1_list": t1_list,
        "tk_mean": tk_mean, "tk_list": tk_list,
        "ttft_pure": ttft_pure, "ttft_dist": ttft_dist,
        "tpot": tpot, "tpot_dist": tpot_dist,
        "prefill_tps": prefill_tps, "decode_tps": decode_tps,
        "agg_tps": agg_tps, "wall_s": wall_s,
        "mfu_prefill": float("nan"), "mfu_decode": float("nan"),
        "mbu_decode": float("nan"), "mbu_prefill": float("nan"),
    }

    if params and hw:
        P_total = params["total"]
        P_active = params["active"]
        pb = hw["params_bytes"]

        # prefill：长序列下几乎全部专家都会被激活 → 用 total
        # FLOPs ≈ 2 * P_total * input_len
        flops_prefill = 2.0 * P_total * input_len
        mfu_p = flops_prefill / (ttft_pure / 1000.0) / hw["flops"] * 100.0
        r["mfu_prefill"] = mfu_p

        # decode：每步只访问 top_k 个专家 → 用 active
        decode_time_s = (tk_mean - ttft_pure) / 1000.0
        flops_decode = 2.0 * P_active * (output_len - 1)
        if decode_time_s > 0:
            r["mfu_decode"] = (flops_decode / decode_time_s / hw["flops"] * 100.0)

        # MBU：decode 每步要把激活权重读一遍
        bytes_decode = P_active * pb
        steps = max(output_len - 1, 1)
        per_step_s = decode_time_s / steps
        if per_step_s > 0:
            r["mbu_decode"] = (bytes_decode / per_step_s / hw["bw"] * 100.0)

        # prefill 读全部权重
        bytes_prefill = P_total * pb
        if ttft_pure > 0:
            r["mbu_prefill"] = (bytes_prefill / (ttft_pure / 1000.0) / hw["bw"] * 100.0)

    return r


# ---------------------------------------------------------------- 输出
def fmt(v: float, suffix: str = "%", width: int = 6) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A".rjust(width) + suffix.ljust(2)
    return f"{v:{width}.2f}{suffix}"


def print_point(r: dict, hw: dict | None, params: dict | None):
    bar = "=" * 78
    print("\n" + bar)
    print(f"  /generate   input_len={r['input_len']}  output_len={r['output_len']}"
          f"  concurrency={r['concurrency']}  requests={r['n']}")
    print(bar)

    # --- 延迟
    print("【延迟】")
    print(f"  TTFT (去卷积)      = {r['ttft_pure']:9.2f} ms   (= t(1) - TPOT)")
    print(f"  TPOT               = {r['tpot']:9.2f} ms   -> {r['decode_tps']:.1f} tok/s (单请求)")
    print(f"  ITL  (非流式≈TPOT) = {r['tpot']:9.2f} ms   (token 间延迟)")
    print(f"  E2E  (t({r['output_len']}))       = {r['tk_mean']:9.2f} ms")
    print("-" * 78)
    print("  " + stats_line("TTFT分布", r["ttft_dist"]))
    print("  " + stats_line("TPOT分布", r["tpot_dist"]))
    print("  " + stats_line("t(1)", r["t1_list"]))
    print("  " + stats_line(f"t({r['output_len']})", r["tk_list"]))
    print("  * p99/p99.9 为尾部延迟；CV=变异系数(std/mean)，越大抖动越大")

    # --- 吞吐
    print("-" * 78)
    print("【吞吐】")
    print(f"  Prefill 吞吐       = {r['prefill_tps']:9.1f} tok/s  "
          f"({r['input_len']} tokens / TTFT)")
    print(f"  Decode 吞吐        = {r['decode_tps']:9.1f} tok/s  (单请求 1/TPOT)")
    print(f"  聚合吞吐           = {r['agg_tps']:9.1f} tok/s  "
          f"(concurrency={r['concurrency']})")

    # --- MFU / MBU
    print("-" * 78)
    print("【硬件利用率】")
    if params and hw:
        print(f"  模型: total={params['total']/1e9:.2f}B  "
              f"active={params['active']/1e9:.2f}B  "
              f"experts={params['num_experts']}/top{params['top_k']}  "
              f"({'MoE' if params['is_moe'] else 'dense'})")
        print(f"  硬件: peak_flops={hw['flops']/1e12:.1f} TFLOPS  "
              f"peak_bw={hw['bw']/1e9:.0f} GB/s  "
              f"params_bytes={hw['params_bytes']}")
        print("-" * 78)
        print(f"  MFU (prefill)      = {fmt(r['mfu_prefill'])}   "
              f"口径: 2 × total_params × input_len  (长序列几乎全专家激活)")
        print(f"  MFU (decode)       = {fmt(r['mfu_decode'])}   "
              f"口径: 2 × active_params × (out-1)")
        print(f"  MBU (decode)       = {fmt(r['mbu_decode'])}   "
              f"口径: active_params × bytes / 每步耗时")
        print(f"  MBU (prefill)      = {fmt(r['mbu_prefill'])}   "
              f"口径: total_params × bytes / TTFT")
        print("-" * 78)
        print("  判读: decode 是访存瓶颈 -> MFU 天然低(个位数到十几%),")
        print("        此时看 MBU 更有意义; prefill 是算力瓶颈 -> 看 MFU。")
        vals = [r["mfu_prefill"], r["mfu_decode"],
                r["mbu_decode"], r["mbu_prefill"]]
        if any((not math.isnan(v)) and v > 100.0 for v in vals):
            print("  ⚠ 有指标 >100% —— 通常是口径问题，按顺序排查:")
            print("     1) 峰值算力口径 (本脚本用 dense 不含 sparsity, "
                  "用 --peak-flops 覆盖)")
            print("     2) 参数量估算偏差 (用 --total-params/--active-params 直接给)")
            print("     3) 模型实际跑了更高精度 (--params-bytes)")
            print("     4) 若只有 prefill MFU>100%: 说明 TTFT 里含非计算开销"
                  "或 input_len 估算偏大")
    else:
        print("  N/A —— 需要 --model-path <config.json所在目录> 或 "
              "--total-params/--active-params")

    if r["errors"]:
        print("-" * 78)
        print(f"  ⚠ 失败请求数: {r['errors']}")
    print(bar)


def sweep_concurrency(base_url, input_len, output_len, conc_list, num_requests,
                      warmup, timeout, t1_requests, prompt, hw, params, slo_ms):
    print(f"[info] 并发扫描 conc={conc_list}  input_len={input_len} "
          f"output_len={output_len}", flush=True)
    p = prompt if prompt else make_prompt(input_len)
    print(f"[info] prompt 约 {len(p)} 字符", flush=True)

    if warmup > 0:
        print("\n[warmup]", flush=True)
        run_batch(base_url, p, max(output_len, 8), warmup, 1, timeout, "warmup")

    print("\n[measure] t(1) 基准（串行）", flush=True)
    t1_list, _ = run_batch(base_url, p, 1, t1_requests, 1, timeout, "t(1)")
    t1_mean = statistics.mean(t1_list) if t1_list else float("nan")

    rows = []
    for c in conc_list:
        print(f"\n[measure] concurrency={c}", flush=True)
        tk, errs = run_batch(base_url, p, output_len, num_requests, c,
                             timeout, f"conc={c}")
        if not tk:
            continue
        wall = sum(tk) / 1000.0 / max(c, 1)
        agg = (num_requests * output_len) / wall if wall > 0 else float("nan")
        per_tok = statistics.mean(tk) / output_len
        good = (sum(1 for v in tk if v <= slo_ms) / len(tk) * 100.0) if slo_ms else float("nan")
        rows.append({
            "conc": c, "n": len(tk), "errors": errs,
            "tk_mean": statistics.mean(tk), "tk_p90": pct(sorted(tk), .9),
            "tk_p99": pct(sorted(tk), .99), "tk_p999": pct(sorted(tk), .999),
            "per_tok": per_tok, "agg": agg, "cv": cv_of(tk),
            "goodput": good,
        })
        print(f"    -> mean={statistics.mean(tk):8.2f}ms  "
              f"p99={pct(sorted(tk),.99):8.2f}ms  "
              f"agg={agg:7.1f} tok/s  CV={cv_of(tk):5.1f}%", flush=True)

    print("\n" + "=" * 78)
    print("  并发扫描结果")
    print("=" * 78)
    hdr = (f"{'conc':>5} {'n':>4} {'E2E mean':>9} {'p90':>8} {'p99':>8} "
           f"{'p99.9':>8} {'CV%':>6} {'per-tok':>9} {'Agg tok/s':>10}")
    if slo_ms:
        hdr += f" {'SLO达标%':>9}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        line = (f"{r['conc']:>5} {r['n']:>4} {r['tk_mean']:>9.2f} "
                f"{r['tk_p90']:>8.2f} {r['tk_p99']:>8.2f} {r['tk_p999']:>8.2f} "
                f"{r['cv']:>6.1f} {r['per_tok']:>9.2f} {r['agg']:>10.1f}")
        if slo_ms:
            line += f" {r['goodput']:>9.1f}"
        print(line)
    print("=" * 78)
    print(f"  串行 TTFT 基准 (引擎属性, 与并发无关): {t1_mean:.2f} ms")
    if rows:
        best = max(rows, key=lambda x: x["agg"])
        print(f"  最高聚合吞吐: concurrency={best['conc']} -> {best['agg']:.1f} tok/s")
        if len(rows) > 1:
            base = rows[0]["agg"]
            for r in rows[1:]:
                print(f"    conc {rows[0]['conc']}->{r['conc']}: "
                      f"吞吐 {base:.1f} -> {r['agg']:.1f} "
                      f"({r['agg']/base:.2f}x, 理想线性应为 {r['conc']/rows[0]['conc']:.2f}x)")
    if slo_ms:
        print(f"  SLO 阈值: {slo_ms} ms (E2E)")
    print("  * per-token = E2E / output_len, 含排队与批处理等待, 非纯 decode TPOT")
    print("  * Agg tok/s 不受排队影响, 是并发场景的硬指标")


def sweep_input(base_url, input_len_list, output_len, num_requests, warmup,
                timeout, t1_requests, hw, params):
    print(f"[info] input_len 扫描 {input_len_list}  output_len={output_len}",
          flush=True)

    rows = []
    for il in input_len_list:
        p = make_prompt(il)
        print(f"\n[input_len={il}] prompt 约 {len(p)} 字符", flush=True)
        if warmup > 0:
            run_batch(base_url, p, max(output_len, 8), 1, 1, timeout, "warmup",
                      verbose=False)
        t1, _ = run_batch(base_url, p, 1, t1_requests, 1, timeout, "t(1)",
                          verbose=False)
        tk, _ = run_batch(base_url, p, output_len, num_requests, 1, timeout,
                          f"t({output_len})", verbose=False)
        if not t1 or not tk:
            continue
        t1m, tkm = statistics.mean(t1), statistics.mean(tk)
        tpot = (tkm - t1m) / max(output_len - 1, 1)
        ttft = t1m - tpot
        row = {
            "input_len": il, "ttft": ttft, "tpot": tpot,
            "prefill_tps": il / (ttft / 1000.0) if ttft > 0 else float("nan"),
            "mfu_prefill": float("nan"),
        }
        if params and hw and ttft > 0:
            row["mfu_prefill"] = (2.0 * params["total"] * il
                                  / (ttft / 1000.0) / hw["flops"] * 100.0)
        rows.append(row)
        print(f"    -> TTFT={ttft:8.2f}ms  TPOT={tpot:7.2f}ms  "
              f"prefill={row['prefill_tps']:8.1f} tok/s  "
              f"MFU_p={row['mfu_prefill']:5.2f}%", flush=True)

    print("\n" + "=" * 78)
    print("  input_len 扫描结果")
    print("=" * 78)
    print(f"{'input_len':>10} {'TTFT ms':>10} {'TPOT ms':>9} "
          f"{'prefill tok/s':>14} {'MFU_prefill %':>14}")
    print("-" * 78)
    for r in rows:
        print(f"{r['input_len']:>10} {r['ttft']:>10.2f} {r['tpot']:>9.2f} "
              f"{r['prefill_tps']:>14.1f} {r['mfu_prefill']:>14.2f}")
    print("=" * 78)
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        ratio_len = b["input_len"] / a["input_len"]
        ratio_t = b["ttft"] / a["ttft"] if a["ttft"] > 0 else float("nan")
        print(f"  input_len ×{ratio_len:.1f} -> TTFT ×{ratio_t:.2f}")
        print(f"  若 TTFT 增长 << {ratio_len:.1f}x, 说明 prefill 时间主要花在"
              f"【读模型权重】而非【算 token】（MoE 的典型特征）")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="mini-sglang /generate 基准测试 v2 (TTFT/TPOT/MFU/MBU/P99)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--mode", default="point",
                    choices=["point", "conc", "input"])
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--concurrency-list", default="1,2,4,8")
    ap.add_argument("--input-len-list", default="64,128,512,1024")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--t1-requests", type=int, default=5)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--probe", action="store_true")

    # MFU / MBU
    ap.add_argument("--model-path", default=None,
                    help="模型目录（含 config.json），用于估算参数量")
    ap.add_argument("--total-params", type=float, default=None,
                    help="总参数量，单位 B（十亿）")
    ap.add_argument("--active-params", type=float, default=None,
                    help="激活参数量，单位 B（十亿），MoE 模型必填")
    ap.add_argument("--gpu", default="3090", choices=list(GPU_SPECS.keys()))
    ap.add_argument("--peak-flops", type=float, default=None,
                    help="覆盖峰值算力，单位 TFLOPS")
    ap.add_argument("--peak-bw", type=float, default=None,
                    help="覆盖峰值带宽，单位 GB/s")
    ap.add_argument("--params-bytes", type=float, default=2.0,
                    help="每参数字节数 (bf16/fp16=2, fp8=1, fp32=4)")
    ap.add_argument("--slo-ms", type=float, default=None,
                    help="E2E 延迟 SLO 阈值(ms)，用于算 goodput")

    args = ap.parse_args()

    if args.probe:
        print(f"[probe] POST {args.base_url}/generate")
        try:
            ms, data = post_generate(args.base_url, "The capital of France is",
                                     8, args.timeout)
            print(f"[probe] 耗时 {ms:.2f} ms")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        except Exception as e:
            print(f"[probe] 失败: {e}")
        return

    # ---- 硬件规格
    spec = GPU_SPECS.get(args.gpu, GPU_SPECS["3090"])
    hw = {
        "flops": (args.peak_flops * 1e12) if args.peak_flops else spec["flops"],
        "bw": (args.peak_bw * 1e9) if args.peak_bw else spec["bw"],
        "params_bytes": args.params_bytes,
    }

    # ---- 参数量
    params = None
    if args.total_params:
        params = {
            "total": args.total_params * 1e9,
            "active": (args.active_params or args.total_params) * 1e9,
            "is_moe": bool(args.active_params
                           and args.active_params < args.total_params),
            "num_experts": 0, "top_k": 0,
        }
    elif args.model_path:
        params = estimate_params(args.model_path)
        if params:
            print(f"[info] 参数量估算: total={params['total']/1e9:.2f}B  "
                  f"active={params['active']/1e9:.2f}B  "
                  f"experts={params['num_experts']}/top{params['top_k']}  "
                  f"({'MoE' if params['is_moe'] else 'dense'})")

    if args.mode == "point":
        r = measure_point(args.base_url, args.input_len, args.output_len,
                          args.num_requests, args.concurrency, args.warmup,
                          args.timeout, args.t1_requests, args.prompt,
                          hw, params)
        print_point(r, hw, params)

    elif args.mode == "conc":
        clist = [int(x) for x in args.concurrency_list.split(",") if x.strip()]
        sweep_concurrency(args.base_url, args.input_len, args.output_len,
                          clist, args.num_requests, args.warmup, args.timeout,
                          args.t1_requests, args.prompt, hw, params, args.slo_ms)

    elif args.mode == "input":
        ilist = [int(x) for x in args.input_len_list.split(",") if x.strip()]
        sweep_input(args.base_url, ilist, args.output_len, args.num_requests,
                    args.warmup, args.timeout, args.t1_requests, hw, params)


if __name__ == "__main__":
    main()
