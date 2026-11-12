#!/usr/bin/env python3
"""
mini-sglang Radix Cache 命中率测试 (v2)

两种模式:
  mode=log      ★推荐★ 解析服务端 [CACHE] 日志, 拿到引擎内部的精确命中数
                (需要你已按文档在 scheduler/cache.py 的 match_prefix 里加了打印)
  mode=blackbox 黑盒差分, 不改源码也能用, 靠耗时反推命中数 (精度较低)

=========================== 模式 log 用法 ===========================
1) 服务端输出重定向到文件 (必须带 -u 关闭缓冲, 否则读不到实时日志):
     python -u -m minisgl --model <M> --port 1919 \
         --attention-backend fa --cuda-graph-max-bs 2 2>&1 | tee /tmp/minisgl.log

2) 另开终端跑本脚本:
     python3 bench_kvcache_hit_rate.py --mode log --log-file /tmp/minisgl.log \
         --model-path /home/jeff/work/demo/OLMoE-1B-7B-0924

========================== 模式 blackbox 用法 =========================
    python3 bench_kvcache_hit_rate.py --mode blackbox \
        --model-path /home/jeff/work/demo/OLMoE-1B-7B-0924
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 文本素材

SYS_PROMPT = (
    "You are a helpful, harmless and honest AI assistant. "
    "Please answer the following questions carefully, concisely, "
    "and based only on factual information."
)

QUERIES = [
    "What is the capital of France?",
    "Who wrote the play Hamlet?",
    "Explain Newton's law of universal gravitation.",
    "What is the boiling point of water at sea level?",
    "Name three planets in the Solar System.",
]

BASE_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The seeds of modern AI were planted by "
    "classical philosophers who attempted to describe the process of human "
    "thinking as the mechanical manipulation of symbols. This work culminated "
    "in the invention of the programmable digital computer in the 1940s, a "
    "machine based on the abstract essence of mathematical reasoning. This "
    "device and the ideas behind it inspired a handful of scientists to begin "
    "seriously discussing the possibility of building an electronic brain. "
)

DIFF_SUFFIXES = [
    "The railway network expanded rapidly across the continent during this era.",
    "Marine biologists discovered several new species near the coral reefs.",
    "Ancient manuscripts were carefully restored by teams of conservators.",
]


# ---------------------------------------------------------------- HTTP

def http_post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"请求失败: {e}") from e
    dt = (time.perf_counter() - t0) * 1000.0
    try:
        return json.loads(body), dt
    except Exception:
        return {"_raw": body.decode(errors="replace")}, dt


def generate(server, prompt, max_tokens=1, temperature=0.0, timeout=120):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return http_post(f"{server}/generate", payload, timeout=timeout)


# ---------------------------------------------------------------- tokenizer

class Tokenizer:
    def __init__(self, model_path):
        self.tok = None
        if model_path:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(model_path)
                print(f"[info] 已加载 tokenizer: {model_path}")
            except Exception as e:
                print(f"[warn] tokenizer 加载失败({e}), 改用 chars/4 估算")

    def ntok(self, text):
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False).input_ids)
        return max(1, len(text) // 4)

    def shared_prefix_len(self, prefix, full):
        """prefix 拼上不同后缀后, 实际能共享的 token 数 (考虑 BPE 跨边界合并)"""
        if self.tok is None:
            return self.ntok(prefix)
        ids_p = self.tok(prefix, add_special_tokens=False).input_ids
        ids_f = self.tok(full, add_special_tokens=False).input_ids
        n = 0
        for a, b in zip(ids_p, ids_f):
            if a != b:
                break
            n += 1
        return n


# ---------------------------------------------------------------- 日志模式

CACHE_RE = re.compile(
    r"\[CACHE\]\s+req#\s*(\d+)\s+hit=\s*(\d+)\s*/\s*(\d+)"
)


def read_new_cache_lines(log_file, from_pos):
    """从文件偏移 from_pos 之后读取新的 [CACHE] 行, 返回 (记录列表, 新偏移)"""
    recs = []
    try:
        with open(log_file, "r", errors="replace") as f:
            f.seek(from_pos)
            chunk = f.read()
            new_pos = f.tell()
    except FileNotFoundError:
        raise RuntimeError(f"日志文件不存在: {log_file}")
    for line in chunk.splitlines():
        m = CACHE_RE.search(line)
        if m:
            recs.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return recs, new_pos


def wait_records(log_file, from_pos, expect, timeout=60):
    """等待日志文件里出现 expect 条新记录"""
    deadline = time.time() + timeout
    pos = from_pos
    got = []
    while time.time() < deadline:
        recs, pos = read_new_cache_lines(log_file, pos)
        got.extend(recs)
        if len(got) >= expect:
            return got[:expect], pos
        time.sleep(0.2)
    return got, pos


def log_file_offset(log_file):
    try:
        return os.path.getsize(log_file)
    except FileNotFoundError:
        raise RuntimeError(
            f"日志文件不存在: {log_file}\n"
            f"请先这样启动服务 (注意 -u 和 2>&1 | tee):\n"
            f"  python -u -m minisgl --model <M> --port 1919 \\\n"
            f"      --attention-backend fa --cuda-graph-max-bs 2 2>&1 | tee {log_file}"
        )


def run_log_mode(server, tk, log_file, repeats, timeout):
    pos = log_file_offset(log_file)
    print(f"[info] 日志文件: {log_file}  当前偏移 {pos} 字节")
    print("[warn] 测试期间请确保没有其他客户端在发请求, 否则日志会串行\n")

    # ---------------- 组1: 冷启动 (各不相同的 prompt) ----------------
    print("=" * 78)
    print("[组 1/3] cold —— 3 个内容完全不同的 prompt (期望命中 0)")
    print("=" * 78)
    g1_prompts = [BASE_TEXT + s for s in DIFF_SUFFIXES]
    g1_ntok = [tk.ntok(p) for p in g1_prompts]
    for i, p in enumerate(g1_prompts):
        generate(server, p, max_tokens=1, timeout=timeout)
        print(f"  发送 {i+1}/{len(g1_prompts)}  (~{g1_ntok[i]} tokens)")
    recs1, pos = wait_records(log_file, pos, len(g1_prompts), timeout=timeout)

    # ---------------- 组2: 完全相同 prompt 重复 ----------------
    print("\n" + "=" * 78)
    print(f"[组 2/3] repeat —— 同一 prompt 重复 {repeats} 次 (期望稳态 100%)")
    print("=" * 78)
    rep_prompt = BASE_TEXT + " The field of AI research was founded at Dartmouth."
    rep_ntok = tk.ntok(rep_prompt)
    print(f"  prompt = {rep_ntok} tokens")
    for i in range(repeats):
        generate(server, rep_prompt, max_tokens=1, timeout=timeout)
        print(f"  发送 {i+1}/{repeats}")
    recs2, pos = wait_records(log_file, pos, repeats, timeout=timeout)

    # ---------------- 组3: 共享 system prompt ----------------
    print("\n" + "=" * 78)
    print(f"[组 3/3] shared —— 固定 system prompt + {len(QUERIES)} 个不同 query")
    print("=" * 78)
    shared_prompts, shared_lens = [], []
    for q in QUERIES:
        full = SYS_PROMPT + " " + q
        shared_prompts.append(full)
        shared_lens.append(tk.shared_prefix_len(SYS_PROMPT, full))
    print(f"  system prompt 可共享 token 数: {set(shared_lens)}")
    for i, p in enumerate(shared_prompts):
        generate(server, p, max_tokens=1, timeout=timeout)
        print(f"  发送 {i+1}/{len(shared_prompts)}  (可共享 {shared_lens[i]})")
    recs3, pos = wait_records(log_file, pos, len(shared_prompts), timeout=timeout)

    # ---------------- 报告 ----------------
    print("\n")
    print("=" * 78)
    print("  Radix Cache 命中率报告   (数据来源: 服务端 [CACHE] 日志, 精确值)")
    print("=" * 78)

    def show_group(title, recs, expect_note, shared_ref=None):
        print(f"\n{title}")
        print("-" * 78)
        if not recs:
            print("  [!] 未读到 [CACHE] 记录 —— 检查源码补丁是否生效 / 服务是否重启")
            return
        if len(recs) != expect_note:
            print(f"  [!] 记录数 {len(recs)} != 请求数 {expect_note}, 可能有其他客户端干扰")
        hits = [r[1] for r in recs]
        tots = [r[2] for r in recs]
        for i, (rid, h, t) in enumerate(recs):
            r = 100.0 * h / t if t > 0 else 0.0
            tag = ""
            if i == 0:
                tag = "  <- 首次(冷)"
            print(f"    req#{rid:<4d} hit={h:6d}/{t:<6d} {r:6.2f}%{tag}")
        cold_hit = hits[0]
        rest = hits[1:]
        if rest:
            avg_steady = statistics.mean(rest)
            tot_avg = statistics.mean(tots[1:])
            rate = 100.0 * avg_steady / tot_avg if tot_avg else 0.0
        else:
            avg_steady = 0
            rate = 0.0
        if cold_hit != 0:
            print(f"  [!] 首次命中={cold_hit} 不为 0 —— 前缀可能已被之前的请求缓存")
            print(f"      请重启 mini-sglang 服务后重跑, 否则本组数据无意义")
        print(f"  -> 首次命中     = {cold_hit}  (期望 0)")
        print(f"  -> 稳态平均命中 = {avg_steady:.1f} tokens")
        print(f"  -> 稳态命中率   = {rate:.2f} %")
        if shared_ref:
            pr = 100.0 * avg_steady / shared_ref if shared_ref > 0 else 0.0
            pr = max(0.0, min(100.0, pr))   # clamp: 估算口径差异可能导致 >100%
            print(f"  -> 前缀命中率   = {pr:.2f} %  (实测 {avg_steady:.1f} / 可共享 {shared_ref})")
        return cold_hit, avg_steady

    show_group("[组 1] cold —— 各不相同的 prompt", recs1, len(g1_prompts))
    show_group("[组 2] repeat —— 同一 prompt 重复", recs2, repeats)
    show_group("[组 3] shared —— 共享 system prompt", recs3, len(shared_prompts),
               shared_ref=shared_lens[0] if shared_lens else None)

    print("\n" + "=" * 78)
    print("  判读标准")
    print("=" * 78)
    print("  cold 组   : 全部 0% 才正常 (每个 prompt 都是首次见到)")
    print("  repeat 组 : 首次 0%, 之后 = 100% -> Radix Cache 完全正常")
    print("  shared 组 : 首次 0%, 之后 '前缀命中率' 应接近 100%")
    print("              (整体命中率低是正常的, 因为 query 部分本来就不该命中)")
    print("=" * 78)


# ---------------------------------------------------------------- 黑盒模式

def make_random_prefix(target_chars, seed=0):
    import random
    words = ("the of and to in a is that for it as was with be by on not "
             "he this are or his from at which but have an they one you "
             "were her all she there would their we him been has when who "
             "will more no if out so said what up its about into them can "
             "only other new some could time than then now look its over "
             "also back after use two how our work first well way even "
             "want because any these give day most us").split()
    rnd = random.Random(seed * 7919 + 13)
    out, n = [], 0
    while n < target_chars:
        w = rnd.choice(words)
        out.append(w)
        n += len(w) + 1
    return " ".join(out)[:target_chars]


def run_blackbox_mode(server, tk, repeats, timeout):
    prefix = BASE_TEXT.strip()
    n_prefix = tk.ntok(prefix)
    print(f"[info] prefix = {n_prefix} tokens")
    print(f"[info] prefix 前 80 字符: {prefix[:80]!r}\n")

    print(f"[step 1/3] 基线 T_base (极小 prompt, {repeats} 次)")
    base_xs = []
    for i in range(repeats):
        _, dt = generate(server, "Hi", max_tokens=1, timeout=timeout)
        base_xs.append(dt)
        print(f"    [base {i+1}/{repeats}] {dt:9.2f} ms")
    t_base = statistics.median(base_xs)
    print(f"  -> T_base = {t_base:.2f} ms\n")

    print(f"[step 2/3] 冷启动 T_cold ({repeats} 次, 每次内容不同的随机前缀)")
    colds = []
    for i in range(repeats):
        p = make_random_prefix(len(prefix), seed=i)
        _, dt = generate(server, p, max_tokens=1, timeout=timeout)
        colds.append(dt)
        print(f"    [cold {i+1}/{repeats}] {dt:9.2f} ms")
    t_cold = statistics.median(colds)
    print(f"  -> T_cold = {t_cold:.2f} ms\n")

    print(f"[step 3/3] 热启动 T_warm (重复同一 prefix, {repeats} 次)")
    warm_prompt = prefix + " Summarize:"
    generate(server, warm_prompt, max_tokens=1, timeout=timeout)  # 丢弃首次
    warm_xs = []
    for i in range(repeats):
        _, dt = generate(server, warm_prompt, max_tokens=1, timeout=timeout)
        warm_xs.append(dt)
        print(f"    [warm {i+1}/{repeats}] {dt:9.2f} ms")
    t_warm = statistics.median(warm_xs)
    print(f"  -> T_warm = {t_warm:.2f} ms\n")

    denom = max(1, n_prefix)
    k = (t_cold - t_base) / denom
    hit_tokens = (t_cold - t_warm) / k if k > 0 else 0.0
    hit_rate = hit_tokens / denom * 100.0

    print("=" * 78)
    print(f"  Radix Cache 命中率 (黑盒差分)   prefix={n_prefix} tokens")
    print("=" * 78)
    print(f"  T_base (固定开销)        = {t_base:9.2f} ms")
    print(f"  T_cold (全量 prefill)    = {t_cold:9.2f} ms")
    print(f"  T_warm (命中后)          = {t_warm:9.2f} ms")
    print("-" * 78)
    print(f"  k (每 token prefill)     = {k*1000:9.2f} us/token")
    print(f"  命中 token 数            = {hit_tokens:9.1f} / {n_prefix}")
    print(f"  命中率                   = {max(0.0, min(100.0, hit_rate)):9.1f} %")
    print("=" * 78)
    print("\n[提示] 黑盒模式精度受 HTTP 开销与调度抖动影响;")
    print("       想要精确值请用 --mode log (需在源码 match_prefix 里加打印)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:1919")
    ap.add_argument("--mode", default="log", choices=["log", "blackbox"])
    ap.add_argument("--log-file", default="/tmp/minisgl.log",
                    help="服务端 [CACHE] 日志文件 (mode=log 用)")
    ap.add_argument("--model-path", default=None,
                    help="模型路径, 用于精确统计 token 数")
    ap.add_argument("--repeats", type=int, default=4,
                    help="repeat 组重复次数")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--probe", action="store_true", help="只打印一次返回结构")
    args = ap.parse_args()

    server = args.base_url.rstrip("/")
    print(f"[info] 目标服务: {server}")
    print(f"[info] 模式    : {args.mode}")

    if args.mode == "log":
        print("\n" + "!" * 78)
        print("! 重要: 每组测试的『首次请求』必须为冷启动, 否则结果无意义。")
        print("! 请务必先重启 minisgl 服务再跑本脚本:")
        print("!   python -u -m minisgl --model <M> --port 1919 \\")
        print(f"!       --attention-backend fa --cuda-graph-max-bs 2 2>&1 | tee {args.log_file}")
        print("! (若首次请求的 hit 不为 0, 脚本会在报告里给出『被污染』告警)")
        print("!" * 78 + "\n")

    if args.probe:
        try:
            r, dt = generate(server, "Hello", max_tokens=4, timeout=args.timeout)
        except RuntimeError as e:
            print(f"[error] {e}")
            sys.exit(1)
        print(f"[probe] 耗时 {dt:.2f} ms\n{json.dumps(r, ensure_ascii=False, indent=2)}")
        return

    # 探活
    try:
        generate(server, "Hi", max_tokens=1, timeout=args.timeout)
    except RuntimeError as e:
        print(f"[error] 服务不可达: {e}")
        print(f"        确认服务已启动: curl {server}/generate ...")
        sys.exit(1)

    tk = Tokenizer(args.model_path)

    if args.mode == "log":
        try:
            run_log_mode(server, tk, args.log_file, args.repeats, args.timeout)
        except RuntimeError as e:
            print(f"[error] {e}")
            sys.exit(1)
    else:
        run_blackbox_mode(server, tk, args.repeats, args.timeout)


if __name__ == "__main__":
    main()
