# Mini-SGLang 中文文档

> 本文档由源码阅读生成，旨在帮助开发者快速理解 mini-sglang 的代码结构、系统架构与核心原理。
> 原英文文档见 [README.md](./README.md)。

---

## 目录

- [1. 项目简介与设计目标](#1-项目简介与设计目标)
- [2. 整体系统架构](#2-整体系统架构)
- [3. 请求生命周期](#3-请求生命周期)
- [4. 代码目录逐模块详解](#4-代码目录逐模块详解)
- [5. 核心原理深入讲解](#5-核心原理深入讲解)
- [6. 启动与使用方式](#6-启动与使用方式)
- [7. 关键代码阅读路径建议](#7-关键代码阅读路径建议)

---

## 1. 项目简介与设计目标

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简教学实现，用约 3000 行 Python 代码复现了生产级 LLM 推理框架的核心技术。它不是玩具——它真正可以加载模型、运行推理、提供 OpenAI 兼容 API。

### 设计目标

| 目标 | 说明 |
|------|------|
| **精简可读** | 用最少的代码讲清楚推理框架的核心原理，代码量控制在可通读的范围内 |
| **功能完整** | 覆盖从 API 接收到文本输出的完整链路，包括分词、调度、推理、解码 |
| **技术对齐** | 实现与 SGLang/vLLM 等生产框架相同的核心优化技术 |
| **易于扩展** | 模块化设计，注意力后端、缓存策略、模型结构均可插拔 |

### 支持的核心技术

- **Radix Cache**：基于 Radix 树的前缀复用，自动共享公共前缀的 KV 缓存
- **Chunked Prefill**：将长 prefill 拆分为多个 chunk，与 decode 交替执行
- **Overlap Scheduling**：用双 CUDA Stream 重叠调度计算与下一批准备
- **Tensor Parallelism**：张量并行，多卡切分模型权重
- **CUDA Graph**：捕获 decode 阶段的计算图，消除 Python 开销
- **可插拔注意力后端**：FlashAttention（sgl_kernel）与 FlashInfer 两种实现
- **分页 KV 缓存**：物理分页存储 + 逻辑前缀管理，两层分离设计

### 支持的模型

- Llama（Llama 2 / Llama 3）
- Qwen3
- Qwen3-MoE（混合专家模型）

---

## 2. 整体系统架构

### 2.1 多进程拓扑

Mini-SGLang 采用多进程架构，通过 ZMQ 消息队列连接各组件：

```
                    ┌─────────────────────────────────────────────────┐
                    │              FrontendManager                    │
                    │         (FastAPI + Uvicorn, 主进程)               │
                    │                                                 │
                    │  /generate    /v1/chat/completions   /v1/models  │
                    └────────┬───────────────────────────┬────────────┘
                             │ ZMQ (send)                │ ZMQ (recv)
                             ▼                           ▼
              ┌──────────────────────────┐  ┌──────────────────────────┐
              │   Tokenizer 进程 (×N)     │  │   Detokenizer 进程 (×1)   │
              │                          │  │                          │
              │  文本 → token ids         │  │  token ids → 增量文本     │
              │  (HuggingFace tokenizer)  │  │  (surr/read offset 技巧)  │
              └──────────┬───────────────┘  └──────────┬───────────────┘
                         │ ZMQ                        │ ZMQ
                         ▼                            ▼
              ┌──────────────────────────────────────────────────────┐
              │            Scheduler 进程 (×world_size)               │
              │                                                      │
              │  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
              │  │ PrefillMgr  │  │ DecodeMgr │  │  Engine        │  │
              │  │ (prefill调度) │  │(decode调度)│  │ (模型前向计算)  │  │
              │  └─────────────┘  └──────────┘  └────────────────┘  │
              │                                                      │
              │  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
              │  │ KV Cache │  │ RadixCache │  │ Attention Backend │  │
              │  │ (物理分页) │  │ (逻辑前缀)  │  │ (FA / FlashInfer) │  │
              │  └──────────┘  └───────────┘  └──────────────────┘  │
              └──────────────────────────────────────────────────────┘
```

### 2.2 进程说明

| 进程 | 数量 | 职责 |
|------|------|------|
| **FrontendManager** | 1 | FastAPI HTTP 服务器，接收请求，通过 ZMQ 分发给 tokenizer，收集回复返回给客户端 |
| **Tokenizer** | `num_tokenizer` | 把文本编码为 token ids，转发给 scheduler |
| **Detokenizer** | 1 | 把 scheduler 返回的 token ids 解码为增量文本，转发给 frontend |
| **Scheduler** | `world_size`（TP 数） | 核心调度与计算：管理请求队列、KV 缓存、prefill/decode 调度、模型前向计算 |

### 2.3 消息流

```
客户端 HTTP 请求
    │
    ▼
FrontendManager.new_user() → 分配 uid
    │
    ▼ TokenizeMsg (ZMQ)
Tokenizer 进程 → 编码为 token ids
    │
    ▼ UserMsg (ZMQ)
Scheduler 进程 → 调度 + 模型前向计算
    │
    ▼ DetokenizeMsg (ZMQ)
Detokenizer 进程 → 解码为增量文本
    │
    ▼ UserReply (ZMQ)
FrontendManager → SSE 流式返回给客户端
```

### 2.4 进程启动流程

启动入口在 `server/launch.py`：

1. 解析命令行参数（`ServerArgs`）
2. 设置 multiprocessing 启动方式为 `spawn`
3. 依次启动子进程：
   - `world_size` 个 Scheduler 进程（每个 TP rank 一个）
   - 1 个 Detokenizer 进程
   - `num_tokenizer` 个 Tokenizer 进程
4. 通过 `ack_queue` 等待所有子进程报告就绪（共 `num_tokenizers + 2` 个 ack）
5. 启动 FastAPI/Uvicorn 对外服务

---

## 3. 请求生命周期

以一个 `/v1/chat/completions` 请求为例，完整生命周期如下：

### 3.1 接收请求

```
客户端 POST /v1/chat/completions
    → FastAPI 路由 v1_completions()
    → FrontendManager.new_user() 分配 uid
    → 构造 TokenizeMsg，通过 ZMQ 发送给 Tokenizer 进程
```

### 3.2 分词

```
Tokenizer 进程收到 TokenizeMsg
    → 用 HuggingFace tokenizer 把文本编码为 token ids
    → 构造 UserMsg(uid, input_ids, sampling_params)
    → 通过 ZMQ 发送给 Scheduler 进程
```

### 3.3 调度与计算

```
Scheduler 进程收到 UserMsg
    → 加入待处理队列
    → PrefillManager 调度：
        1. RadixPrefixCache.match_prefix() 匹配已有前缀
        2. 分配 KV 缓存页（PageTable.alloc()）
        3. 按 token 预算分批 prefill（Chunked Prefill）
    → Engine 执行模型前向计算：
        1. prepare_metadata（生成注意力元数据）
        2. 逐层前向：embedding → layers → norm → lm_head
        3. 每层：RMSNorm → attention → MLP（MoE 可选）
        4. 采样（temperature/top-k/top-p）
    → DecodeManager 调度后续 decode 步骤
    → 每步生成一个 token，写入 KV 缓存
    → 构造 DetokenizeMsg(uid, next_token, incremental_output)
    → 通过 ZMQ 发送给 Detokenizer 进程
```

### 3.4 解码与返回

```
Detokenizer 进程收到 DetokenizeMsg
    → 用 surr_offset/read_offset 技巧计算增量文本
    → 构造 UserReply(uid, incremental_output, finished)
    → 通过 ZMQ 发送给 FrontendManager

FrontendManager.listen() 收到 UserReply
    → 按 uid 分发到 ack_map
    → 唤醒等待的 wait_for_ack() 生成器
    → stream_chat_completions() 编码为 SSE chunk
    → 流式返回给客户端
```

### 3.5 请求完成

当 scheduler 采样到 EOS token 或达到 `max_tokens` 时，标记 `finished=True`。
FrontendManager 收到 finished 消息后，发送最终 chunk 并关闭流。

---

## 4. 代码目录逐模块详解

### 4.1 目录结构

```
mini-sglang/
├── python/minisgl/          # 核心代码
│   ├── __main__.py           # 入口：python -m minisgl
│   ├── core.py               # 全局上下文、Batch、SamplingParams 等核心数据结构
│   ├── env.py                # 环境变量配置
│   ├── shell.py              # 交互式 shell
│   ├── engine/               # 计算引擎
│   │   ├── engine.py         # Engine：模型加载与前向计算
│   │   ├── config.py         # EngineConfig
│   │   ├── graph.py          # CUDA Graph 捕获与回放
│   │   └── sample.py         # 采样逻辑（temperature/top-k/top-p）
│   ├── scheduler/             # 调度器
│   │   ├── scheduler.py       # Scheduler：主调度循环
│   │   ├── prefill.py         # PrefillManager + PrefillAdder：prefill 调度
│   │   ├── decode.py          # DecodeManager：decode 调度
│   │   ├── cache.py           # 缓存管理（前缀缓存 + 页表）
│   │   ├── table.py           # PageTable：物理页分配与回收
│   │   ├── io.py              # ZMQ 消息收发
│   │   ├── config.py          # SchedulerConfig
│   │   └── utils.py           # 辅助函数
│   ├── models/                # 模型定义
│   │   ├── base.py            # BaseLLMModel 基类
│   │   ├── config.py          # ModelConfig
│   │   ├── llama.py           # Llama 模型
│   │   ├── qwen3.py           # Qwen3 模型
│   │   ├── qwen3_moe.py       # Qwen3-MoE 模型
│   │   ├── weight.py          # 权重加载（safetensors 流式读取 + TP 切分）
│   │   ├── register.py        # 模型注册表
│   │   └── utils.py           # 辅助函数
│   ├── layers/                # 神经网络层
│   │   ├── base.py            # StateLessOP / Module 基类
│   │   ├── linear.py          # 线性层（列并行 / 行并行）
│   │   ├── norm.py            # RMSNorm + 融合残差
│   │   ├── embedding.py       # Token embedding
│   │   ├── rotary.py          # 旋转位置编码（RoPE）
│   │   ├── attention.py       # 注意力层
│   │   ├── activation.py     # 激活函数（SiLU 等）
│   │   └── moe.py             # MoE MLP 层
│   ├── kvcache/               # KV 缓存
│   │   ├── base.py            # BaseKVCachePool + BasePrefixCache 抽象接口
│   │   ├── mha_pool.py        # MHAKVCache：物理分页存储
│   │   ├── radix_cache.py     # RadixPrefixCache：Radix 树前缀复用
│   │   └── naive_cache.py     # NaivePrefixCache：无复用占位实现
│   ├── attention/             # 注意力后端
│   │   ├── base.py            # BaseAttnBackend + HybridBackend
│   │   ├── fa.py              # FlashAttention 后端（sgl_kernel）
│   │   ├── fi.py              # FlashInfer 后端
│   │   └── utils.py           # CUDA Graph 捕获数据基类
│   ├── distributed/           # 分布式通信
│   │   ├── impl.py            # 通信实现（Torch / PyNCCL）
│   │   └── info.py            # DistributedInfo（TP rank/size）
│   ├── message/               # 消息定义
│   │   ├── frontend.py        # 前端消息（TokenizeMsg 等）
│   │   ├── backend.py         # 后端消息（UserMsg / DetokenizeMsg 等）
│   │   ├── tokenizer.py       # Tokenizer 消息
│   │   └── utils.py           # 序列化工具
│   ├── server/                # 服务器
│   │   ├── api_server.py     # FastAPI 应用 + FrontendManager
│   │   ├── launch.py          # 进程启动入口
│   │   └── args.py            # 参数解析
│   ├── tokenizer/             # 分词器
│   │   ├── tokenize.py        # Tokenizer 工作进程
│   │   ├── detokenize.py      # Detokenizer 工作进程
│   │   └── server.py         # Tokenizer 服务
│   ├── llm/                   # 离线推理
│   │   └── llm.py             # LLM 类（离线模式入口）
│   ├── kernel/                # CUDA 内核封装
│   └── utils/                 # 通用工具
├── docs/                      # 英文文档
├── benchmark/                 # 基准测试
├── tests/                     # 测试
└── pyproject.toml             # 项目配置
```

### 4.2 核心模块详解

#### core.py — 全局上下文与核心数据结构

| 类/函数 | 职责 |
|---------|------|
| `GlobalContext` | 全局单例，持有 KV 缓存、页表、注意力后端、模型配置等，通过 `get_global_ctx()` 获取 |
| `Batch` | 一次前向计算的批次，包含请求列表、输入输出位置、注意力元数据等 |
| `SamplingParams` | 采样参数（temperature、top_k、top_p、max_tokens、ignore_eos） |
| `Req` | 单个请求的运行时状态（input_ids、output_ids、缓存位置等） |

**设计要点**：模型 `forward` 不显式传 batch，统一从 `get_global_ctx()` 获取 batch、attn_backend、moe_backend，减少参数传递。

#### engine/ — 计算引擎

| 文件 | 关键类/函数 | 职责 |
|------|-------------|------|
| `engine.py` | `Engine` | 模型加载、前向计算（`forward` / `forward_batch`）、权重加载 |
| `graph.py` | `CudaGraphRunner` | CUDA Graph 捕获与回放管理 |
| `sample.py` | `sample()` | 采样实现：temperature 缩放 → top-k → top-p → 多项式采样 |
| `config.py` | `EngineConfig` | 引擎配置（模型路径、dtype 等） |

**前向计算流程**：
1. `prepare_batch()`：准备输入张量、位置编码、注意力元数据
2. 逐层前向：`RMSNormFused`（残差+归一化）→ `Attention`（含 KV 写入）→ `MLP`/`MoEMLP`
3. 最终 norm + `lm_head` → logits
4. `sample()` 采样

#### scheduler/ — 调度器

| 文件 | 关键类/函数 | 职责 |
|------|-------------|------|
| `scheduler.py` | `Scheduler` | 主调度循环 `run_forever()`，协调 prefill/decode/IO |
| `prefill.py` | `PrefillManager`, `PrefillAdder` | prefill 调度：token 预算控制、在飞 decode 预留 |
| `decode.py` | `DecodeManager` | decode 调度：按 uid 排序、batch 管理 |
| `cache.py` | `CacheManager` | 前缀缓存管理：匹配、插入、驱逐 |
| `table.py` | `PageTable` | 物理页分配与回收（buddy allocator） |
| `io.py` | `SchedulerIO` | ZMQ 消息收发 |

**调度策略**：
- **Prefill 优先**：有 prefill 请求时优先处理，但受 token 预算限制
- **Token 预算**：`prefill_budget` 限制单批 prefill 的 token 数，避免 OOM
- **在飞 decode 预留**：prefill 时预留足够空间给正在进行的 decode 请求
- **Overlap Scheduling**：`overlap_loop()` 用双 stream 重叠调度与计算

#### models/ — 模型定义

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `base.py` | `BaseLLMModel` | 模型基类：定义 `forward`、`load_weight` 接口 |
| `llama.py` | `LlamaModel` | Llama 架构实现 |
| `qwen3.py` | `Qwen3Model` | Qwen3 架构（带 QK norm） |
| `qwen3_moe.py` | `Qwen3MoeModel` | Qwen3 MoE 架构 |
| `weight.py` | `load_weight()` | 流式读取 safetensors，边 TP 切分边合并 |

**权重加载流程**（`weight.py`）：
1. 流式读取 safetensors 文件（避免一次性加载全部权重）
2. 对每个权重张量做 TP 切分：
   - 列并行权重（qkv_proj、gate_up_proj）：沿输出维度切分
   - 行并行权重（o_proj、down_proj）：沿输入维度切分
3. 合并相关权重：q/k/v → qkv_proj，gate/up → gate_up_proj
4. MoE 专家：按索引收集后 `stack` 打包

#### layers/ — 神经网络层

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `base.py` | `StateLessOP`, `Module` | 层基类：无状态层不参与权重存取 |
| `linear.py` | `ColumnParallelLinear`, `RowParallelLinear` | 张量并行线性层 |
| `norm.py` | `RMSNormFused` | 融合残差加法与 RMSNorm，返回 `(x, residual)` |
| `embedding.py` | `VocabParallelEmbedding` | 词表并行 embedding |
| `rotary.py` | `RotaryEmbedding` | 旋转位置编码 |
| `attention.py` | `Attention` | 注意力层（调用 attn_backend） |
| `moe.py` | `MoEMLP` | 混合专家 MLP |

**残差流设计**：
`RMSNormFused` 融合了残差加法与归一化，`forward` 返回 `(x, residual)` 元组，贯穿所有层。这样每层只需一次归一化操作，减少 kernel launch 开销。

#### kvcache/ — KV 缓存（两层分离设计）

| 文件 | 关键类 | 层次 | 职责 |
|------|--------|------|------|
| `base.py` | `BaseKVCachePool`, `BasePrefixCache` | 抽象 | 定义物理存储与逻辑前缀的接口 |
| `mha_pool.py` | `MHAKVCache` | 物理 | 分页存储 KV 张量，TP 切分 KV 头 |
| `radix_cache.py` | `RadixPrefixCache` | 逻辑 | Radix 树前缀复用，按页对齐，LRU 驱逐 |
| `naive_cache.py` | `NaivePrefixCache` | 逻辑 | 无复用占位实现 |

**物理存储**（`MHAKVCache`）：
- 缓存张量形状：`(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)`
- 第 0 维：2，分别对应 K 和 V
- KV 头按 TP 切分，KV 头数 < TP 数时允许复制

**逻辑前缀**（`RadixPrefixCache`）：
- 用 Radix 树组织已缓存的前缀，公共前缀只存一份
- 插入时按页对齐（`align_down`），只缓存完整页
- LRU 驱逐：用最小堆按时间戳排序，优先驱逐最久未访问的叶子节点
- 引用计数保护：`ref_count > 0` 的节点不可驱逐

#### attention/ — 注意力后端（可插拔设计）

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `base.py` | `BaseAttnBackend`, `HybridBackend` | 抽象接口 + 混合后端 |
| `fa.py` | `FlashAttentionBackend` | sgl_kernel 分页 flash_attn |
| `fi.py` | `FlashInferBackend` | FlashInfer 分页注意力 |
| `utils.py` | `BaseCaptureData` | CUDA Graph 捕获数据基类 |

**混合后端**（`HybridBackend`）：
- 按 `batch.is_prefill` 自动切换 prefill/decode 后端
- prefill 和 decode 的最优实现往往不同（计算密集 vs 访存密集）

**FlashAttention 后端**（`fa.py`）：
- 依赖 `sgl_kernel.flash_attn.flash_attn_with_kvcache`
- 支持分页 KV 缓存（通过 page_table 参数）
- 根据 GPU 架构选择 FA v3/v4

**FlashInfer 后端**（`fi.py`）：
- 依赖 `flashinfer` 库
- 仅支持 `page_size=1`
- 使用 prefill/decode 两个 wrapper，plan 一次后复用
- 支持 Tensor Core（GQA 比例 >= 4 时自动启用）

#### distributed/ — 分布式通信

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `impl.py` | `DistributedCommunicator` | 通信器：插件式，默认 Torch，可追加 PyNCCL |
| `info.py` | `DistributedInfo` | TP rank/size 信息 |

**张量并行设计**：
- **列并行**（`ColumnParallelLinear`）：qkv_proj、gate_up_proj 沿输出维度切分，各卡独立计算
- **行并行**（`RowParallelLinear`）：o_proj、down_proj 沿输入维度切分，各卡计算后 `all_reduce`
- KV 头数 < TP 数时允许复制（`allow_replicate=True`）

#### server/ — 服务器

| 文件 | 关键类/函数 | 职责 |
|------|-------------|------|
| `api_server.py` | `FrontendManager`, `app` | FastAPI 应用、HTTP 路由、ZMQ 桥接 |
| `launch.py` | `launch_server()` | 进程启动入口 |
| `args.py` | `ServerArgs`, `parse_args()` | 参数解析 |

**API 端点**：
- `POST /generate`：简单生成接口，返回 SSE 流
- `POST /v1/chat/completions`：OpenAI 兼容聊天补全，支持流式/非流式
- `GET /v1/models`：列出可用模型
- `GET /v1`：健康检查

#### llm/ — 离线推理

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `llm.py` | `LLM` | 离线推理入口，继承 Scheduler，单进程完成编码→调度→解码 |

`LLM` 类提供类似 HuggingFace transformers 的接口：
```python
llm = LLM(model_path="Qwen/Qwen3-8B")
results = llm.generate(["你好"], SamplingParams(max_tokens=100))
print(results[0]["text"])
```

#### tokenizer/ — 分词器

| 文件 | 关键函数 | 职责 |
|------|----------|------|
| `tokenize.py` | `tokenize_worker()` | Tokenizer 工作进程：文本→token ids |
| `detokenize.py` | `detokenize_worker()` | Detokenizer 工作进程：token ids→增量文本 |

**增量解码技巧**：
Detokenizer 使用 `surr_offset` / `read_offset` 两个偏移量输出增量文本，避免输出半个词或半个 UTF-8 字符：
- `surr_offset`：已安全输出的位置（不会产生半个字符）
- `read_offset`：当前已解码的位置
- 每次只输出 `surr_offset` 到 `read_offset` 之间的安全文本

#### message/ — 消息定义

| 文件 | 关键类 | 职责 |
|------|--------|------|
| `frontend.py` | `TokenizeMsg`, `UserReply`, `BatchFrontendMsg` | 前端↔tokenizer 消息 |
| `backend.py` | `UserMsg`, `DetokenizeMsg`, `AbortMsg` | tokenizer↔scheduler 消息 |
| `tokenizer.py` | Tokenizer 相关消息 | detokenizer↔frontend 消息 |
| `utils.py` | `serialize_type()`, `deserialize_type()` | 序列化/反序列化（支持 1D tensor） |

---

## 5. 核心原理深入讲解

### 5.1 Radix Cache（基数树前缀复用）

#### 问题

多个请求常有公共前缀（如 system prompt），如果每个请求都从头计算 prefill，会浪费大量计算。

#### 解决方案

用 Radix 树（基数树）组织已缓存的 token 序列，公共前缀在树中只存一份。

#### 数据结构

```
                    root (ref_count=1, 永不驱逐)
                   /                        \
          [system prompt]              [其他前缀...]
          ref_count=2                  ref_count=0
              |
        [user message 1]
        ref_count=1
```

每个节点保存：
- `key`：token id 序列
- `value`：对应的物理缓存索引
- `ref_count`：引用计数（>0 不可驱逐）
- `timestamp`：最近访问时间（LRU 用）

#### 匹配流程（`_tree_walk`）

1. 从根节点开始，用 `key_fn(input_ids)` 查找子节点
2. 找到子节点后，用 `fast_compare_key` 计算公共前缀长度
3. 按页对齐（`align_down`），只复用完整页
4. 若部分匹配：在匹配处拆分节点（`split_at`），返回拆分后的节点
5. 更新访问时间戳

#### 插入流程（`insert_prefix`）

1. 按页对齐截断输入
2. 沿树匹配已有前缀
3. 未匹配部分作为新节点挂到树上
4. 新节点加入可驱逐区（`evictable_size += length`）

#### 驱逐流程（`evict`）

1. DFS 收集所有 `ref_count==0` 的叶子节点
2. 用最小堆按时间戳排序
3. 依次弹出最旧的叶子，释放其物理页
4. 若父节点因此变成可驱逐叶子，加入堆继续

### 5.2 Chunked Prefill（分块预填充）

#### 问题

长 prompt 的 prefill 会占用大量 GPU 资源，阻塞正在进行的 decode 请求，导致延迟增加。

#### 解决方案

将长 prefill 拆分为多个 chunk，每个 chunk 不超过 token 预算，与 decode 交替执行。

#### 实现（`PrefillManager` + `PrefillAdder`）

- `prefill_budget`：单批 prefill 的 token 上限
- `PrefillAdder` 逐个添加请求，累计 token 数不超过预算
- 预留在飞 decode 请求的空间：`reserved = sum(decode_lens)`
- 可用预算 = `prefill_budget - reserved`

#### 调度优先级

1. 有 prefill 请求 → 优先 prefill（受预算限制）
2. 无 prefill → 执行 decode
3. 两者交替进行，保证 decode 不被长时间阻塞

### 5.3 Overlap Scheduling（重叠调度）

#### 问题

调度逻辑（Python 端：准备 batch、分配缓存等）与 GPU 计算是串行的，GPU 在调度期间空闲。

#### 解决方案

用双 CUDA Stream 重叠调度与计算：

```
Stream 1: [计算 batch N]          [计算 batch N+1]
Stream 2:          [调度 batch N+1]          [调度 batch N+2]
```

#### 实现（`overlap_loop`）

1. Stream 1 执行当前 batch 的模型前向计算
2. 同时 Stream 2 准备下一个 batch（调度、分配缓存、准备元数据）
3. 计算完成后切换到已准备好的下一个 batch
4. 循环往复

### 5.4 Tensor Parallelism（张量并行）

#### 列并行（Column Parallel）

适用于权重矩阵的输出维度可以切分的层（qkv_proj、gate_up_proj）：

```
权重 W: [out_dim, in_dim]
切分: W = [W_0 | W_1 | ... | W_{tp-1}]  (沿 out_dim 切)
各卡: Y_i = X @ W_i  (独立计算，无需通信)
```

#### 行并行（Row Parallel）

适用于权重矩阵的输入维度可以切分的层（o_proj、down_proj）：

```
权重 W: [out_dim, in_dim]
切分: W = [W_0; W_1; ...; W_{tp-1}]  (沿 in_dim 切)
各卡: Y_i = X_i @ W_i  (X 也切分)
合并: Y = sum(Y_0, Y_1, ..., Y_{tp-1})  (all_reduce)
```

#### KV 头复制

当 KV 头数 < TP 数时（如 GQA），部分 GPU 没有独立的 KV 头，此时允许复制 KV 头（`allow_replicate=True`），保证每张卡都有 KV 头可用。

### 5.5 CUDA Graph（计算图捕获）

#### 问题

Decode 阶段每步只生成 1 个 token，计算量小但 Python 调度开销大，GPU 利用率低。

#### 解决方案

捕获 decode 阶段的计算图，后续直接回放，跳过 Python 调度。

#### 实现（`engine/graph.py` + `attention/`）

1. **初始化**（`init_capture_graph`）：为每个 batch 大小分配静态缓冲
2. **捕获**（`prepare_for_capture`）：用假数据运行一次，捕获计算图
3. **回放**（`prepare_for_replay`）：把真实数据拷贝到静态缓冲，回放计算图

**关键约束**：回放时张量地址必须不变，因此用静态缓冲 + `copy_` 更新数据。

### 5.6 Attention 后端（可插拔设计）

#### 抽象接口（`BaseAttnBackend`）

```python
class BaseAttnBackend(ABC):
    def forward(self, q, k, v, layer_id, batch) -> Tensor  # 注意力计算
    def prepare_metadata(self, batch) -> None               # 生成元数据
    def init_capture_graph(self, max_seq_len, bs_list)       # CUDA Graph 初始化
    def prepare_for_capture(self, batch)                     # 捕获前准备
    def prepare_for_replay(self, batch)                      # 回放前准备
```

#### 混合后端（`HybridBackend`）

按 `batch.is_prefill` 自动切换：
- Prefill：用 FlashAttention 后端（支持分页，计算密集型优化好）
- Decode：用 FlashInfer 后端（访存密集型优化好，支持 Tensor Core）

#### 两种后端对比

| 特性 | FlashAttention (sgl_kernel) | FlashInfer |
|------|---------------------------|------------|
| 依赖 | sgl_kernel | flashinfer |
| page_size | 任意 | 仅 1 |
| FA 版本 | v3/v4（按 GPU 架构） | fa2 |
| CUDA Graph | ✅ | ✅ |
| Tensor Core | 内部处理 | GQA >= 4 时启用 |

### 5.7 KV Cache 管理（两层分离设计）

#### 物理层（`MHAKVCache`）

- 负责实际的 KV 张量存储与读写
- 分页管理：`shape = (2, layers, pages, page_size, kv_heads, head_dim)`
- `store_kv()`：把新算出的 KV 写入指定物理位置
- `k_cache(layer_id)` / `v_cache(layer_id)`：返回某层的缓存视图

#### 逻辑层（`RadixPrefixCache`）

- 负责前缀的匹配、插入、驱逐
- 通过缓存句柄（`RadixCacheHandle`）把逻辑前缀映射到物理索引
- 不直接操作物理存储，只管理"哪些前缀在哪个位置"

#### 页表（`PageTable`）

- 管理物理页的分配与回收
- 每个请求有一个 `table_idx`，对应页表中的一行
- 页表记录每个序列的页号序列

#### 工作流程

```
请求到来
    → RadixPrefixCache.match_prefix()  匹配已有前缀
    → PageTable.alloc()                分配新页
    → 模型前向计算
    → MHAKVCache.store_kv()            写入物理缓存
    → RadixPrefixCache.insert_prefix() 更新逻辑缓存
```

### 5.8 采样（Sampling）

#### 流程（`engine/sample.py`）

```
logits (raw scores)
    │
    ▼ temperature 缩放
logits = logits / temperature
    │
    ▼ top-k 过滤（保留最大的 k 个）
logits[top_k_indices] 保留，其余设为 -inf
    │
    ▼ top-p（核采样）过滤
按概率降序累加，超过 p 时截断
    │
    ▼ 多项式采样
softmax → 按概率分布随机选一个 token
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 1.0 | 温度：越高越随机，越低越确定 |
| `top_k` | -1（禁用） | 只从概率最高的 k 个 token 中采样 |
| `top_p` | 1.0（禁用） | 核采样：从累积概率达到 p 的最小集合中采样 |
| `max_tokens` | - | 最大生成长度 |
| `ignore_eos` | False | 是否忽略 EOS token（不停止生成） |

---

## 6. 启动与使用方式

### 6.1 安装

```bash
# 克隆项目
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 安装依赖
pip install -e .
```

### 6.2 在线服务模式

```bash
# 基本启动
python -m minisgl --model_path Qwen/Qwen3-8B

# 指定端口和 TP 数
python -m minisgl --model_path Qwen/Qwen3-8B --port 30000 --tp 2

# 使用交互式 shell
python -m minisgl --model_path Qwen/Qwen3-8B --run-shell

# 使用 dummy 权重（测试用，不需要真实模型）
python -m minisgl --model_path Qwen/Qwen3-8B --use-dummy-weight
```

### 6.3 API 调用

#### 简单生成

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，请介绍一下自己", "max_tokens": 100}'
```

#### OpenAI 兼容聊天补全（流式）

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

#### OpenAI 兼容聊天补全（非流式）

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
    "stream": false
  }'
```

#### Python 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### 6.4 离线推理模式

```python
from minisgl import LLM, SamplingParams

llm = LLM(model_path="Qwen/Qwen3-8B")
results = llm.generate(
    prompts=["你好", "请写一首诗"],
    sampling_params=SamplingParams(max_tokens=100, temperature=0.7),
)
for r in results:
    print(r["text"])
```

### 6.5 交互式 Shell

```bash
python -m minisgl --model_path Qwen/Qwen3-8B --run-shell
```

Shell 内支持：
- 直接输入文本进行对话
- `/reset`：清空对话历史
- `/exit`：退出

### 6.6 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 模型路径（HuggingFace 格式） |
| `--port` | 30000 | API 服务端口 |
| `--tp` | 1 | 张量并行数 |
| `--num_tokenizer` | 1 | Tokenizer 进程数 |
| `--use_dummy_weight` | False | 使用随机权重（测试用） |
| `--run_shell` | False | 启动交互式 shell |
| `--silent_output` | False | 静默输出（禁用 INFO 日志） |

---

## 7. 关键代码阅读路径建议

### 7.1 推荐阅读顺序

按以下顺序阅读可以快速建立对整个系统的理解：

```
1. core.py                    ← 理解全局上下文与核心数据结构
   ↓
2. server/launch.py           ← 理解进程启动与拓扑
   ↓
3. server/api_server.py       ← 理解 HTTP 接口与消息桥
   ↓
4. message/                   ← 理解进程间消息格式
   ↓
5. scheduler/scheduler.py     ← 理解主调度循环
   ↓
6. scheduler/prefill.py       ← 理解 prefill 调度策略
   ↓
7. scheduler/decode.py        ← 理解 decode 调度策略
   ↓
8. engine/engine.py           ← 理解模型前向计算
   ↓
9. models/base.py             ← 理解模型基类与权重加载
   ↓
10. layers/                   ← 理解张量并行层与残差流
   ↓
11. kvcache/                  ← 理解分页 KV 缓存与 Radix 树
   ↓
12. attention/                ← 理解可插拔注意力后端
   ↓
13. distributed/              ← 理解张量并行通信
   ↓
14. tokenizer/                ← 理解增量解码
   ↓
15. llm/llm.py                ← 理解离线推理入口
```

### 7.2 按主题阅读

#### 想理解"请求怎么流转"

```
api_server.py (v1_completions)
  → message/frontend.py (TokenizeMsg)
  → tokenizer/tokenize.py (tokenize_worker)
  → message/backend.py (UserMsg)
  → scheduler/scheduler.py (run_forever)
  → scheduler/io.py (SchedulerIO)
  → message/backend.py (DetokenizeMsg)
  → tokenizer/detokenize.py (detokenize_worker)
  → message/frontend.py (UserReply)
  → api_server.py (stream_chat_completions)
```

#### 想理解"模型怎么计算"

```
engine/engine.py (forward_batch)
  → models/base.py (BaseLLMModel.forward)
  → layers/norm.py (RMSNormFused)
  → layers/attention.py (Attention)
  → attention/base.py (HybridBackend.forward)
  → attention/fa.py 或 fi.py
  → layers/linear.py (ColumnParallelLinear / RowParallelLinear)
  → layers/moe.py (MoEMLP, 仅 MoE 模型)
  → engine/sample.py (sample)
```

#### 想理解"KV 缓存怎么管理"

```
kvcache/base.py (接口定义)
  → kvcache/mha_pool.py (物理分页存储)
  → kvcache/radix_cache.py (Radix 树前缀复用)
  → scheduler/table.py (PageTable 页分配)
  → scheduler/cache.py (CacheManager 缓存管理)
```

#### 想理解"张量并行怎么工作"

```
distributed/info.py (DistributedInfo)
  → distributed/impl.py (DistributedCommunicator)
  → layers/linear.py (ColumnParallelLinear / RowParallelLinear)
  → models/weight.py (load_weight, TP 切分)
  → kvcache/mha_pool.py (KV 头切分)
```

#### 想理解"CUDA Graph 怎么用"

```
engine/graph.py (CudaGraphRunner)
  → attention/base.py (init_capture_graph / prepare_for_capture / prepare_for_replay)
  → attention/fa.py (FACaptureData)
  → attention/fi.py (FICaptureData)
```

### 7.3 已加中文注释的文件清单

以下 26 个文件已添加详细的中文注释（模块注释 + 类注释 + 方法注释 + 关键行行内注释）：

**核心引擎与调度器（5 个）**：
- `python/minisgl/core.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/scheduler/prefill.py`
- `python/minisgl/scheduler/decode.py`

**模型与层（10 个）**：
- `python/minisgl/models/base.py`
- `python/minisgl/models/llama.py`
- `python/minisgl/models/qwen3.py`
- `python/minisgl/models/weight.py`
- `python/minisgl/layers/base.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/layers/norm.py`
- `python/minisgl/layers/rotary.py`
- `python/minisgl/layers/attention.py`
- `python/minisgl/layers/embedding.py`

**基础设施（11 个）**：
- `python/minisgl/kvcache/base.py`
- `python/minisgl/kvcache/mha_pool.py`
- `python/minisgl/kvcache/naive_cache.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/attention/base.py`
- `python/minisgl/attention/fa.py`
- `python/minisgl/attention/fi.py`
- `python/minisgl/distributed/impl.py`
- `python/minisgl/server/api_server.py`
- `python/minisgl/server/launch.py`
- `python/minisgl/llm/llm.py`

---

## 已知限制

- 已加注释的文件行尾从 CRLF 转为 LF（`write` 工具副作用），`git diff` 会显示整文件变更。代码逻辑未改变，已通过 `py_compile` 语法验证。
- 原英文 `README.md` 保持不动。

---

*本文档基于源码阅读生成，如有与代码不一致之处以代码为准。*
