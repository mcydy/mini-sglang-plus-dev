import functools
from typing import Dict, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.utils import div_ceil
from minisgl.utils import div_even, init_logger

logger = init_logger(__name__)

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


# 把"token 路由到不同专家、各专家算完再按权重合并"
def fused_experts_impl(
    hidden_states: torch.Tensor,          # [M, K] 输入隐藏状态，M=token数，K=hidden_size
    w1: torch.Tensor,                     # [E, N, K]   gate_up_proj，N = 2 * intermediate_size
    w2: torch.Tensor,                     # [E, K, I]   down_proj，I = intermediate_size
    topk_weights: torch.Tensor,           # [M, top_k]  每个 token 对各专家的路由权重
    topk_ids: torch.Tensor,               # [M, top_k]  每个 token 选中的专家编号
    activation: str = "silu",             # 激活函数类型，SwiGLU 用 "silu"
    apply_router_weight_on_input: bool = False,  # 路由权重乘在哪一侧的开关
) -> torch.Tensor:
    # 延迟导入：避免启动时就加载 Triton kernel（编译较慢），真正用到 MoE 时才 import
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    # ==================================================================
    # 【块 1】参数校验
    # ==================================================================
    # padded_size：为量化权重预留的钩子（例如 fp8 权重尾部会填充 scale），
    # 当前非量化路径恒为 0，所以 K 必须严格等于 w1.shape[2]

    if not torch.cuda.is_current_stream_capturing():
        logger.debug(f"token 大小 = {topk_ids.shape}")
        logger.debug(f"token 0选中的专家编号 = {topk_ids[0].tolist()}")
        logger.debug(f"token 0各个专家的权重 = {topk_weights[0].tolist()}")

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    # Triton kernel 要求所有输入输出都是连续内存，否则访存会错乱
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # ==================================================================
    # 【块 2】取关键维度
    # ==================================================================
    num_tokens, _ = hidden_states.shape    # M：本次参与计算的 token 数
    E, N, _ = w1.shape                     # E=专家数(64)，N=w1的第1维=2*intermediate(2048)
    M = num_tokens

    # ==================================================================
    # 【块 3】自动选择 Triton kernel 配置（BLOCK_SIZE_M/N/K、num_warps 等）
    # ==================================================================
    # 根据权重形状和 top_k 查表/实测得到最优分块参数，直接影响 GEMM 性能
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    # ==================================================================
    # 【块 4】中间缓存分配（cache1 与 cache3 复用同一块显存）
    # ==================================================================
    # 生命周期分析：
    #   GEMM1 写 cache1 → 激活 读 cache1 写 cache2 → GEMM2 读 cache2 写 cache3
    #   cache1 在激活读完后即作废，所以 cache3 可以覆盖同一段内存，省一半显存
    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # cache1：GEMM1 的输出，[M, top_k, N]，N = 2*intermediate（gate 和 up 拼在一起）
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    # cache2：激活后的输出，[M*top_k, N//2]，注意是【展平】的二维，且单独分配
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # cache3：GEMM2 的输出，[M, top_k, K]，与 cache1 共用 cache 起始地址
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    # ==================================================================
    # 【块 5】输出原地写回 + 分块循环骨架
    # ==================================================================
    # 注意：这里是【别名】不是拷贝，结果会直接覆盖 hidden_states
    # 安全性由执行顺序保证：GEMM1 先读完 hidden_states，归约最后才写
    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    # 分块处理骨架（上游 sglang 会把长 prefill 切成多块以控制峰值显存）
    # 当前实现只有一个 chunk：从头到尾
    begin_token_idx, end_token_idx = 0, num_tokens

    # 按实际 token 数裁剪缓存视图，并重算该规模下的最优 kernel 配置
    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    # ==================================================================
    # 【块 6】块对齐排序 —— MoE 最核心的预处理
    # ==================================================================
    # 问题：top-k 路由后，同一专家的 token 在内存中是离散的，无法直接做 GEMM
    # 解法：按专家分组排序 + 每段补齐到 BLOCK_SIZE_M 的整数倍
    # 产出：
    #   sorted_token_ids     排序后的 token 索引（含 padding 的无效行）
    #   expert_ids           每个块对应哪个专家
    #   num_tokens_post_padded  补齐后的总 token 数
    # 效果：把"稀疏离散的路由"变成"规则的密集分块 GEMM"
    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    # ==================================================================
    # 【块 7】GEMM1：gate_up 投影
    # ==================================================================
    # 计算：cache1[m, j] = hidden[m] @ W1[expert_id]ᵀ
    # 即一次把 gate 和 up 两个投影都算出来（w1 的第 1 维 N = 2*intermediate，
    # 前一半是 gate 的输出，后一半是 up 的输出）
    # 倒数第 3 个参数 mul_routed_weight=apply_router_weight_on_input（默认 False）
    #   → 这里【不乘】路由权重
    # 倒数第 2 个参数 topk_num=8：cache1 仍是 [M, top_k, N] 的三维结构
    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,   # mul_routed_weight = False
        topk_ids.shape[1],              # topk_num = 8
        config,
        compute_type=compute_type,
    )

    # ==================================================================
    # 【块 8】SwiGLU 激活：silu(gate) * up
    # ==================================================================
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    # N 本身就是 2*intermediate，所以直接 view(-1, N) 即可
    #   view(-1, N)     → [M*top_k, 2048] = [gate(1024), up(1024)]
    #   silu_and_mul    → [M*top_k, 1024] = silu(gate) * up
    # view_as 是保险：形状不匹配会立刻报错，不会静默算错
    act_out = FN_MAP[activation](intermediate_cache1.view(-1, N))
    intermediate_cache2.copy_(act_out.view_as(intermediate_cache2))
    del act_out

    # ==================================================================
    # 【块 9】GEMM2：down 投影（在此处乘路由权重）
    # ==================================================================
    # 计算：cache3[m, j] = cache2[m, j] @ W2[expert_id]ᵀ
    # 关键：mul_routed_weight = not apply_router_weight_on_input（默认 True）
    #   与 GEMM1 的 False 形成【取反配对】，保证路由权重在整个流程中只乘一次
    #   - 两次都乘 → 权重变成 w²，结果错
    #   - 两次都不乘 → 丢失路由权重，退化成等权平均
    # topk_num 传 1：cache2 已展平成 [M*8, 1024]，每行是一条独立样本
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,   # mul_routed_weight = True ★
        1,                                  # topk_num = 1
        config,
        compute_type=compute_type,
    )

    # ==================================================================
    # 【块 10】归约输出：对 top_k 维求和
    # ==================================================================
    # [M, top_k, K] → [M, K]
    # 因为 GEMM2 已经乘过路由权重，这里【只做加法】，不能重复乘
    # 结果直接写回 hidden_states（原地）
    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
