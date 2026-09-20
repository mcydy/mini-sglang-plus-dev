// ============================================================================
// 文件：index.cu —— Embedding 查表（gather）kernel
//
// 【一句话功能】output[i] = weight[indices[i]]
//   即 PyTorch 的 F.embedding(indices, weight)，这里手写 CUDA 实现。
//
// 【它融合/替代了什么】
//   普通版（无 mask）替代：F.embedding 一次调用
//   masked 版则是本文件的重点，它把 VocabParallelEmbedding（词表并行）的四步
//   融合成一个 kernel，消除了 3 个中间张量和 4 次 kernel launch：
//
//     PyTorch 写法（nano-vLLM 里就是这么写的）：
//       mask = (x >= start) & (x < end)      ← kernel 1，中间张量 ①（bool）
//       x    = mask * (x - start)            ← kernel 2/3，中间张量 ②
//       y    = F.embedding(x, weight)        ← kernel 4
//       y    = mask.unsqueeze(1) * y         ← kernel 5，中间张量 ③
//       dist.all_reduce(y)                   ← 通信（本文件不含）
//
//     本文件 masked 版：
//       pos = indices[..] - start            ← 寄存器里算
//       if (pos < length) 拷数据 else 写 0   ← 同一个 kernel 内分支
//     → 1 次 launch，0 个中间张量，mask 判断完全在寄存器中完成
//
// 【为什么值得手写】
//   1. kElementSize 是编译期常量 → 循环完全展开 + 定长 16B 向量化拷贝
//   2. kNumSplits 可调：几个 warp 合作搬一个条目，适配不同 hidden_size
//   3. PDL：让下一个 kernel 提前 launch，吃掉 launch 间隙
//   4. 融合越界判断，省掉 mask 相关的一串小算子
// ============================================================================

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>   // warp::copy / warp::reset —— warp 级向量化搬运原语

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <bit>       // std::has_single_bit
#include <concepts>  // std::integral（约束 T 必须是整数类型）
#include <cstddef>
#include <cstdint>

namespace {

// ============================================================================
// kernel 参数结构体（按值传参，配合 __grid_constant__ 放常量内存）
// ============================================================================
struct IndexKernelParams {
  void *__restrict__ output;       // 输出 [L, D]，每张 i 行 = 第 i 个 token 的向量
  const void *__restrict__ weight; // 权重表 [vocab, D]，查表的源
  const void *__restrict__ indice; // 索引 [L]，每个 token 的 id
  std::size_t num_warps;           // 总共启动的 warp 数 = kNumSplits * L
};

// masked 版在基础参数上多了 [start, start+length) 这个合法区间
struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;   // 本 rank 词表起始（= vocab_start_idx）
  std::size_t length;  // 本 rank 词表长度（= 本地行数）
};

// ============================================================================
// 模板参数一览
//   kNumThreads     : 每个 block 的线程数（默认 128 = 4 个 warp）
//   kMaxOccupancy   : 每个 SM 最多几个 block（__launch_bounds__ 第二参数）
//                     纯访存 kernel 限制占用率，让访存更有序
//   kUsePDL         : 是否启用 Programmatic Dependent Launch
//   kElementSize    : ★ 一个条目的字节数 = D * dtype_bytes
//                     编译期常量 → 循环展开 + 定长向量化，这是性能关键
//   kNumSplits      : ★ 几个 warp 合作搬一个条目（必须是 2 的幂）
//                     element_size 大时用多个 warp 并行搬，提高并行度
//   T               : 索引类型 int32 或 int64（int32 省一半索引带宽）
// ============================================================================
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const __grid_constant__ IndexKernelParams params) {
  using namespace device;

  constexpr auto kSize = kElementSize;              // 一个条目总字节数
  constexpr auto kSizePerWarp = kSize / kNumSplits; // 每个 warp 负责搬多少字节
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  // ---- 编译期断言：把错误挡在编译阶段，而不是运行时 ----
  static_assert(kNumThreads % 32 == 0);          // 线程数必须是 warp 整数倍
  static_assert(std::has_single_bit(kNumSplits));// kNumSplits 必须是 2 的幂
  static_assert(kElementSize % kNumSplits == 0); // 必须能整除，否则分不均匀

  // 结构化绑定，解包参数
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);

  // ---- ★ 网格映射：全局 warp id（注意单位是 warp 不是 thread）----
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();   // 等待前一个 kernel 的数据就绪（若启用 PDL）

  if (warp_id < num_warps) {   // 尾块可能有空转 warp，需边界检查
    // 第几个 token：kNumSplits 个 warp 合作搬一个条目
    const auto pos = indices[warp_id / kNumSplits];

    // 目的地址：output 的第 (warp_id) 段
    //   注意按 kSizePerWarp 偏移，因为每个 warp 写自己那一段
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);

    // 源地址：weight 第 pos 行的起始 + 本 warp 负责的那段偏移
    //   pos * kSize               → 跳到第 pos 行
    //   (warp_id % kNumSplits) * kSizePerWarp → 行内偏移
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);

    // ★ 核心动作：warp 内 32 线程协作，向量化拷贝 kSizePerWarp 字节
    //   内部按 16B（uint4）对齐搬运，每线程分几条 —— 这就是 kElementSize
    //   编成模板参数换来的收益：编译期知道长度，完全展开
    warp::copy<kSizePerWarp>(dst, src);
  }

  PDL::launch<kUsePDL>();   // 提前触发下一个 kernel（若启用 PDL）
}

// ============================================================================
// masked 版：融合"减偏移 + 越界判断 + 查表 + 清零"
//
// 【融合点详解】对应 VocabParallelEmbedding 里这段 Python：
//     mask = (x >= start) & (x < end)
//     x    = mask * (x - start)
//     y    = F.embedding(x, weight)
//     y    = mask * y
// 下面几行 CUDA 一次搞定，mask 从不落显存，全在寄存器里：
//     pos = indices[..] - start      ← 对应 (x - start)
//     if (pos < length) copy         ← 对应 mask 判断 + F.embedding
//     else reset(dst)                ← 对应 mask * y（写零）
//
// 【写零为什么必要】
//   词表并行下，不属于本 rank 的 token 必须输出零向量，
//   这样后续 all_reduce 求和时才不会污染其他 rank 的结果
//   （同一 token 只有一个 rank 非零，求和 = 取那一行）
// ============================================================================
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    masked_index_kernel(
        const __grid_constant__ MaskedKernelParams mask_params) {
  using namespace device;

  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = mask_params;
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();

  //warp_id这个线程所属的 warp 在"整个 grid"里的全局编号。这个 kernel 的调度单位——注意：不是按线程分工，而是按 warp 分工。一个 warp 负责一块数据的搬运。

  // num_warps  = num_splits × L          // 一个 token 用 num_splits 个 warp 搬
  // warp_id / num_splits = 第几个 token   // 决定读哪个 index
  // warp_id % num_splits = 行内第几段     // 决定搬这一行的哪一段

  if (warp_id < num_warps) {
    // ★ 全局 token id → 本地行号（减去本 rank 的词表起始偏移）
    //   对应 Python 里的 (x - start)
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);

    // ★ 边界判断：落在 [0, length) 才查表，否则写零
    //   注意 pos 若被减成负数会绕回成巨大无符号数，
    //   所以这里必须用无符号比较（pos 是 size_t/无符号差）
    if (pos < length) {
      const auto src = pointer::offset(weight, pos * kSize,
                                       (warp_id % kNumSplits) * kSizePerWarp);
      warp::copy<kSizePerWarp>(dst, src);   // 命中 → 查表拷贝
    } else {
      warp::reset<kSizePerWarp>(dst);       // ★ 未命中 → 整块写 0
    }
  }

  PDL::launch<kUsePDL>();
}

// ============================================================================
// Host 侧启动器：做形状/设备校验，算出网格维度，按索引类型分派 kernel
// ============================================================================
template <std::size_t element_size,   // 每条目字节数 = D * dtype_bytes（编译期）
          std::size_t num_splits = 1, // 几个 warp 合作搬一个条目
          std::size_t num_threads = 128,   // block 大小
          std::size_t max_concurrency = 1, // 每 SM 最多几个 block
          bool use_pdl = false>            // 是否启用 PDL
struct IndexKernel {
  static void run(const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView output,
                  tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_opts) {
    using namespace host;

    // ---- 符号化变量：校验时推导实际值 ----
    auto D = SymbolicSize{"D"};              // embedding 维度
    auto L = SymbolicSize{"L"};              // 索引个数（token 数）
    auto device_ = SymbolicDevice{};
    auto weights_dtype_ = SymbolicDType{};
    auto indices_dtype_ = SymbolicDType{};

    // ---- 形状与设备校验（不匹配会直接报错，避免静默写坏显存）----
    TensorMatcher({-1, D})   // [vocab, D]，-1 表示任意
        .with_dtype(weights_dtype_)
        .with_device<kDLCUDA>(device_)   // ★ 强制要求 CUDA，这是它跑 GPU 的原因
        .verify(weights);
    TensorMatcher({L, D})    // [L, D]
        .with_dtype(weights_dtype_)      // dtype 必须与 weights 一致
        .with_device<kDLCUDA>(device_)
        .verify(output);
    TensorMatcher({L})       // [L]，一维索引
        .with_dtype<int32_t, int64_t>(indices_dtype_)  // 只接受这两种
        .with_device<kDLCUDA>(device_)
        .verify(indices);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto num_indices = L.unwrap();

    // ★ 运行时算出的实际字节数，必须等于编译期模板参数
    //   这就是"换模型要重新 JIT 编译一次"的原因
    const auto entry_size = dtype_bytes(weights_dtype_.unwrap()) * D.unwrap();
    RuntimeCheck(entry_size == element_size,
                 "IndexKernel: element_size mismatch.");

    // ---- 网格维度计算：以 warp 为单位 ----
    //   num_warps = 每个 token 用的 warp 数 × token 数
    //   num_blocks = ceil(num_warps / 每 block 的 warp 数)
    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_warps = num_splits * num_indices;
    const auto num_blocks = div_ceil(num_warps, kWarpPerBlock);

    // ---- 打包参数（统一转成 void*/char* 做字节级偏移）----
    const auto params = IndexKernelParams{
        .output = static_cast<char *>(output.data_ptr()),
        .weight = static_cast<const char *>(weights.data_ptr()),
        .indice = indices.data_ptr(),
        .num_warps = num_warps,
    };

    // ---- 分派：masked 版 / 普通版 × int32 / int64，共 4 个实例 ----
    if (mask_opts.has_value()) {
      const auto &obj = mask_opts.value();
      const auto [start, length] = obj;   // 合法区间 [start, start+length)
      const auto m_params = MaskedKernelParams{
          .params = params,
          .start = static_cast<std::size_t>(start),
          .length = static_cast<std::size_t>(length),
      };
      // 三元运算符选出具体函数指针（同类型，可合并）
      const auto kernel =
          use_int32 ? masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int32_t>
                    : masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, m_params);
    } else {
      const auto kernel =
          use_int32 ? index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int32_t>
                    : index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, params);
    }
  }
};

} // namespace
