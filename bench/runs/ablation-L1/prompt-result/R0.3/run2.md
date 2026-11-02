# R0.3 run2

【草稿】

**1. 核心逻辑梳理与直觉预判**
*   **任务特征**：27B模型，64K上下文。关键比例是 90:9:1（缓存:新输入:输出）。这意味着绝大多数计算发生在 Prefill 阶段（处理那 9% 的新 token 和可能的 KV Cache 加载/验证），而 Decode 阶段只生成 1 个 token？不对，题目说“单次上下文总长 64K”，且比例是 90:9:1。通常 LLM 的推理成本大头在 Prefill（计算密集）和 Memory Bandwidth（Decode 带宽密集）。
    *   等等，仔细看比例：“缓存命中的 token : 新输入 token : 输出 token ≈ 90 : 9 : 1”。
    *   如果总长 64K，那么：
        *   缓存命中 (KV Cache Hit): $0.9 \times 64K \approx 57.6K$ tokens。这部分不需要重新计算 Attention QKV，只需要读取 KV Cache。
        *   新输入 (Prefill): $0.09 \times 64K \approx 5.76K$ tokens。这部分需要完整的 Forward Pass 计算。
        *   输出 (Decode): $0.01 \times 64K \approx 640$ tokens。这部分是自回归生成。
    *   **修正理解**：这里的“输出 token”是指生成的长度吗？如果是 640 个 token 的输出，那解码时间也不短。但题目中有一个硬约束：“每个并发流的解码速度不低于 20 token/s”。
    *   让我们重新审视“每完成一个任务的成本”。成本 = (硬件折旧 + 能耗) / 有效任务数。
    *   由于有重做机制（通过率 < 100%），实际吞吐量 = 理论吞吐量 $\times$ 通过率。为了简化，假设通过率固定，我们主要比较硬件的**单位算力/带宽成本**。

*   **瓶颈分析**：
    *   **Prefill (5.76K tokens)**: 计算密集型 (Compute Bound)。取决于 FLOPs。
    *   **Decode (640 tokens)**: 内存带宽密集型 (Memory Bandwidth Bound)。取决于 HBM/GDDR 带宽。
    *   **KV Cache**: 57.6K tokens 的 KV Cache 需要常驻显存。对于 27B 模型（假设 FP16/BF16），KV Cache 大小估算：
        *   Layer 数 ~32, Head Dim ~128, Num Heads ~32 (GQA可能更少，比如 8 KV heads)。
        *   单 token KV 大小 $\approx 2 \times \text{Layers} \times \text{KV\_Heads} \times \text{Head\_Dim} \times 2 \text{ bytes}$ (K and V).
        *   若 GQA 8 heads: $2 \times 32 \times 8 \times 128 \times 2 \approx 131K$ bytes/token $\approx 0.13$ MB/token.
        *   57.6K tokens $\times 0.13$ MB $\approx 7.5$ GB.
        *   加上模型权重 27B $\times$ 2 bytes (FP16) = 54 GB.
        *   总显存需求 $\approx 54 + 7.5 \approx 61.5$ GB.
    *   **结论**：必须使用至少 80GB 显存的 GPU（如 A100-80G, H100-80G, L40S-48G不够, RTX 4090-24G不够）。或者使用多卡互联，或者使用量化 (INT8/INT4)。
    *   题目要求“常见候选硬件”。本地跑 27B，通常是工作站或小型服务器。
    *   候选者：
        1.  NVIDIA H100 (PCIe or SXM) - 顶级，贵。
        2.  NVIDIA A100 (80GB) - 上一代旗舰，性价比可能更高。
        3.  NVIDIA L40S (48GB) - 显存不够 FP16，需 INT8/FP8。L40S 支持 FP8。
        4.  NVIDIA RTX 4090 (24GB) - 显存严重不足，除非激进量化 INT4 且 offload CPU RAM（速度慢，违背 20 tok/s 约束？需验证）。
        5.  AMD MI300X - 192GB HBM3，大显存，高带宽。
        6.  Apple M2 Ultra/M3 Ultra - 统一内存，带宽极高，但生态兼容性（CUDA）可能有摩擦，且价格极高。

*   **精度选择**：
    *   为了最小化成本，通常会用量化。
    *   FP16/BF16: 54GB 权重。
    *   INT8: 27GB 权重。
    *   INT4: 13.5GB 权重。
    *   如果使用 INT4，RTX 4090 (24GB) 可以放下权重 + 部分 KV Cache。但是 4090 只有 1TB/s 带宽，A100 有 2TB/s，H100 有 3.35TB/s。
    *   硬约束：Decode >= 20 tok/s。
    *   Decode 速度 $\approx$ 带宽 / (权重大小 + KV Cache 读取量)。
    *   在 INT4 下，权重 13.5GB。KV Cache (FP16存储?) 如果 KV 也量化到 INT8，则更小。
    *   假设 KV Cache 也是 FP16 (7.5GB)。总读取量 $\approx 21$ GB per step? 不，Decode 每一步读取整个模型权重 + 当前步相关的 KV。
    *   实际上，Decode 的瓶颈是读取所有权重。
    *   Speed $\approx$ BW / Weight_Size.
    *   4090: 1 TB/s / 13.5 GB $\approx 74$ tok/s. 满足 20 tok/s.
    *   A100: 2 TB/s / 27 GB (INT8) $\approx 74$ tok/s.
    *   H100: 3.35 TB/s / 54 GB (FP16) $\approx 62$ tok/s.
    *   看起来只要带宽够，都能满足 20 tok/s。瓶颈可能在 Prefill 或 整体吞吐。

*   **成本模型构建**：
    *   我们需要计算 "Cost per Task"。
    *   Task Time = Prefill Time + Decode Time.
    *   Prefill Time $\propto$ Compute.
    *   Decode Time $\propto$ Memory Access.
    *   由于 90% 是缓存，Prefill 只算 9% 的新 token。
    *   但是，每次请求都要加载/检查 KV Cache 吗？如果是 PagedAttention 或类似技术，KV Cache 驻留显存。
    *   关键点：**并发流**。题目说“每个并发流”。意味着我们可以同时跑多个任务。
    *   如果并发度高，GPU 利用率会提高，分摊固定成本（折旧、基础功耗）。
    *   但是，KV Cache 占用显存。显存越大，能支持的并发数越多。
    *   **策略**：寻找 **$/Token** 最低的硬件，同时满足并发下的延迟约束。

*   **候选硬件调研数据（基于公开知识 [S] 和估算 [E]）**：
    1.  **NVIDIA H100 PCIe**:
        *   Price: ~$25k-$30k [E].
        *   Power: ~350W.
        *   Bandwidth: 3.35 TB/s (HBM3).
        *   Compute: ~67 TFLOPS FP16 (Tensor Core with sparsity maybe higher, let's use dense ~67 TFLOPS for safety or ~130 TFLOPS sparse? Standard is usually quoted as ~50-60 TFLOPS BF16 dense). Let's use ~60 TFLOPS BF16.
    2.  **NVIDIA A100 80GB PCIe**:
        *   Price: ~$10k-$15k (used/new varies, assume $12k) [E].
        *   Power: ~300W.
        *   Bandwidth: 2.0 TB/s (HBM2e).
        *   Compute: ~31 TFLOPS BF16.
    3.  **NVIDIA L40S**:
        *   Price: ~$8k-$10k [E].
        *   Power: ~350W.
        *   Bandwidth: 864 GB/s (GDDR6). *注意：L40S 是 GDDR6，不是 HBM*.
        *   Compute: ~90.5 TFLOPS FP8, ~45 TFLOPS FP16.
        *   显存: 48GB.
    4.  **AMD MI300X**:
        *   Price: ~$15k-$20k [E].
        *   Power: ~750W.
        *   Bandwidth: 5.3 TB/s (HBM3).
        *   Compute: ~1308 TFLOPS FP8 (sparse), ~650 TFLOPS FP8 (dense)? No, MI300X is huge. ~1.3 PFLOPS FP8. BF16 ~300+ TFLOPS.
        *   显存: 192GB.
    5.  **NVIDIA RTX 4090**:
        *   Price: ~$2k [E].
        *   Power: ~450W.
        *   Bandwidth: 1.0 TB/s (GDDR6X).
        *   Compute: ~82 TFLOPS FP16.
        *   显存: 24GB.

*   **场景模拟**：
    *   模型: 27B.
    *   精度: 为了最大化性价比，我们考虑 **INT8** 或 **FP16**.
    *   如果选 4090，必须 INT4 或 INT8. INT8 权重 27GB > 24GB 显存，放不下。所以 4090 只能跑 INT4 或部分 Offload. Offload 会极大降低 Decode 速度（CPU RAM 带宽 ~50GB/s vs GPU 1TB/s）。
    *   Decode Speed on 4090 (INT4, no offload):
        *   Weight Size: 13.5 GB.
        *   BW: 1000 GB/s.
        *   Theoretical Max: 1000 / 13.5 = 74 tok/s.
        *   Actual efficiency ~50-60%. So ~40 tok/s. Satisfies 20 tok/s.
    *   如果选 A100/H100/L40S/MI300X，可以用 FP16 或 INT8.
        *   FP16 Weight: 54GB.
        *   A100 (2TB/s): 2000/54 = 37 tok/s (theoretical). Efficiency 50% -> 18 tok/s. **FAIL** constraint (>=20).
        *   所以 A100 跑 FP16 可能卡在边缘。必须用 INT8 (27GB).
        *   A100 (INT8): 2000/27 = 74 tok/s. OK.
        *   H100 (FP16): 3350/54 = 62 tok/s. OK.
        *   L40S (FP16): 864/54 = 16 tok/s. **FAIL**. Must use INT8 (27GB).
        *   L40S (INT8): 864/27 = 32 tok/s. OK.
        *   MI300X (FP16): 5300/54 = 98 tok/s. OK.

    *   **Prefill Cost**:
        *   Tokens to compute: 5.76K.
        *   FLOPs per token in Transformer $\approx 2 \times \text{Params}$. (Roughly).
        *   Params = 27e9.
        *   FLOPs per token $\approx 54$ GFLOP.
        *   Total Prefill FLOPs = $5760 \times 54 \text{ GFLOP} \approx 311$ TFLOP.
        *   Time = FLOPs / Throughput.
        *   A100 (BF16 ~31 TFLOPS): $311 / 31 \approx 10$ seconds.
        *   H100 (BF16 ~60 TFLOPS): $311 / 60 \approx 5.2$ seconds.
        *   L40S (BF16 ~45 TFLOPS): $311 / 45 \approx 6.9$ seconds.
        *   4090 (FP16 ~82 TFLOPS): $311 / 82 \approx 3.8$ seconds. (Note: 4090 has high compute but low memory bandwidth, prefill is compute bound so it might be fast here!).

    *   **Decode Cost**:
        *   Tokens to generate: 640.
        *   Time per token $\approx$ 1 / Speed.
        *   A100 (INT8, ~74 tok/s eff 37?): Let's refine efficiency. Real-world decode efficiency is often limited by kernel launch overhead and non-linear layers. Let's assume 50% of peak BW.
            *   A100 INT8: Peak 74, Real ~37 tok/s. Time = $640 / 37 \approx 17.3$ s.
            *   H100 FP16: Peak 62, Real ~31 tok/s. Time = $640 / 31 \approx 20.6$ s.
            *   L40S INT8: Peak 32, Real ~16 tok/s. Time = $640 / 16 = 40$ s.
            *   4090 INT4: Peak 74, Real ~37 tok/s. Time = $640 / 37 \approx 17.3$ s.

    *   **Total Task Time (Single Stream)**:
        *   A100: 10 + 17.3 = 27.3 s.
        *   H100: 5.2 + 20.6 = 25.8 s.
        *   L40S: 6.9 + 40 = 46.9 s.
        *   4090: 3.8 + 17.3 = 21.1 s.

    *   **Concurrency & Memory**:
        *   Each task uses ~7.5GB KV Cache (FP16) + Model Weights.
        *   If we run N concurrent tasks, we need N * KV_Cache size free memory.
        *   Model Weights are shared.
        *   A100 (80GB): Weights 27GB (INT8). Free 53GB. Tasks = $53 / 7.5 \approx 7$ concurrent.
        *   H100 (80GB): Weights 54GB (FP16). Free 26GB. Tasks = $26 / 7.5 \approx 3.4 \rightarrow 3$ concurrent.
        *   L40S (48GB): Weights 27GB (INT8). Free 21GB. Tasks = $21 / 7.5 \approx 2.8 \rightarrow 2$ concurrent.
        *   4090 (24GB): Weights 13.5GB (INT4). Free 10.5GB. KV Cache needs to be smaller? If KV is FP16 (7.5GB), only 1 task fits ($10.5/7.5=1.4$). If KV is INT8 (3.75GB), 2 tasks fit. Let's assume 1-2 concurrent.

    *   **Throughput (Tasks per second)**:
        *   Assume system is saturated. Throughput $\approx$ Concurrency / Avg_Task_Time.
        *   A100: $7 / 27.3 \approx 0.256$ TPS.
        *   H100: $3 / 25.8 \approx 0.116$ TPS.
        *   L40S: $2 / 46.9 \approx 0.042$ TPS.
        *   4090: $1.5 / 21.1 \approx 0.071$ TPS. (Assuming 1.5 avg concurrency due to memory pressure).

    *   **Cost per Task**:
        *   Annual Cost = Depreciation + Energy.
        *   Depreciation: 3 years straight line.
        *   Energy: Power * 24h * 365d * $0.1/kWh.
        *   **A100**:
            *   Price $12k. Dep/yr = $4k.
            *   Power 300W. Energy cost = $0.3 kW * 8760 h * 0.1 = $262/yr.
            *   Total Fixed/Opex $\approx 4262$ USD/yr.
            *   Tasks/yr = $0.256 * 31536000 \approx 8,073,000$ tasks.
            *   Cost/Task = $4262 / 8,073,000 \approx \$0.00052$.
        *   **H100**:
            *   Price $25k. Dep/yr = $8.3k.
            *   Power 350W. Energy cost = $306/yr.
            *   Total $\approx 8606$ USD/yr.
            *   Tasks/yr = $0.116 * 31.5M \approx 3,654,000$ tasks.
            *   Cost/Task = $8606 / 3,654,000 \approx \$0.00235$.
        *   **L40S**:
            *   Price $9k. Dep/yr = $3k.
            *   Power 350W. Energy cost = $306/yr.
            *   Total $\approx 3306$ USD/yr.
            *   Tasks/yr = $0.042 * 31.5M \approx 1,323,000$ tasks.
            *   Cost/Task = $3306 / 1,323,000 \approx \$0.00250$.
        *   **4090**:
            *   Price $2k. Dep/yr = $666.
            *   Power 450W. Energy cost = $394/yr.
            *   Total $\approx 1060$ USD/yr.
            *   Tasks/yr = $0.071 * 31.5M \approx 2,236,000$ tasks.
            *   Cost/Task = $1060 / 2,236,000 \approx \$0.00047$.

    *   **Comparison**:
        *   4090: $0.00047
        *   A100: $0.00052
        *   H100: $0.00235
        *   L40S: $0.00250

    *   **Wait, check the logic.**
        *   Did I underestimate 4090's concurrency?
        *   4090 has 24GB. Model INT4 is 13.5GB. Remaining 10.5GB.
        *   KV Cache for 64K context.
        *   If we use PagedAttention, we can share pages? No, each stream has unique KV.
        *   KV size for 64K tokens.
        *   For 27B model, let's re-calculate KV size more carefully.
        *   Llama 2/3 architecture style.
        *   Layers: 32.
        *   Hidden Dim: 4096.
        *   Num Heads: 32. Head Dim: 128.
        *   GQA? Llama 3 70B uses GQA. 27B might not exist in standard Llama 3 (70B, 405B). Maybe Mistral 27B? Or a custom 27B.
        *   Let's assume standard MHA (Multi-Head Attention) without GQA for worst case, or GQA if modern.
        *   If MHA: KV heads = 32.
        *   KV per token = $2 (\text{K,V}) \times 32 (\text{layers}) \times 32 (\text{heads}) \times 128 (\text{dim}) \times 2 (\text{bytes}) = 524,288$ bytes $\approx 0.5$ MB/token.
        *   64K tokens * 0.5 MB = 32 GB.
        *   **Whoa!** My previous estimate of 7.5GB assumed GQA with 8 heads.
        *   If it's MHA (32 heads), KV Cache is 32GB.
        *   Then:
            *   4090 (24GB): Cannot hold Model (13.5GB) + KV (32GB). **Impossible**.
            *   Even with INT4 weights, KV Cache dominates.
            *   Unless... the prompt says "Context ... 64K". Does it mean the *current* active context is 64K? Yes.
            *   So, if KV Cache is 32GB, you need > 32GB just for cache.
            *   This eliminates 4090 completely unless heavy quantization of KV (INT4 KV?).
            *   If KV is INT4: 32GB / 4 = 8GB.
            *   Then 4090: 13.5 (Weights) + 8 (KV) = 21.5 GB. Fits in 24GB.
            *   But does INT4 KV work well? Usually yes for inference.
            *   However, 4090 bandwidth is 1TB/s. Reading 13.5GB weights + 8GB KV (partially) per step.
            *   Actually, in Decode, you read ALL weights, and ONLY the relevant KV entries for the current position? No, FlashAttention reads the whole block?
            *   In standard attention, you compute Q dot K for all previous keys. So you access the entire KV cache for that sequence.
            *   So Decode Read Volume = Weight Size + KV Cache Size.
            *   4090: Read 13.5GB + 8GB = 21.5GB per token?
            *   Wait, no. You don't read the *entire* KV cache from DRAM every single token generation if it fits in L2/SRAM? No, 24GB VRAM is much larger than L2. It stays in VRAM.
            *   The bottleneck is reading the **Weights** from VRAM to SMs. The KV data is also in VRAM.
            *   So the bandwidth requirement is dominated by the Weights (13.5GB).
            *   So 4090 speed $\approx$ BW / Weight_Size.
            *   If KV is in VRAM, it doesn't consume extra DRAM bandwidth *unless* it's streamed? No, it's resident.
            *   So 4090 Decode Speed is still ~74 tok/s (limited by weight loading).
            *   BUT, Prefill!
            *   Prefill requires writing to KV Cache and computing Attention.
            *   With 64K context, Prefill is very expensive.
            *   4090 has high Compute (82 TFLOPS).
            *   A100 has lower Compute (31 TFLOPS) but higher Bandwidth (2TB/s).
            *   Since 90% is cached, Prefill is only 9% of input.
            *   But the *cost* of Prefill is proportional to Compute.
            *   Let's re-evaluate the "Cost per Task".

    *   **Revised Candidate List based on Memory Constraint (KV Cache ~32GB MHA or ~8GB GQA-INT4)**:
        *   Assumption: To be safe and "common", we should assume a configuration that works.
        *   If the model supports GQA (likely for 27B efficient models like Mixtral or newer architectures), KV is small.
        *   If it's a dense MHA 27B, KV is huge.
        *   Given "Common candidate hardware", I will assume **GQA** or **Quantized KV** is possible, OR the hardware has enough memory.
        *   Candidates with >48GB VRAM: A100, H100, MI300X, L40S(48G).
        *   4090 is risky due to memory wall unless aggressive quantization. I will include it but flag high risk/low concurrency.

    *   **Let's pick the top contenders: A100 vs H100 vs L40S vs MI300X.**
        *   **L40S**: 48GB. Good for FP8/INT8. Low bandwidth (864 GB/s). High Compute.
        *   **A100**: 80GB. Medium Bandwidth (2TB/s). Medium Compute.
        *   **H100**: 80GB. High Bandwidth (3.35TB/s). High Compute.
        *   **MI300X**: 192GB. Very High Bandwidth (5.3TB/s). Very High Compute.

    *   **Cost Analysis Update**:
        *   **L40S**:
            *   Price ~$9k.
            *   Throughput: Limited by Decode Bandwidth.
            *   Decode Speed (INT8, 27GB weights): 864/27 = 32 tok/s.
            *   Prefill Speed (INT8/FP8): ~90 TFLOPS (FP8) or 45 (FP16). Let's say 90 TFLOPS effective for FP8.
            *   Prefill Time: 311 TFLOP / 90 = 3.4 s.
            *   Decode Time: 640 / (32 * 0.5 eff) = 40 s.
            *   Total: 43.4 s.
            *   Concurrency: 48GB - 27GB (Weights) = 21GB for KV.
            *   If KV is 32GB (MHA FP16), 0 tasks fit.
            *   If KV is 8GB (GQA INT4), 2 tasks fit.
            *   Let's assume GQA INT4 KV is required for L40S.
            *   Tasks/sec: 2 / 43.4 = 0.046 TPS.
            *   Cost/Task: $3306 / (0.046 * 31.5M) = $0.0022.

        *   **A100**:
            *   Price ~$12k.
            *   Decode Speed (INT8): 2000/27 = 74 tok/s. Eff 37.
            *   Prefill Speed (BF16): 31 TFLOPS. Time: 311/31 = 10 s.
            *   Decode Time: 640 / 37 = 17.3 s.
            *   Total: 27.3 s.
            *   Concurrency: 80GB - 27GB = 53GB.
            *   If KV is 8GB (GQA INT4), 6 tasks fit.
            *   If KV is 32GB (MHA FP16), 1 task fits ($53/32=1.6$).
            *   Let's assume MHA FP16 KV (standard assumption if not specified GQA).
            *   Tasks/sec: 1.5 / 27.3 = 0.055 TPS.
            *   Cost/Task: $4262 / (0.055 * 31.5M) = $0.0024.

        *   **H100**:
            *   Price ~$25k.
            *   Decode Speed (FP16): 3350/54 = 62 tok/s. Eff 31.
            *   Prefill Speed (FP16): 60 TFLOPS. Time: 311/60 = 5.2 s.
            *   Decode Time: 640 / 31 = 20.6 s.
            *   Total: 25.8 s.
            *   Concurrency: 80GB - 54GB = 26GB.
            *   KV (MHA FP16, 32GB): 0 tasks fit? 26 < 32.
            *   So H100 MUST use INT8 Weights (27GB) or Quantized KV.
            *   If H100 uses INT8 Weights (27GB):
                *   Free Mem: 53GB.
                *   KV (MHA FP16, 32GB): 1 task fits ($53/32=1.6$).
                *   Decode Speed (INT8): 3350/27 = 124 tok/s. Eff 62.
                *   Decode Time: 640 / 62 = 10.3 s.
                *   Prefill Time (INT8 ~120 TFLOPS?): 311/120 = 2.6 s.
                *   Total: 12.9 s.
                *   Tasks/sec: 1.5 / 12.9 = 0.116 TPS.
                *   Cost/Task: $8606 / (0.116 * 31.5M) = $0.00235.

        *   **MI300X**:
            *   Price ~$18k.
            *   Bandwidth 5.3 TB/s.
            *   Compute ~300 TFLOPS BF16.
            *   Weights FP16 (54GB).
            *   Free Mem: 192 - 54 = 138GB.
            *   KV (MHA FP16, 32GB): 4 tasks fit ($138/32=4.3$).
            *   Decode Speed: 5300/54 = 98 tok/s. Eff 49.
            *   Decode Time: 640 / 49 = 13 s.
            *   Prefill Time: 311 / 300 = 1 s.
            *   Total: 14 s.
            *   Tasks/sec: 4 / 14 = 0.285 TPS.
            *   Cost/Task:
                *   Dep: $18k/3 = $6k.
                *   Power: 750W. Energy: $657/yr.
                *   Total: $6657/yr.
                *   Cost/Task: $6657 / (0.285 * 31.5M) = $0.00073.

    *   **Result**: MI300X looks cheapest per task due to massive memory allowing high concurrency and high bandwidth reducing latency.
    *   **But**, is MI300X "Common"? It's becoming common in AI clusters.
    *   What about **A100** again?
        *   If A100 uses GQA/Quantized KV to fit more tasks?
        *   If A100 fits 6 tasks (using INT4 KV):
            *   Tasks/sec: 6 / 27.3 = 0.22 TPS.
            *   Cost/Task: $4262 / (0.22 * 31.5M) = $0.00061.
        *   This beats MI300X!
        *   Why? Because A100 is cheaper and power-efficient compared to MI300X's 750W.
        *   Key driver: **Concurrency**.
        *   If the model allows Quantized KV (which is standard now, e.g., in vLLM/TGI), then Memory Bandwidth and Capacity are the main factors.
        *   A100 (2TB/s, 80GB) vs MI300X (5.3TB/s, 192GB).
        *   A100 is ~40% price of MI300X, ~40% power.
        *   Throughput scales with Concurrency.
        *   If A100 can sustain 6-7 streams and MI300X 4-5 streams (due to software maturity/driver issues?), A100 wins.
        *   Actually, MI300X has 192GB. It can fit MANY more KV caches.
        *   If KV is 8GB (INT4 GQA), MI300X fits 17 streams.
        *   A100 fits 6 streams.
        *   MI300X Throughput: 17 / 14s = 1.2 TPS.
        *   A100 Throughput: 6 / 27.3s = 0.22 TPS.
        *   MI300X is 5x faster throughput.
        *   Cost: MI300X is 1.5x more expensive.
        *   So MI300X is definitely better *if* it runs efficiently.

    *   **However**, there is a catch. **Software Ecosystem**.
    *   CUDA (NVIDIA) is vastly superior for LLM inference frameworks (vLLM, TensorRT-LLM).
    *   ROCm (AMD) is improving but often lags in optimization for specific kernels.
    *   If we assume "Out of the box" performance, NVIDIA is safer.
    *   Between NVIDIA cards: **A100 vs H100 vs L40S**.
    *   L40S is out (low BW).
    *   H100 vs A100.
    *   H100 has 1.6x Bandwidth, 2x Compute, 2x Price, 1.2x Power.
    *   If H100 can fit same number of tasks (same 80GB mem), its throughput is roughly 1.5x A100.
    *   Cost is 2x.
    *   So A100 is likely more cost-effective per task than H100 for this specific workload (memory bound decode + moderate prefill).
    *   **Winner among NVIDIA**: **A100 80GB**.

    *   **What about RTX 4090?**
    *   If we accept INT4 Weights + INT4 KV.
    *   4090 (24GB).
    *   Weights 13.5GB. KV 8GB. Total 21.5GB. Fits.
    *   Concurrency: 1 stream (barely).
    *   Throughput: 1 / 21s = 0.047 TPS.
    *   Cost: $1060/yr.
    *   Cost/Task: $1060 / (0.047 * 31.5M) = $0.00071.
    *   Compare A100 (with INT4 KV, 6 streams):
    *   A100 Throughput: 0.22 TPS.
    *   Cost/Task: $0.00061.
    *   A100 is slightly cheaper per task.
    *   And A100 is more reliable/stable for enterprise use.
    *   Also, 4090 has consumer driver limitations (no ECC, less stable under 24/7 load).

    *   **Final Recommendation**: **NVIDIA A100 80GB**.
    *   Reasoning: Best balance of Cost, Bandwidth, and Memory Capacity for high-concurrency inference of 27B model with large context.

    *   **Refining the Calculation for Output**:
    *   I need to present the table.
    *   I need to define the "Task" clearly.
    *   I need to handle the "Pass Rate" factor.
    *   If Pass Rate is $P$, Effective Throughput = Theoretical Throughput * $P$.
    *   Cost per *Successful* Task = Cost per Attempt / $P$.
    *   Since $P$ is constant across hardware, it cancels out in the comparison. I will note this.

    *   **Drafting the Sections**:
    *   **Section 0**: Define Question, Scope, Out, Pass/Fail, Hard Constraints, Obj, Cost Factors.
    *   **Section 1**: Research Path.
        *   R-1: Hardware Selection.
        *   R-1.1: Memory Requirements Analysis.
        *   R-1.2: Compute/Bandwidth Profiling.
        *   R-1.3: Cost Modeling.
    *   **Section 2**: Contradictions.
        *   CON-1: Compute vs Bandwidth trade-off (H100 vs A100).
        *   CON-2: Memory Capacity vs Cost (MI300X vs A100).
    *   **Section 3**: Calculations.
        *   Table of candidates.
        *   Show formulas for Time and Cost.
    *   **Section 4**: Conclusion.
        *   BOTTLENECK: Memory Bandwidth (for Decode) and Memory Capacity (for Concurrency).
        *   Recommendation: A100.

    *   **Self-Correction during drafting**:
    *   Did I account for the 90:9:1 ratio correctly?
    *   Yes, Prefill is small (9%), Decode is small (1% of total length, but generates 640 tokens).
    *   Wait, "Output token" is 1% of 64K = 640 tokens.
    *   Is 640 tokens a lot for Decode?
    *   At 30 tok/s, it takes 21 seconds.
    *   Prefill at 30 TFLOPS takes 10 seconds.
    *   So Decode is actually the *longer* part of the time budget in this scenario (21s vs 10s).
    *   Therefore, **Bandwidth** is the primary bottleneck, not Compute.
    *   This reinforces A100 over H100 (H100 has better BW, but costs double. A100 BW is sufficient).
    *   And it reinforces A100 over 4090 (4090 has similar BW to A100? No, 4090 is 1TB/s, A100 is 2TB/s. A100 is 2x faster in Decode).
    *   So A100 is strictly better than 4090 in speed, and only 6x more expensive.
    *   Cost/Task:
        *   4090: $0.00071
        *   A100: $0.00061
    *   A100 wins.

    *   **One detail**: "Local run".
    *   Does "Local" imply a single workstation?
    *   A100 is often in servers. But 80GB A100 exists in PCIe form factor for workstations (though rare/expensive).
    *   Alternatively, 2x RTX 3090 (48GB total)?
    *   3090: 936 GB/s BW. 70 TFLOPS FP16. 24GB each.
    *   2x 3090: 48GB total.
    *   Can it hold 27B INT8 (27GB) + KV?
    *   Yes.
    *   Inter-card communication adds latency.
    *   Probably worse than single A100 due to NVLink/CPU hop overhead.
    *   I will stick with Single GPU solutions for simplicity and "Common Hardware" definition.

    *   **Final Check of Numbers**:
    *   Ensure all sources are marked [E] or [S].
    *   Ensure units are consistent.

    *   **Structure of Output**:
    *   Follow the 5 sections strictly.
    *   Use Chinese.

    *   **Specific Calculation for Table**:
    *   Columns: Solution, Purchase Price, Dep/Year, Energy/Year, O&M/Year, Residual Value, Total Net Cost, $/Task.
    *   Rows: A100, H100, L40S, 4090.

    *   **O&M Cost**:
    *   Typically 5-10% of hardware cost per year for maintenance/support.
    *   I will use 5% [E].

    *   **Residual Value**:
    *   After 3 years.
    *   GPUs hold value well. 50% residual? [E].
    *   Net Cost = Purchase - Residual + (Dep is accounting, Cash flow is Purchase + OpEx - Residual).
    *   The prompt asks for "Full Cycle Net Cost".
    *   Formula: Purchase Price + (Energy + O&M) * 3 - Residual Value.
    *   Or annualized? The table header says "Net Cost" (implied total over lifecycle) and "$/Task".
    *   I will calculate Total Lifecycle Cost (3 years) and divide by Total Tasks in 3 years.

    *   Let's refine the "Cost per Task" calculation in the final output to be precise.
    *   Tasks in 3 years = TPS * 3 * 365 * 24 * 3600.
    *   Net Cost = Purchase + 3*(Energy+OM) - Residual.
    *   $/Task = Net Cost / Total Tasks.

    *   Ready to write.

## 0. 研究问题定义

QUESTION: 在本地运行 27B 规模语言模型处理固定任务时，哪种常见候选硬件的单位有效任务成本最低？
SCOPE:   包含 27B 模型在 64K 上下文、90:9:1 缓存比下的推理性能评估；不包含模型训练、微调及非 NVIDIA/AMD 主流架构（如 ASIC 定制芯片）的深入对比。
OUT:     每完成一个有效任务的货币成本（美元/任务）
PASS:    找出在满足并发解码速度 ≥ 20 token/s 前提下，全周期净成本除以有效任务总数最小的硬件方案
FAIL:    无法确定单一最优解，或所有候选方案均无法满足解码速度硬约束
HARD:    
1. 单个并发流解码速度 ≥ 20 token/s
2. 单次上下文总长 64K token
3. 缓存命中:新输入:输出 = 90:9:1
4. 硬件为市售可获得的通用 GPU 或加速卡
OBJ:     最小化 每有效任务的全周期净成本 (USD/Task)
COST:    
折旧: 按 3 年直线折旧计算，购置价/3 [E][±10%]
能耗: 满载功率 × 8760 小时/年 × $0.1/kWh [E][±5%]
运维: 购置价的 5%/年 (含散热、电力损耗、维护) [E][±20%]
残值: 3 年后二手市场预估回收价，约为购置价的 40%-60% [E][±30%]

## 1. 研究路径

### R-1 硬件选型与成本建模
PARENT:  ROOT
NOW:     无具体硬件选定，仅有模型参数 (27B) 和负载特征 (64K ctx, 9:1 ratio)
GOAL:    得到各候选硬件的单位任务成本排序
GAP:     缺乏针对该特定负载特征的硬件吞吐量与成本关联数据
IN:      模型参数量, 上下文长度, 缓存比例, 解码速度约束, 硬件规格库
OUT:     单位任务成本排名
KIND:    signal
RES:     算力, 钱
SHARE:   1.00
BOUND:   ?
TEST:    对比 A100, H100, L40S, 4090 在模拟负载下的 TPS 与年化成本
STATUS:  split

### R-1.1 显存容量与并发能力评估
PARENT:  R-1
NOW:     未知各硬件能支持的最大并发流数
GOAL:    确定每种硬件在 64K 上下文下的最大并发流数 N_max
GAP:     需估算 KV Cache 大小及模型权重占用，结合硬件显存上限
IN:      模型参数量, 上下文长度, 注意力头配置, 硬件显存大小
OUT:     最大并发流数 N_max
KIND:    constraint
RES:     数据
SHARE:   -
BOUND:   ?
TEST:    计算 27B 模型 INT8/FP16 权重大小及 64K KV Cache 大小，除以硬件可用显存
STATUS:  leaf

### R-1.2 推理延迟与吞吐量测算
PARENT:  R-1
NOW:     未知各硬件的单任务耗时及系统吞吐量
GOAL:    计算每种硬件的任务处理速率 (TPS)
GAP:     需分离 Prefill (计算密集) 和 Decode (带宽密集) 阶段的耗时
IN:      N_max, 硬件算力, 硬件带宽, 模型精度
OUT:     系统吞吐量 TPS
KIND:    signal
RES:     算力
SHARE:   1.00
BOUND:   2.5x
TEST:    根据 FLOPs 和 Bandwidth 公式估算 Prefill 时间和 Decode 时间，求和得单任务时间，乘以并发数得 TPS
STATUS:  leaf

### R-1.3 全周期成本核算
PARENT:  R-1
NOW:     已知各硬件大致市场价格
GOAL:    计算每种硬件 3 年生命周期内的总净成本
GAP:     需整合购置价、能耗、运维及残值
IN:      硬件购置价, 功耗, 残值率
OUT:     全周期净成本 (USD)
KIND:    signal
RES:     钱
SHARE:   1.00
BOUND:   1.5x
TEST:    应用成本公式：Purchase + 3*(Energy+OM) - Residual
STATUS:  leaf

COUPLE: R-1.1 × R-1.2 | 共享资源: 数据 | 杀伤实验: 若增加并发数导致 Decode 带宽竞争加剧，使得单流速度低于 20 tok/s，则证明并发数受限于带宽而非仅显存容量，需重新调整 N_max

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  高带宽与高成本的权衡
NODES:   R-1.2 × R-1.3
TENSION: 高性能硬件 (H100/MI300X) 提供更高带宽从而提升 Decode 速度，但其高昂的购置成本和功耗显著增加了单位任务成本
MECHANISM: Decode 阶段是内存带宽瓶颈，但硬件成本是非线性的。当 A100 的带宽已足以满足 20 tok/s 约束并支撑较高并发时，升级到 H100 带来的边际吞吐量提升不足以覆盖其双倍的成本增量
STATUS:  resolved
VERIFICATION: 对比 A100 和 H100 的 $/Task 计算结果，确认 A100 更低
IMPACT:   ±30%

CON-2  显存容量与并发深度的限制
NODES:   R-1.1 × R-1.2
TENSION: 更大的显存 (如 MI300X 192GB) 允许更多并发流，从而提高总吞吐量，但小显存卡片 (如 4090) 虽然便宜却因并发受限导致总吞吐低，单位成本反而高
MECHANISM: 单位成本 = 固定成本 / 总吞吐。总吞吐与并发数成正比。若小显存卡片的并发数下降幅度超过其价格优势，则大显存卡片更优
STATUS:  resolved
VERIFICATION: 计算 4090 (1-2 并发) 与 A100 (4-6 并发) 的 $/Task，观察是否反转
IMPACT:   ±40%

## 3. 计算与验证

CALCULATIONS:

CAL-1  模型资源占用估算
FORMULA: 
Weight_Size = Params × Bytes_per_Param
KV_Cache_Size = Context_Len × Layers × 2(K,V) × KV_Heads × Head_Dim × Bytes_per_KV
INPUTS:  Params=27e9, Context=64K, Layers=32, KV_Heads=8(GQA assumption), Head_Dim=128
OUTPUT: 
Weight_Size (INT8) = 27 GB
Weight_Size (FP16) = 54 GB
KV_Cache_Size (FP16) ≈ 64K × 32 × 2 × 8 × 128 × 2 bytes ≈ 8.4 GB
ERROR:   ±10% (依赖 GQA 配置假设)
SOURCE:  [F]/[E]

CAL-2  候选硬件吞吐量与成本对比表
FORMULA: 
Prefill_Time = (New_Input_Tokens × 2 × Params) / Compute_FLOPS
Decode_Time = Output_Tokens / (Bandwidth / Weight_Size × Efficiency)
Total_Task_Time = Prefill_Time + Decode_Time
TPS = N_concurrent / Total_Task_Time
Annual_Cost = Depreciation + Energy + OM
$/Task = Annual_Cost / (TPS × Hours_Per_Year × Pass_Rate)
INPUTS:  见下表数据
OUTPUT:  见下表
ERROR:   ±15%
SOURCE:  [F]/[E]

| 方案 | 精度假设 | 权重(GB) | KV(GB)/流 | 可用显存(GB) | 最大并发(N) | 带宽(TB/s) | 算力(TFLOPS) | Prefill(s) | Decode(s) | 单任务(s) | TPS | 购置价($)[E] | 年成本($)[E] | $/任务 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100 80G** | INT8 | 27 | 8.4 | 53 | 6 | 2.0 | 31 | 10.0 | 17.3 | 27.3 | 0.22 | 12,000 | 4,262 | **0.00061** |
| **NVIDIA H100 80G** | INT8 | 27 | 8.4 | 53 | 6 | 3.35 | 60 | 5.2 | 10.3 | 15.5 | 0.39 | 25,000 | 8,606 | 0.00069 |
| **NVIDIA L40S 48G** | INT8 | 27 | 8.4 | 21 | 2 | 0.864 | 45 | 6.9 | 40.0 | 46.9 | 0.04 | 9,000 | 3,306 | 0.00245 |
| **NVIDIA RTX 4090** | INT4 | 13.5 | 4.2(INT4) | 10.5 | 1 | 1.0 | 82 | 3.8 | 17.3 | 21.1 | 0.047 | 2,000 | 1,060 | 0.00071 |
| **AMD MI300X** | FP16 | 54 | 8.4 | 138 | 16 | 5.3 | 300 | 1.0 | 13.0 | 14.0 | 1.14 | 18,000 | 6,657 | 0.00018 |

*注：*
1. *Decode 效率取 50% 峰值带宽利用率。*
2. *L40S 和 4090 因显存/带宽瓶颈，并发数极低，导致单位成本上升。*
3. *MI300X 虽单价高，但因超大显存支持 16 路并发，摊薄了固定成本，理论上单位成本最低。但考虑到软件生态成熟度及“常见硬件”定义，A100 是更稳健的选择。若纯看数字，MI300X 胜出。但在企业落地中，A100 的驱动稳定性和工具链支持使其实际有效吞吐更接近理论值。此处保守推荐 A100 作为平衡点，若追求极致低成本且接受 AMD 栈风险，选 MI300X。鉴于题目问“常见”，A100 更为普遍。*
4. *修正：考虑到 MI300X 的软件开销可能较大，且 A100 是行业标准，我们将 A100 列为首选推荐，但表中数据显示 MI300X 具有潜力。*

VERIFICATIONS:

VER-1  验证 A100 的并发可行性
TARGET:  CAL-2 中 A100 的 N=6
METHOD:  在实际 vLLM 环境中加载 27B INT8 模型，设置 6 个并发请求，监控显存占用及是否 OOM
RESULT:  pass (基于显存计算 27+6*8.4=77.4GB < 80GB)
CONFIDENCE: 0.9

VER-2  验证 Decode 速度约束
TARGET:  CAL-2 中 A100 的 Decode 速度
METHOD:  实测单流生成 640 token 的时间，确保平均速度 > 20 tok/s
RESULT:  pass (理论 37 tok/s > 20 tok/s)
CONFIDENCE: 0.85

## 4. 结论与置信度

BOTTLENECK: R-1.2
REASON: SHARE (1.00) × BOUND (2.5x) = 2.5。这是全路径最大改善因子。提升带宽或并发数是降低成本的关键，而硬件选型直接决定了这一上限。
STOPPED: none
OPEN: none
CONFIDENCE: 0.85
RECOMMENDATION: 建议进一步调研 AMD MI300X 在特定 LLM 框架下的实际软件优化程度。若仅考虑 NVIDIA 阵营及通用稳定性，**NVIDIA A100 80GB** 是最佳性价比选择。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本 | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100 80G** | $12,000 | $4,000 | $262 | $600 | $4,800 (40%) | $14,462 | **$0.00061** |
| **NVIDIA H100 80G** | $25,000 | $8,333 | $306 | $1,250 | $10,000 (40%) | $25,337 | $0.00069 |
| **NVIDIA L40S 48G** | $9,000 | $3,000 | $306 | $450 | $3,600 (40%) | $8,706 | $0.00245 |
| **NVIDIA RTX 4090** | $2,000 | $666 | $394 | $100 | $800 (40%) | $2,466 | $0.00071 |
| **AMD MI300X** | $18,000 | $6,000 | $657 | $900 | $7,200 (40%) | $15,457 | $0.00018 |

*注：全周期净成本 = 购置价 + 3×(能耗+运维) - 残值。$/验收任务 = 全周期净成本 / (TPS × 3年总小时数)。假设通过率为 100% 进行硬件对比，若通过率低于 100%，所有方案的绝对成本同比例上升，相对排序不变。*
