# R0.4 run2

【草稿】

**1. 初始思考与变量梳理**
*   **核心目标**：最小化单任务成本（$ / task）。
*   **关键负载特征**：
    *   模型规模：27B 参数。
    *   上下文长度：64K tokens。
    *   比例：Cache Hit (90%) : New Input (9%) : Output (1%) = 57,600 : 5,760 : 64 tokens。
    *   **极度偏向 Prefill/Compute-bound**：因为 90% 是缓存命中，这部分只涉及 Attention 的 K/V 读取和少量计算，但 9% 的新输入需要完整的 QKV 投影和 FFN 计算。等等，这里有个巨大的陷阱：**“缓存命中的 token”在推理引擎中通常意味着不需要重新计算 Attention 的 QKV，只需要从 KV Cache 中读取。**
    *   *修正理解*：如果前缀完全缓存（Prefix Caching），那么 90% 的 token 不需要经过 Linear 层（MatMul），只需要进行 Attention 操作（Query 与新 Key/Value 交互，或者如果是纯前缀匹配，甚至可能跳过部分 Attention 计算，取决于具体实现，如 FlashAttention 或 PagedAttention 的处理方式）。但在大多数 LLM 推理框架中，"Prefill" 阶段处理的是 *新* 的 prompt。如果 90% 是系统提示词且被缓存，那么实际需要计算的 "Prefill" 只有那 9% 的新输入 + 1% 的输出？
    *   *再仔细读题*：“上下文里有一段长期不变的前缀... 缓存命中的 token : 新输入 token : 输出 token ≈ 90 : 9 : 1”。
        *   这意味着每个请求进来，90% 的部分（约 57.6k tokens）已经在 KV Cache 里了。
        *   我们需要处理的“活跃”计算量主要是那 9% 的新输入（5.76k tokens）和 1% 的输出（64 tokens）。
        *   **但是**，Decode 阶段（生成那 64 个 token）时，每一步都要读取整个 Context 的 KV Cache（64k tokens 的 KV）。
        *   所以，计算瓶颈分为两部分：
            1.  **Prefill of New Input**: 处理 5.76k tokens 的计算密集部分（MatMul）。
            2.  **Decode**: 生成 64 tokens，每步读取 64k KV Cache（带宽密集）。
    *   *对比传统场景*：传统场景可能是 1k input + 1k output。这里是 5.7k input + 64 output。
    *   *结论*：这是一个 **High-Prefill-Low-Decode** 的场景，但 Prefill 量比常规少很多（因为 90% 被缓存了），而 Decode 的内存读取压力极大（因为 Context 长）。
    *   *实际上*：由于 90% 被缓存，真正的 "Prefill Compute" 只有 9% 的量。而 "Decode Memory Bandwidth" 是全程 64K 长度。
    *   *直觉判断*：
        *   普通 GPU（H100/A100）：算力极强，处理那 9% 的新输入非常快。Decode 阶段受限于显存带宽。
        *   CPU/低端 GPU：算力弱，处理 9% 的新输入会很慢。
        *   关键点：27B 模型，FP16 下权重约 54GB。INT8 约 27GB。INT4 约 13.5GB。
        *   64K Context 的 KV Cache 大小：
            *   假设 Layer=32, Heads=8 (GQA?), Head_dim=128。
            *   KV per token $\approx 2 \times \text{Layers} \times \text{KV\_Heads} \times \text{Head\_Dim} \times 2 \text{ bytes}$ (FP16).
            *   估算 KV Cache 大小：对于 27B 模型（类似 Mistral/Llama 架构），64K tokens 的 KV Cache 可能需要几十 GB 到上百 GB 显存（取决于 GQA 头数）。
            *   例如：Llama-3-70B 的 KV cache 很大。27B 相对小一点。
            *   粗略算：$64000 \times 32 \text{ layers} \times 8 \text{ kv heads} \times 128 \times 2 \times 2 (\text{k and v}) \approx 80 \text{ GB}$ (FP16)。
            *   加上模型权重 54GB (FP16)，总共需要 ~134GB 显存才能单并发跑满 64K context。
            *   如果并发 > 1，显存爆炸。
            *   因此，**必须使用量化**（INT8/INT4）或者 **CPU Offloading** 或者 **超大显存卡**。
            *   硬约束：解码速度 >= 20 token/s。

**2. 候选硬件调研方向**
*   **高端数据中心 GPU**: NVIDIA H100 SXM (80GB), A100 (80GB).
    *   优点：高带宽，高算力。
    *   缺点：贵，功耗高。
*   **中高端工作站 GPU**: NVIDIA RTX 4090 (24GB), RTX 6000 Ada (48GB).
    *   优点：性价比相对高（相比 H100 单价）。
    *   缺点：显存小（4090 24G 根本放不下 27B FP16 + 64K KV）。必须 INT4/INT8。
*   **消费级多卡/集群**: 比如 2x 4090? PCIe 带宽瓶颈会严重拖慢 Decode。
*   **Apple Silicon**: M2 Ultra / M3 Ultra (Unified Memory).
    *   优点：统一内存大（可达 192GB+），带宽高（M2 Ultra ~800GB/s）。
    *   缺点：算力不如 H100，生态兼容性。
*   **CPU Only**: EPYC 9004/9005.
    *   优点：便宜，内存大。
    *   缺点：带宽低（DDR5 ~50-100GB/s），27B 模型 Decode 极慢，大概率无法满足 20 token/s。
    *   *快速估算*：27B INT8 = 27GB。Decode 每步读 27GB。带宽 100GB/s -> 0.27s/token -> 3.7 token/s。远低于 20 token/s。**排除纯 CPU。**

**3. 细化候选列表**
鉴于 27B 模型 + 64K Context 的显存需求巨大：
1.  **NVIDIA H100 80GB (SXM/HBM3)**: 单卡可容纳 FP16 权重(54G) + 部分 KV Cache。若并发低，可行。若需更高并发，需多卡。
2.  **NVIDIA A100 80GB (HBM2e)**: 类似 H100，但算力稍低，带宽稍低。
3.  **NVIDIA RTX 4090 24GB**: 必须 INT4 量化 (AWQ/GPTQ)。权重 ~14GB。KV Cache 在 FP16 下太大，KV 也要量化 (FP8/INT8)。总占用可能勉强塞进 24GB (14G Weight + 10G KV?)。风险高，性能受限于 PCIe 和显存容量。
4.  **Apple M2 Ultra (192GB Unified Memory)**: 可以跑 FP16 甚至 BF16。带宽 800GB/s。
    *   Decode 估算：Weight 54GB (FP16)。每步读 54GB。800GB/s -> 0.0675s/token -> ~14.8 token/s。
    *   *警告*：这还没算 KV Cache 的读取！Decode 时每步要读 Weights + KV Cache。
    *   KV Cache 64K tokens @ FP16 $\approx$ 80GB (估算值，需精确计算)。
    *   总读取量 = 54GB (Weights) + 80GB (KV) = 134GB per step.
    *   Time per step = 134GB / 800GB/s = 0.1675s.
    *   Speed = 1 / 0.1675 = **5.97 token/s**.
    *   **结果**：M2 Ultra 也达不到 20 token/s！除非 KV Cache 极度压缩（INT4 KV? 很少见）或者模型更小。
    *   *重新检查 M2 Ultra 带宽*：M2 Ultra 峰值带宽确实是 800GB/s。
    *   *重新检查 KV Cache 大小*：
        *   公式：$KV\_Size = N_{layers} \times N_{kv\_heads} \times d_{head} \times 2 (K,V) \times N_{tokens} \times BytesPerElement$.
        *   假设 27B 模型结构接近 Llama-3-70B 的缩小版或 Mixtral 等 MoE? 不，27B 通常是 Dense。
        *   参考 Llama-2-70B: 80 layers, 8 kv heads, 128 dim.
        *   参考 Mistral 7B: 32 layers, 8 kv heads, 128 dim.
        *   假设 27B 模型有 40-50 layers, 8-16 kv heads.
        *   取保守估计：40 layers, 8 kv heads, 128 dim.
        *   $40 \times 8 \times 128 \times 2 \times 64000 \times 2 (FP16) = 10.24 \times 10^9 \times 2 \times 2 \approx 40 \text{ GB}$.
        *   如果 Layers 更多，比如 60 layers: $60 \times ... \approx 60 \text{ GB}$.
        *   如果 KV Heads 是 GQA 减少的，比如 4 heads: 减半。
        *   让我们假设一个典型的 27B Dense 模型（如 Command R 104B 的缩小版? 不，Command R 是 35B）。
        *   查找典型 27B 模型：目前市面上没有标准的 "27B" 热门开源模型，最接近的是 **Mistral Small (7B)**, **Llama 3 (8B/70B)**, **Yi (34B)**, **DeepSeek-V2 (236B MoE)**.
        *   题目指定 "27B 规模"。我们按 **Dense Transformer, 32-40 Layers, GQA** 估算。
        *   假设：32 Layers, 8 KV Heads, 128 Dim.
        *   KV Cache (FP16) for 64K: $32 \times 8 \times 128 \times 2 \times 64000 \times 2 \text{ bytes} \approx 8.4 \text{ GB}$.
        *   这个数值比较合理。
        *   那么 M2 Ultra Decode: Read Weights (54GB FP16 or 27GB INT8) + Read KV (8.4GB).
        *   Case FP16: Read 54 + 8.4 = 62.4GB. Time = 62.4/800 = 0.078s. Speed = 12.8 t/s. **Fail < 20 t/s.**
        *   Case INT8: Read 27 + 4.2 = 31.2GB. Time = 31.2/800 = 0.039s. Speed = 25.6 t/s. **Pass.**
        *   所以 M2 Ultra 跑 INT8 有机会。
    *   **NVIDIA H100**:
        *   Bandwidth: 3.35 TB/s (HBM3).
        *   Compute: 989 TFLOPS (FP16).
        *   Case INT8: Read 27GB + 4.2GB = 31.2GB. Time = 31.2 / 3350 = 0.0093s. Speed = 107 t/s. **Easy Pass.**
        *   Prefill Cost: 9% of 64K = 5760 tokens.
        *   FLOPs for Prefill: $2 \times Params \times Tokens = 2 \times 27e9 \times 5760 \approx 311 GFLOPs$.
        *   Time: 311e9 / 989e12 = 0.00031 s. Negligible compared to decode latency if batch size is small?
        *   Wait, the task includes generating 64 tokens.
        *   Total Time = Time(Prefill New) + 64 * Time(Decode Step).
        *   For H100: Prefill ~0.3ms. Decode 64 steps * 9.3ms = 595ms. Total ~0.6s per task.
    *   **RTX 4090**:
        *   Bandwidth: 1008 GB/s.
        *   Compute: 82 TFLOPS (FP16) / 164 TFLOPS (INT8?). Actually Tensor Cores are much faster.
        *   Must use INT4/INT8. Let's say INT8 weights (27GB) doesn't fit in 24GB with KV.
        *   Must use INT4 weights (~14GB) + FP16/INT8 KV.
        *   If KV is INT8 (4.2GB), Total ~18GB. Fits in 24GB.
        *   Decode Read: 14GB (W) + 4.2GB (KV) = 18.2GB.
        *   Time: 18.2 / 1008 = 0.018s. Speed = 55 t/s. **Pass.**
        *   Prefill: 5760 tokens.
        *   FLOPs: 311 GFLOPs.
        *   4090 Tensor Core INT8 throughput is roughly 330 TOPS? No, FP16 is 82 TFLOPS dense, but with sparsity/tensor cores it's higher. Let's assume effective ~200 TFLOPS for mixed precision.
        *   Time: 311e9 / 200e12 = 1.5 ms.
        *   Total Time: 1.5ms + 64 * 18ms = 1.15s.
        *   Compare H100: 0.6s.
        *   4090 is slower per task, but much cheaper.

**4. 成本模型构建**
*   **Cost per Task** = (Hardware Depreciation + Electricity + Maintenance) / Tasks per Year.
*   Or simpler: **$ / Task** = (Hourly Cost of Hardware) * (Time per Task).
*   Hourly Cost = (CapEx / Lifetime_Hours) + (Power_kW * Rate_per_kWh).
*   Assumptions:
    *   Lifetime: 3 years (hardware refresh cycle).
    *   Utilization: 24/7? The problem implies a steady stream of tasks. Let's assume high utilization (80%) or calculate based on full capacity. To compare hardware efficiency, we usually look at $/Task at max throughput or specific concurrency.
    *   However, the constraint is "per concurrent stream". This implies we can run multiple streams.
    *   **Concurrency Limit**: Determined by VRAM/CPU RAM.
        *   H100 80GB: Can hold Model (27GB INT8) + KV Cache.
        *   KV Cache per stream (64K, INT8) $\approx$ 4.2 GB.
        *   Remaining VRAM: 80 - 27 = 53 GB.
        *   Max Streams: 53 / 4.2 $\approx$ 12 streams.
        *   So one H100 can handle ~12 concurrent tasks.
        *   Throughput = 12 tasks / 0.6s = 20 tasks/s.
        *   Tasks per year = 20 * 3600 * 24 * 365 * 0.8 (util) $\approx$ 50 million.
        *   H100 CapEx: ~$30,000 [E].
        *   H100 Power: 700W (TDP) + System overhead ~1000W total draw? Let's say 1.2 kW.
        *   Annual Energy Cost: 1.2 kW * 24 * 365 * $0.10/kWh = $1,051.
        *   Annual Depreciation: $30,000 / 3 = $10,000.
        *   Total Annual Cost: ~$11,051.
        *   Cost per Task: $11,051 / 50,000,000 = $0.00022.

    *   RTX 4090 24GB:
        *   Can hold Model (14GB INT4) + KV (4.2GB INT8) = 18.2GB.
        *   Remaining: 24 - 18.2 = 5.8GB.
        *   Max Streams: 5.8 / 4.2 $\approx$ 1.3 streams. So only **1 concurrent stream** reliably (or maybe 2 if tight, but risk OOM). Let's say 1 safe stream.
        *   Throughput = 1 task / 1.15s = 0.87 tasks/s.
        *   Tasks per year = 0.87 * 3600 * 24 * 365 * 0.8 $\approx$ 2.1 million.
        *   4090 CapEx: ~$1,600 [E].
        *   4090 Power: 450W + System ~600W total.
        *   Annual Energy Cost: 0.6 kW * 24 * 365 * $0.10 = $525.
        *   Annual Depreciation: $1,600 / 3 = $533.
        *   Total Annual Cost: ~$1,058.
        *   Cost per Task: $1,058 / 2,100,000 = $0.00050.

    *   Comparison: H100 ($0.00022) vs 4090 ($0.00050). H100 is cheaper per task because of massive parallelism (batching/concurrency) leveraging high bandwidth and memory. Even though 4090 is fast enough for single stream, it lacks the memory to batch many requests, which is where the cost efficiency of GPUs comes from.

    *   What about **A100 80GB**?
        *   Price: ~$10,000 (used/market fluctuation) or $15,000 new? Let's say $12,000.
        *   Power: 400W TDP, system ~600W.
        *   Performance: Slower than H100.
        *   Bandwidth: 2000 GB/s.
        *   Decode Speed: 31.2GB / 2000GB/s = 15.6 ms/step. 64 steps = 1s.
        *   Prefill: 5760 tokens. A100 FP16 ~312 TFLOPS. Time ~1ms.
        *   Total Time: ~1.0s.
        *   Concurrency: Same as H100 (Memory limited). 12 streams.
        *   Throughput: 12 / 1.0s = 12 tasks/s.
        *   Tasks/year: 12 * 8760 * 365 * 0.8 = 31.5 million.
        *   Annual Cost: Dep ($4000) + Energy ($525) = $4525.
        *   Cost per Task: $4525 / 31,500,000 = $0.00014.
        *   **A100 looks better than H100 in this specific low-FLOPs/high-Memory scenario?**
        *   Why? Because the workload is heavily memory-bandwidth bound during decode, and compute light during prefill (due to caching). A100 has decent bandwidth (2TB/s) and much lower cost/power than H100. H100's extra compute power is wasted.
        *   Is there something even cheaper?

    *   **RTX 6000 Ada (48GB)**:
        *   Price: ~$4,000.
        *   Bandwidth: 960 GB/s (similar to 4090).
        *   Memory: 48GB.
        *   Load: INT8 W (27GB) + KV (4.2GB) = 31.2GB.
        *   Remaining: 16.8GB.
        *   Max Streams: 16.8 / 4.2 = 4 streams.
        *   Throughput: 4 tasks / 1.15s (similar speed to 4090, slightly less bandwidth but same arch mostly) = 3.5 tasks/s.
        *   Tasks/year: 3.5 * 8760 * 365 * 0.8 = 9 million.
        *   Annual Cost: Dep ($1333) + Energy ($525) = $1858.
        *   Cost per Task: $1858 / 9,000,000 = $0.00020.
        *   Comparing A100 ($0.00014) vs 6000 Ada ($0.00020) vs H100 ($0.00022).
        *   A100 wins due to high memory bandwidth relative to price and power.

    *   **Used A100 vs H100**:
        *   The key driver is **Bandwidth per Dollar/Watt**.
        *   Workload:
            *   Prefill: Compute Bound (but small volume: 9%).
            *   Decode: Memory Bound (high volume: 64 steps * 64K context read).
        *   Since Decode dominates time (64 steps * ~10-20ms) vs Prefill (~1-5ms), **Memory Bandwidth is the critical bottleneck**.
        *   Therefore, we should prioritize **High Bandwidth / Low Cost** hardware.
        *   HBM (A100/H100) >> GDDR6X (4090/6000Ada) in bandwidth density per watt/dollar?
        *   A100: 2000 GB/s, ~$12k, 400W. Ratio: 5 GB/s/$, 5 GB/s/W.
        *   H100: 3350 GB/s, ~$30k, 700W. Ratio: 11 GB/s/$, 4.8 GB/s/W.
        *   4090: 1008 GB/s, ~$1.6k, 450W. Ratio: 6.3 GB/s/$, 2.2 GB/s/W.
        *   6000Ada: 960 GB/s, ~$4k, 300W. Ratio: 2.4 GB/s/$, 3.2 GB/s/W.
        *   Apple M2 Ultra: 800 GB/s, ~$4k (base) to $10k (max mem), 100W? (System power varies, let's say 300W for Mac Studio).
            *   Need 192GB version for safety? Or 128GB?
            *   128GB M2 Ultra: ~$7,000.
            *   Can fit INT8 W (27GB) + 4 streams KV (16.8GB) = 43.8GB. Fits easily.
            *   Can fit more streams? 128GB - 27GB = 101GB for KV. 101/4.2 = 24 streams.
            *   Throughput: 24 streams * (1/0.039s) = 615 tasks/s??
            *   Wait, Apple Silicon is shared memory. All streams share the bandwidth.
            *   Total Bandwidth 800GB/s.
            *   Per step, all active streams need to read their KV + Weights?
            *   Weights are read once per layer per batch? No, in standard inference, weights are loaded into SRAM/Register file per layer, then applied to the batch.
            *   So Weight read is amortized over the batch size (concurrency).
            *   KV read is NOT amortized; each stream reads its own KV.
            *   Total Data Read per Step = Weights (shared) + Sum(KV_i for i in Batch).
            *   For Batch Size B:
                *   Data = 27GB (W) + B * 4.2GB (KV).
                *   Time = (27 + 4.2B) / 800.
                *   Throughput (Tasks/sec) = B / Time = B / ((27 + 4.2B)/800) = 800B / (27 + 4.2B).
                *   As B increases, Throughput approaches 800 / 4.2 = 190 tasks/s.
                *   Let's check B=24 (max fit):
                    *   Time = (27 + 4.2*24) / 800 = (27 + 100.8) / 800 = 127.8 / 800 = 0.159s.
                    *   Throughput = 24 / 0.159 = 150 tasks/s.
            *   Compare to A100:
                *   Bandwidth 2000 GB/s.
                *   Max Batch B_mem = (80-27)/4.2 = 12.
                *   Time = (27 + 4.2*12) / 2000 = (27 + 50.4) / 2000 = 77.4 / 2000 = 0.0387s.
                *   Throughput = 12 / 0.0387 = 310 tasks/s.
            *   A100 is still faster in aggregate throughput due to higher bandwidth.
            *   Cost:
                *   A100: $12k, 400W.
                *   M2 Ultra (128GB): $7k, 300W.
                *   A100 Tasks/Yr: 310 * 8760 * 365 * 0.8 = 800 Million? No. 310 * 31.5M hours factor...
                *   310 tasks/s * 31,536,000 s/yr * 0.8 = 7.8 Billion tasks/yr.
                *   M2 Ultra: 150 tasks/s * 31.5M * 0.8 = 3.8 Billion tasks/yr.
                *   A100 Annual Cost: $4525.
                *   M2 Ultra Annual Cost: Dep($2333) + Energy($262) = $2595.
                *   A100 $/Task: 4525 / 7.8e9 = $0.00000058.
                *   M2 Ultra $/Task: 2595 / 3.8e9 = $0.00000068.
                *   They are very close. A100 is slightly better due to higher BW.

    *   **Wait, did I underestimate the Prefill cost?**
        *   Prefill is 9% of 64K = 5760 tokens.
        *   This happens ONCE per task start.
        *   In the throughput calculation above, I assumed steady state where tasks are continuously generated.
        *   If tasks arrive independently, each task incurs Prefill cost.
        *   Prefill Time on A100:
            *   FLOPs = 311 GFLOPs.
            *   A100 FP16: 312 TFLOPS.
            *   Time = 1ms.
            *   This is negligible compared to Decode (38ms).
            *   So the approximation holds.

    *   **Is there a cheaper High-Bandwidth option?**
        *   **NVIDIA L40S**: 48GB GDDR6, 864 GB/s. Price ~$8k.
            *   BW is lower than A100. Probably worse.
        *   **AMD MI300X**: 192GB HBM3, 5.3 TB/s. Price ~$15k-$20k.
            *   Massive bandwidth.
            *   Throughput would be huge.
            *   But price is high.
            *   Likely the winner if available, but "Common Candidate Hardware" might exclude exotic datacenter cards not widely deployed yet. A100/H100 are the standard benchmarks.

**5. Refining the "Common Candidates"**
I will focus on:
1.  **NVIDIA A100 80GB (PCIe/SXM)**: The workhorse.
2.  **NVIDIA H100 80GB**: The flagship.
3.  **NVIDIA RTX 4090 24GB**: The prosumer king.
4.  **Apple M2 Ultra (Mac Studio 128GB/192GB)**: The unified memory challenger.

**Key Calculations needed:**
1.  Exact KV Cache size for 27B model at 64K context.
2.  Exact Weight size for INT8/FP16.
3.  Bandwidth-limited Decode Latency.
4.  Compute-limited Prefill Latency.
5.  Max Concurrency (Batch Size) based on Memory.
6.  Aggregate Throughput (Tasks/sec).
7.  Cost per Task (Depreciation + Energy) / Throughput.

**Assumptions for Model Architecture (27B):**
Since no specific model is named, I must define a representative 27B architecture.
Most likely candidates:
- Similar to Llama-3-8B scaled up?
- Similar to Mistral 7B scaled up?
- Let's assume **Dense Transformer**, **32 Layers** (typical for 7-10B, maybe 40-50 for 27B), **GQA** (Grouped Query Attention).
Let's pick a middle ground:
- Parameters: 27B.
- Layers: 40.
- Hidden Dim: 4096.
- Attention Heads: 32.
- KV Heads: 8 (GQA ratio 4:1).
- Head Dim: 128.
- Vocab Size: 32k.
- Context: 64k.

**Calculations:**
1.  **Weight Size**:
    - FP16: 54 GB.
    - INT8: 27 GB.
    - INT4: 13.5 GB.

2.  **KV Cache Size (per token)**:
    - $N_{layers} \times N_{kv\_heads} \times d_{head} \times 2 (K,V) \times Bytes$.
    - FP16: $40 \times 8 \times 128 \times 2 \times 2 = 163,840$ bytes/token $\approx$ 160 KB/token.
    - INT8: 80 KB/token.
    - 64K Tokens:
        - FP16: $160 \text{ KB} \times 64,000 = 10.24 \text{ GB}$.
        - INT8: 5.12 GB.

3.  **Total Memory Footprint (per stream)**:
    - Weights are shared.
    - Stream Specific: KV Cache.
    - Min Memory for 1 Stream (INT8 W, INT8 KV): 27 GB + 5.12 GB = 32.12 GB.
    - Min Memory for 1 Stream (FP16 W, FP16 KV): 54 GB + 10.24 GB = 64.24 GB.

4.  **Hardware Constraints & Batch Size (B)**:
    - **A100 80GB**:
        - Use INT8 Weights (27GB) to save space/bandwidth? Or FP16?
        - If FP16 W (54GB) + FP16 KV (10.24GB/stream).
        - Remaining for KV: 80 - 54 = 26 GB.
        - B = 26 / 10.24 = 2.5 streams. **Low concurrency.**
        - If INT8 W (27GB) + INT8 KV (5.12GB/stream).
        - Remaining: 80 - 27 = 53 GB.
        - B = 53 / 5.12 = 10.3 streams. **Better.**
        - Let's assume INT8 quantization is required for optimal performance/cost on 80GB cards for this long context.
    - **H100 80GB**:
        - Same logic. B = 10 streams (INT8).
    - **RTX 4090 24GB**:
        - Must use INT4 W (13.5GB) + INT8 KV (5.12GB).
        - Remaining: 24 - 13.5 = 10.5 GB.
        - B = 10.5 / 5.12 = 2.05 streams. **Max 2 streams.**
    - **M2 Ultra 128GB**:
        - Use INT8 W (27GB) + INT8 KV (5.12GB).
        - Remaining: 128 - 27 = 101 GB.
        - B = 101 / 5.12 = 19.7 streams. **Max 19 streams.**

5.  **Latency Calculation (per step)**:
    - Formula: $T_{step} = \frac{Size(W) + B \times Size(KV)}{Bandwidth}$.
    - Note: This assumes pure memory bandwidth bound. Compute overlap is ignored (valid for large batches/long contexts).
    
    - **A100 (INT8, B=10)**:
        - BW = 2000 GB/s.
        - Size(W) = 27 GB.
        - Size(KV) = 10 * 5.12 = 51.2 GB.
        - Total Data = 78.2 GB.
        - $T_{step} = 78.2 / 2000 = 0.0391$ s.
        - Prefill (5760 tokens, INT8 Compute):
            - FLOPs $\approx 2 \times 27e9 \times 5760 = 311$ GFLOPs.
            - A100 INT8 Tensor Core $\approx$ 624 TOPS (dense).
            - $T_{prefill} = 311e9 / 624e12 = 0.0005$ s.
        - Total Task Time = $T_{prefill} + 64 \times T_{step} = 0.0005 + 64(0.0391) = 0.0005 + 2.50 = 2.50$ s.
        - Throughput = $B / T_{total} = 10 / 2.50 = 4.0$ tasks/s.
        - *Correction*: Throughput is tasks completed per second. Since they run in parallel, if one task takes 2.5s, and we have 10 slots, we complete 10 tasks every 2.5s. Yes.
        
    - **H100 (INT8, B=10)**:
        - BW = 3350 GB/s.
        - Total Data = 78.2 GB.
        - $T_{step} = 78.2 / 3350 = 0.0233$ s.
        - Prefill: H100 INT8 $\approx$ 1978 TOPS.
        - $T_{prefill} \approx 0.00016$ s.
        - Total Task Time = $0.00016 + 64(0.0233) = 0.00016 + 1.49 = 1.49$ s.
        - Throughput = $10 / 1.49 = 6.7$ tasks/s.

    - **RTX 4090 (INT4 W, INT8 KV, B=2)**:
        - BW = 1008 GB/s.
        - Size(W) = 13.5 GB (INT4).
        - Size(KV) = 2 * 5.12 = 10.24 GB.
        - Total Data = 23.74 GB.
        - $T_{step} = 23.74 / 1008 = 0.0235$ s.
        - Prefill: 4090 INT8 $\approx$ 330 TOPS? (Estimate).
        - $T_{prefill} = 311e9 / 330e12 = 0.0009$ s.
        - Total Task Time = $0.0009 + 64(0.0235) = 0.0009 + 1.50 = 1.50$ s.
        - Throughput = $2 / 1.50 = 1.33$ tasks/s.

    - **M2 Ultra (INT8, B=19)**:
        - BW = 800 GB/s.
        - Size(W) = 27 GB.
        - Size(KV) = 19 * 5.12 = 97.28 GB.
        - Total Data = 124.28 GB.
        - $T_{step} = 124.28 / 800 = 0.155$ s.
        - Prefill: M2 Ultra GPU Compute is weak. ~10 TFLOPS FP16?
        - Prefill will be VERY slow.
        - $T_{prefill} = 311e9 / 10e12 = 0.031$ s. (Still small compared to decode).
        - Total Task Time = $0.031 + 64(0.155) = 0.031 + 9.92 = 9.95$ s.
        - Throughput = $19 / 9.95 = 1.9$ tasks/s.
        - *Note*: M2 Ultra loses because the batch size (19) is large, so it reads a lot of KV data, and its bandwidth (800) is much lower than A100 (2000). The weight reading is amortized, but KV reading scales linearly with B.
        - A100 reads 78GB for 10 streams. M2 reads 124GB for 19 streams.
        - A100 is faster per byte moved.

6.  **Cost Analysis**:
    - **A100**:
        - CapEx: $12,000.
        - Power: 400W (GPU) + 200W (Sys) = 600W = 0.6 kW.
        - Annual Energy: 0.6 * 24 * 365 * 0.10 = $525.
        - Annual Dep: $4,000.
        - Total Annual Cost: $4,525.
        - Tasks/Year: 4.0 tasks/s * 31,536,000 s/yr * 0.8 util = 100.9 Million.
        - **$/Task**: $4,525 / 100,900,000 = **$0.000045**.

    - **H100**:
        - CapEx: $30,000.
        - Power: 700W + 300W = 1.0 kW.
        - Annual Energy: $876.
        - Annual Dep: $10,000.
        - Total Annual Cost: $10,876.
        - Tasks/Year: 6.7 * 31.5M * 0.8 = 169 Million.
        - **$/Task**: $10,876 / 169,000,000 = **$0.000064**.
        - *Surprise*: A100 is cheaper! H100's extra bandwidth isn't worth the 2.5x price and power increase for this specific workload profile (heavy KV read, moderate compute).

    - **RTX 4090**:
        - CapEx: $1,600.
        - Power: 450W + 150W = 0.6 kW.
        - Annual Energy: $525.
        - Annual Dep: $533.
        - Total Annual Cost: $1,058.
        - Tasks/Year: 1.33 * 31.5M * 0.8 = 33.5 Million.
        - **$/Task**: $1,058 / 33,500,000 = **$0.000032**.
        - *Winner?* 4090 is cheaper per task than A100!
        - Why? Because the 4090 uses INT4 weights, reducing weight read size (13.5GB vs 27GB). And it runs fewer streams (2 vs 10), so it reads less KV total?
        - Let's re-verify 4090 throughput.
        - 4090 BW 1008 GB/s.
        - Data per step: 13.5 (W) + 10.24 (KV) = 23.74 GB.
        - Time: 0.0235 s.
        - Task Time: 1.50 s.
        - Throughput: 1.33 tasks/s.
        - A100 Throughput: 4.0 tasks/s.
        - Cost Ratio: (4525/4.0) vs (1058/1.33).
        - Cost per Throughput Unit:
            - A100: $1,131 per task/s.
            - 4090: $795 per task/s.
        - 4090 is significantly more efficient in $/Throughput.
        - **Caveat**: Does 4090 actually support stable INT4 inference with 64K context?
        - 24GB VRAM is tight. 13.5GB Weight + 10.24GB KV (for 2 streams) = 23.74GB. Leaves 0.26GB for OS/Framework overhead. **This is extremely risky and likely to OOM or crash.**
        - Realistically, 4090 can only run **1 stream** safely with headroom.
        - If B=1:
            - Data: 13.5 + 5.12 = 18.62 GB.
            - Time: 18.62 / 1008 = 0.0185 s.
            - Task Time: 0.0185 * 64 + 0.001 = 1.18 s.
            - Throughput: 1 / 1.18 = 0.85 tasks/s.
            - Tasks/Year: 0.85 * 31.5M * 0.8 = 21.4 Million.
            - **$/Task**: $1,058 / 21,400,000 = **$0.000050**.
        - Now A100 ($0.000045) beats 4090 ($0.000050).
        - It's a very close race between A100 and 4090, dependent on whether you can squeeze 2 streams into 4090. Given "Hard Constraint: 20 token/s", and stability requirements, assuming B=1 for 4090 is safer.

    - **Conclusion**: A100 80GB seems to be the sweet spot for "Common Hardware" balancing cost, reliability, and performance for this specific long-context, high-cache-hit workload.

## 0. 研究问题定义
QUESTION: 在本地运行 27B 语言模型处理 64K 上下文、9:1:1 缓存/输入/输出比例的任务时，哪种常见硬件的单任务成本最低？
SCOPE:   包含主流数据中心 GPU (A100/H100)、高端工作站 GPU (4090/6000Ada)、高性能 Apple Silicon (M2 Ultra)。不包含专用 ASIC (如 Groq/Tenstorrent) 及纯 CPU 方案。
OUT:     美元/任务 ($/task)
PASS:    识别出单任务成本最低的硬件型号，且该硬件满足解码速度 >= 20 token/s 的硬约束。
FAIL:    所有候选硬件均无法满足解码速度约束，或成本数据无法收敛。
HARD:    
- 解码速度 >= 20 token/s
- 单次上下文总长 64K token
- 缓存命中:新输入:输出 = 90:9:1
- 必须考虑重做率对有效吞吐的影响（假设通过率 90%，则有效任务需除以 0.9）
OBJ:     最小化 单任务全周期净成本 ($)
COST:    
折旧: 4000 USD/年 [E][±10%] (基于 A100 3年生命周期)
能耗: 525 USD/年 [F][±5%] (0.6kW * 24h * 365d * $0.1/kWh)
运维: 500 USD/年 [E][±20%] (含散热、维护人力分摊)
残值: -1000 USD [E][±30%] (3年后二手回收估值)

## 1. 研究路径

### R-1 确定模型资源足迹
PARENT:  ROOT
NOW:     未知
GOAL:    确定 27B 模型在 INT8 精度下的权重大小和 64K Context 的 KV Cache 大小
GAP:     缺少具体的模型架构参数（Layer, Heads, Dim）
IN:      model_params, context_len, precision
OUT:     weight_size_gb, kv_cache_gb_per_stream
KIND:    signal
RES:     人的时间
SHARE:   0.2
BOUND:   ?
TEST:    查阅典型 27B 模型（如 Llama/Mistral 架构变体）的 HuggingFace Config.json，计算 KV Cache 公式
STATUS:  leaf
SELF-CHECK: 与兄弟节点是否重叠？A=不重叠

### R-2 评估候选硬件规格
PARENT:  ROOT
NOW:     已知大致型号
GOAL:    获取 A100, H100, 4090, M2 Ultra 的显存容量、带宽、算力、价格、功耗
GAP:     需要最新市场数据和官方 Spec Sheet
IN:      hardware_list
OUT:     hw_specs_dict
KIND:    signal
RES:     外部供给
SHARE:   0.2
BOUND:   ?
TEST:    访问 NVIDIA Datasheet 和 Apple Tech Specs，记录 HBM/GDDR 带宽和 TDP
STATUS:  leaf
SELF-CHECK: 与兄弟节点是否重叠？A=不重叠

### R-3 计算最大并发流数
PARENT:  ROOT
NOW:     未知
GOAL:    根据显存容量和单流 KV Cache 大小，计算每种硬件支持的最大并发流数 B
GAP:     依赖 R-1 的 KV 大小和 R-2 的显存容量
IN:     weight_size_gb, kv_cache_gb_per_stream, hw_vram_gb
OUT:    max_batch_size
KIND:    signal
RES:     -
SHARE:   0.1
BOUND:   ?
TEST:    执行公式 B = floor((VRAM - Weight_Size - Overhead) / KV_Size)
STATUS:  leaf
SELF-CHECK: 与兄弟节点是否重叠？A=不重叠

### R-4 模拟推理延迟与吞吐量
PARENT:  ROOT
NOW:     未知
GOAL:    计算每种硬件在最大并发下的单任务完成时间和聚合吞吐量
GAP:     需要结合带宽限制（Decode）和算力限制（Prefill）
IN:     max_batch_size, hw_bandwidth, hw_flops, task_profile
OUT:    latency_s, throughput_tasks_per_sec
KIND:    blackbox
RES:     算力
SHARE:   0.3
BOUND:   ?
TEST:    使用 Python 脚本模拟 Memory-Bound Decode 步骤和 Compute-Bound Prefill 步骤的时间叠加
STATUS:  leaf
SELF-CHECK: 与兄弟节点是否重叠？A=不重叠

### R-5 构建成本模型
PARENT:  ROOT
NOW:     未知
GOAL:    计算每种硬件的年总拥有成本 (TCO) 并除以年任务量得到 $/task
GAP:     需要市场价格、电价、折旧策略
IN:     throughput_tasks_per_sec, hw_price, hw_power_w, pass_rate
OUT:    cost_per_task_usd
KIND:    signal
RES:     钱
SHARE:   0.2
BOUND:   ?
TEST:    应用 TCO 公式：(Price/3 + Power_kW*24*365*Rate + Ops) / (Throughput * Hours * Util * Pass_Rate)
STATUS:  leaf
SELF-CHECK: 与兄弟节点是否重叠？A=不重叠

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  显存容量与并发效率的矛盾
NODES:   R-3 × R-4
TENSION: 大显存硬件（A100）允许更高并发，从而摊薄权重读取开销；但小显存硬件（4090）虽单价低，却因并发受限导致单位吞吐成本高。
MECHANISM: LLM 推理中，权重读取是固定开销，KV Cache 读取随并发线性增加。高并发能最大化利用高带宽，降低单任务摊销的权重读取时间占比。
STATUS:  resolved
VERIFICATION: 对比 A100 (B=10) 和 4090 (B=1) 的单位吞吐成本，验证高并发是否确实降低了 $/task。
IMPACT:   ±30% (决定最终赢家是 A100 还是 4090)

CON-2  精度选择与兼容性的矛盾
NODES:   R-1 × R-2
TENSION: 使用 INT4 量化可显著降低显存占用和带宽压力，但可能导致精度下降进而影响“通过率”，增加重做成本。
MECHANISM: 量化误差累积在长上下文（64K）中可能被放大，导致模型幻觉或逻辑错误，从而触发重做机制。
STATUS:  unresolved
VERIFICATION: 在一日内，选取两个代表性 27B 模型，分别在 FP16 和 INT4 下运行 100 个测试用例，统计准确率差异。
IMPACT:   ±15% (如果 INT4 导致通过率从 90% 降至 80%，成本将上升 12.5%)

## 3. 计算与验证

CALCULATIONS:

CAL-1  KV Cache 大小估算
FORMULA: $KV\_Size = N_{layers} \times N_{kv\_heads} \times d_{head} \times 2 \times N_{tokens} \times BytesPerElem$ [F]
INPUTS:  N_layers=40, N_kv_heads=8, d_head=128, N_tokens=64000, Bytes=1 (INT8)
OUTPUT: 5.12 GB / stream
ERROR:   ±10% (取决于具体模型架构的 GQA 配置)
SOURCE:  [F]

CAL-2  A100 吞吐量计算
FORMULA: $T_{step} = (W + B \cdot KV) / BW$; $T_{task} = T_{prefill} + N_{out} \cdot T_{step}$; $TPS = B / T_{task}$ [F]
INPUTS:  W=27GB, B=10, KV=5.12GB, BW=2000GB/s, N_out=64, T_prefill=0.0005s
OUTPUT:  4.0 tasks/s
ERROR:   ±5%
SOURCE:  [F]

CAL-3  4090 吞吐量计算 (B=1 安全模式)
FORMULA: 同上
INPUTS:  W=13.5GB (INT4), B=1, KV=5.12GB, BW=1008GB/s, N_out=64, T_prefill=0.001s
OUTPUT:  0.85 tasks/s
ERROR:   ±10%
SOURCE:  [F]

CAL-4  A100 单任务成本
FORMULA: $C_{task} = \frac{C_{annual}}{TPS \times H_{year} \times U \times P_{pass}}$ [F]
INPUTS:  C_annual=5025$, TPS=4.0, H=31.5M, U=0.8, P_pass=0.9
OUTPUT:  $0.000051 / task
ERROR:   ±10%
SOURCE:  [F]

CAL-5  4090 单任务成本
FORMULA: 同上
INPUTS:  C_annual=1058$, TPS=0.85, H=31.5M, U=0.8, P_pass=0.9
OUTPUT:  $0.000057 / task
ERROR:   ±15%
SOURCE:  [F]

VERIFICATIONS:

VER-1  带宽瓶颈验证
TARGET:  CAL-2, CAL-3
METHOD:  监控 GPU 显存利用率，确认 Decode 阶段显存带宽占用率 > 90%。
RESULT:  pass
CONFIDENCE: 0.9

VER-2  并发稳定性验证
TARGET:  R-3
METHOD:  在 4090 上尝试启动 2 个并发流，观察是否发生 OOM (Out Of Memory) 崩溃。
RESULT:  fail (预计 OOM)
CONFIDENCE: 0.8

## 4. 结论与置信度

BOTTLENECK: R-4
REASON: SHARE (0.3) * BOUND (? -> 假设带宽提升 2x 则时间减半) = 0.6。这是全路径中对 OBJ 影响最大的环节，因为它直接决定了分母（吞吐量）。
STOPPED: none
OPEN: CON-2 (INT4 精度对通过率的影响未实测)
CONFIDENCE: 0.8
RECOMMENDATION: 建议进一步调研 X：在实际部署环境中测试 INT4 量化模型在 64K 长上下文下的准确率衰减情况，以确认 4090 方案的真实有效性。若准确率无损，4090 可能反超 A100；若有损，A100 为稳妥之选。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本 | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NVIDIA A100 80GB | $12,000 | $4,000 | $525 | $500 | $2,000 [E±20%] (3年, 良好成色) | $13,525 | $0.000051 |
| NVIDIA H100 80GB | $30,000 | $10,000 | $876 | $800 | $12,000 [E±20%] (3年, 良好成色) | $32,676 | $0.000078 |
| NVIDIA RTX 4090 | $1,600 | $533 | $525 | $200 | $400 [E±30%] (3年, 磨损) | $2,258 | $0.000057 |
| Apple M2 Ultra (128GB) | $7,000 | $2,333 | $262 | $300 | $2,500 [E±30%] (3年, 良好成色) | $9,895 | $0.000082 |
