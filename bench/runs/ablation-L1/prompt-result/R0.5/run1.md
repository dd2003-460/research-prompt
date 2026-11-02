# R0.5 run1

【草稿】

**1. 问题拆解与初步思考：**
*   **核心目标**：最小化单任务成本（$/task）。
*   **关键参数**：27B 模型，64K 上下文，9:1:1 比例（缓存:新输入:输出），并发解码 >= 20 tok/s/流。
*   **隐含假设**：
    *   “缓存命中”意味着 KV Cache 复用。在本地硬件上，这通常意味着长序列推理。
    *   27B 模型参数量较大。如果是 FP16，权重约 54GB；INT8 约 27GB；INT4 约 13.5GB。
    *   64K 上下文的 KV Cache 大小取决于模型架构（Layer, Head, Dim）。以 LLaMA-2/3 类似架构估算：
        *   假设 32 层，GQA (Grouped Query Attention) 或 MHA。
        *   若 MHA (Head_dim=128, Heads=32): $2 \times 32 \times 32 \times 128 \times 2 (\text{bytes for FP16}) \approx 512 \text{KB/token}$。
        *   64K token $\times$ 512 KB = 32 GB 仅 KV Cache。加上权重 54GB (FP16)，总显存需求 > 86GB。这需要 A100/H100 级别或多卡。
        *   若 GQA (Heads=8, KV_Heads=8): $2 \times 32 \times 8 \times 128 \times 2 \approx 128 \text{KB/token}$。
        *   64K token $\times$ 128 KB = 8 GB KV Cache。
        *   权重 INT4 ~14GB + KV 8GB = 22GB。这在 RTX 4090 (24GB) 边缘，但考虑到 Batch Size 和 Activation，可能爆显存或性能极差。
        *   通常 27B 模型（如 Qwen2.5-32B 或 Mistral Small 等）为了支持长上下文和高并发，往往需要更大的显存带宽和容量。
    *   **瓶颈分析**：
        *   **Prefill (新输入)**: 计算密集型 (Compute Bound)。9% of 64K = ~5760 tokens.
        *   **Decode (输出)**: 内存带宽密集型 (Memory Bandwidth Bound). 1% of 64K = ~640 tokens.
        *   **Cache Hit**: 几乎零计算/带宽成本（如果已加载），主要是延迟等待。
        *   **约束**: Decode >= 20 tok/s. 对于 27B 模型，单流 20 tok/s 要求不高，但如果是“每完成一个任务的成本”，我们需要看吞吐量。
        *   实际上，题目问的是“每完成一个任务的成本”。任务包含 64K context 的处理。
        *   由于有 90% 缓存命中，实际计算量主要在新输入的 9% 和输出的 1%。
        *   但是，**KV Cache 的驻留**是必须的。如果不复用 KV Cache，每次都要重新 Prefill 64K，那成本会高很多。题目说“可以被缓存”，暗示我们利用 Prefix Caching。
        *   因此，稳态下，每个新任务只需要处理 9% 的新 Input (Prefill) 和 1% 的 Output (Decode)。
        *   然而，初始状态或冷启动时，可能需要全量 Prefill。但题目问的是“常见候选硬件...每完成一个任务”，通常指稳态运行下的平均成本。
        *   **关键点**：如果系统能保持 KV Cache 常驻显存，那么边际成本极低。但如果并发度高，显存会被多个流的 KV Cache 占满。
        *   让我们估算显存占用：
            *   假设使用 INT4 量化以节省显存。
            *   权重: ~14-16 GB.
            *   KV Cache per stream: 假设 GQA, ~100-150 MB per 10K tokens? 不，前面算过，64K 大约 8-16 GB (取决于精度和架构)。
            *   如果 KV Cache 是 FP16/BF16，64K 很大。如果用 FP8 或 INT8 KV Cache，可以减半。
            *   假设单流显存占用 (Weights + KV) ≈ 16GB (Weights) + 8GB (KV, optimized) = 24GB.
            *   这意味着一张 24GB 显卡 (RTX 4090) 只能跑 1 个并发流？或者更少，因为还要留空间给 OS 和 Activation。
            *   如果要跑高并发（为了降低单位时间成本，虽然题目没明确要求高并发，但“每任务成本”通常随吞吐量增加而下降，直到带宽饱和），我们需要更多显存。
            *   等等，题目硬约束是“每个并发流的解码速度不低于 20 token/s”。它没说必须有多少并发。
            *   但是，如果是“固定任务”，且前缀不变，我们可以把前缀的 KV Cache 预计算并保存在显存中。
            *   如果只跑 1 个流，RTX 4090 可能跑得动吗？
                *   Decode 20 tok/s 对于 27B INT4 在 4090 上是可行的（通常能达到 30-50 tok/s）。
                *   Prefill 5760 tokens 在 4090 上大概需要几秒到十几秒。
            *   如果跑 N 个流，显存不够怎么办？
            *   候选硬件：
                1.  **NVIDIA RTX 4090 (24GB)**: 消费级旗舰。
                2.  **NVIDIA A100 80GB / H100 80GB**: 数据中心级。
                3.  **AMD MI300X / Intel Gaudi**: 其他数据中心选项。
                4.  **Mac Studio (Unified Memory)**: 大内存优势。
                5.  **多卡组合 (e.g., 2x 4090 or 4x 3090)**: 本地服务器方案。

**2. 成本模型构建：**
*   **成本构成**：
    *   折旧 (Depreciation): 硬件价格 / 寿命 (年)。
    *   能耗 (Energy): 功耗 (W) × 电价 ($/kWh) × 时间 (h/task)。
    *   运维 (O&M): 人工、维护、冷却等，通常按硬件价值的一定比例或固定值估算。
    *   残值 (Residual Value): 期末剩余价值。
*   **任务时间估算**：
    *   任务包含：Prefill (9% of 64K = 5760 tokens) + Decode (1% of 64K = 640 tokens).
    *   注意：这里有一个陷阱。如果 90% 是缓存命中，是指这 90% 的 token 不需要重新计算 Prefill，也不需要 Decode。
    *   所以，单个任务的**计算工作量**仅为：
        *   Prefill: 5760 tokens.
        *   Decode: 640 tokens.
    *   **但是**，我们必须考虑**并发**。如果只有 1 个流在跑，GPU 利用率可能很低（尤其是 Decode 阶段，受限于带宽，单流带宽利用率低）。
    *   为了最小化“每任务成本”，我们应该最大化 GPU 利用率。
    *   设 $T_{prefill}$ 为 Prefill 5760 tokens 的时间，$T_{decode}$ 为 Decode 640 tokens 的时间。
    *   总任务时间 $T_{task} = T_{prefill} + T_{decode}$ (串行执行，因为同一请求内是先 Prefill 后 Decode)。
    *   如果并行度为 $P$ (同时处理的请求数)，则总吞吐量为 $P / (T_{prefill} + T_{decode})$ requests/sec? 不对。
    *   正确的吞吐量模型：
        *   所有流共享 GPU。
        *   Prefill 是 Compute Bound，Decode 是 Memory Bound。
        *   混合负载下，GPU 会在两种模式间切换或重叠调度。
        *   简化模型：假设 Prefill 和 Decode 不能完全重叠（通常如此，除非有特殊的 Continuous Batching 且 Prefill 很短）。
        *   更准确的模型：
            *   每个请求经历 Phase 1 (Prefill) 和 Phase 2 (Decode).
            *   在任意时刻，有些请求在 Prefill，有些在 Decode。
            *   为了简化，我们计算**单请求的平均资源占用时间**。
            *   或者，直接计算**每 Token 的成本**，然后乘以每任务的 Token 数。
            *   每任务有效 Token 数 = 5760 (Prefill) + 640 (Decode) = 6400 tokens.
            *   注意：被缓存的 90% (57600 tokens) 不计入计算成本，但计入显存占用成本。
            *   **显存占用成本**：这是关键！即使不计算，KV Cache 也占着显存。显存越小，能并发的流越少，导致 GPU 算力闲置，从而推高单位任务成本。
            *   所以，**显存容量**决定了最大并发数 $N_{max}$。
            *   $N_{max} = \lfloor (VRAM_{total} - VRAM_{weights} - VRAM_{os}) / VRAM_{kv\_per\_stream} \rfloor$.
            *   实际并发数 $N_{actual}$ 应小于等于 $N_{max}$，且需满足 Decode 速度约束。
            *   当 $N$ 增加时，Prefill 吞吐线性增加（直到算力饱和），Decode 吞吐增加较慢（受带宽限制）。
            *   最优 $N$ 通常在带宽/算力平衡点附近。

**3. 硬件调研与参数估算：**

*   **模型规格假设** (基于常见的 27B 模型，如 Qwen2.5-32B 或类似规模，这里取 27B 为近似，实际可能是 32B，但题目指定 27B，我们就按 27B 算，或者假设它是 32B 级别的架构特性):
    *   参数量: 27e9.
    *   架构: Transformer, Layer=32, Hidden=4096, Heads=32, KV_Heads=8 (GQA), Head_Dim=128.
    *   精度:
        *   Weights: INT4 (AWQ/GPTQ) -> ~14 GB.
        *   KV Cache: FP16 (为了精度) 或 INT8 (为了省显存). 假设 FP16 以保证质量，因为任务判定严格。
        *   KV Cache per token: $2 \times Layers \times KV\_Heads \times Head\_Dim \times BytesPerElement$.
        *   $2 \times 32 \times 8 \times 128 \times 2 = 131,072$ Bytes/token $\approx 128$ KB/token.
        *   64K Context KV Cache: $64,000 \times 128$ KB = 8,192,000 KB = 8 GB.
    *   总显存需求 per stream (Weight shared + KV unique):
        *   Weights are shared across streams.
        *   $VRAM_{required} = VRAM_{weights} + N \times VRAM_{kv}$.
        *   $VRAM_{weights} \approx 14$ GB.
        *   $VRAM_{kv} \approx 8$ GB.
        *   Overhead (OS, CUDA ctx, Activations): ~2-4 GB.
        *   Total Fixed Cost: ~18 GB.

*   **候选硬件 1: NVIDIA RTX 4090**
    *   VRAM: 24 GB.
    *   Available for KV: $24 - 18 = 6$ GB.
    *   Max Streams ($N$): $6 / 8 = 0.75$. **只能跑 1 个流！**
    *   甚至可能 OOM，因为 6GB < 8GB.
    *   如果强行跑，必须减小上下文或使用 FP8/INT4 KV Cache.
    *   假设使用 INT8 KV Cache: 4 GB/stream.
    *   Then $N = 6/4 = 1.5 \rightarrow 1$ stream.
    *   结论: 4090 只能跑 1 并发。
    *   Performance (1 stream):
        *   Decode: 27B INT4 on 4090. Speed ~ 30-40 tok/s. Meets 20 tok/s constraint easily.
        *   Prefill: 5760 tokens. Throughput ~ 2000-3000 tok/s. Time ~ 2s.
        *   Decode Time: 640 tokens / 30 tok/s = 21s.
        *   Total Task Time: ~23s.
        *   Throughput: 1 task / 23s = 0.043 tasks/s.

*   **候选硬件 2: NVIDIA A100 80GB**
    *   VRAM: 80 GB.
    *   Available for KV: $80 - 18 = 62$ GB.
    *   Max Streams ($N$, FP16 KV): $62 / 8 = 7.75 \rightarrow 7$ streams.
    *   Let's check performance with 7 streams.
    *   Decode Bandwidth Limit: A100 PCIe is 600 GB/s? No, A100 SXM is 2TB/s, PCIe is ~600GB/s. Let's assume SXM for server.
    *   Memory Bandwidth for Decode: Weight readout dominates.
    *   For 27B INT4, weight size 14GB.
    *   Each decode step reads all weights? No, only the relevant rows, but effectively limited by bandwidth to fetch weights from HBM.
    *   Max Decode Throughput (Total tokens/s) $\approx BW / SizePerToken$.
    *   Actually, for batched decode, it's roughly $BW / (WeightSize + KVRead)$.
    *   Dominant factor is WeightSize if Batch is small. If Batch is large, KV Read grows.
    *   Approximation: Max Total Decode Tokens/s $\approx BW / WeightSizeBytes$.
    *   A100 SXM BW: 2039 GB/s. WeightSize: 14 GB.
    *   Max Total Decode Rate $\approx 2039 / 14 \approx 145$ tokens/s (Total across all streams).
    *   Per Stream Decode Rate: $145 / 7 \approx 20.7$ tokens/s.
    *   This is right at the edge of the 20 tok/s constraint!
    *   Maybe use 6 streams to be safe? $145/6 = 24$ tok/s.
    *   Let's assume $N=6$ optimal for safety and efficiency.
    *   Prefill Compute Limit:
    *   A100 FP16 TFLOPS: 312 TFLOPS.
    *   FLOPs per Prefill Token: $2 \times Params \times 2$ (Matmul) $\approx 4 \times 27e9 = 108$ GFLOP/token.
    *   Max Prefill Rate $\approx 312e12 / 108e9 \approx 2888$ tokens/s.
    *   With 6 streams, if they arrive simultaneously, Prefill queue might form.
    *   But Prefill is short (2s single stream).
    *   Let's estimate Task Time for 6 concurrent streams.
    *   Since Prefill and Decode overlap in time domain (Continuous Batching), we look at steady state throughput.
    *   Bottleneck is likely Decode (Bandwidth) vs Prefill (Compute).
    *   Ratio of Work:
        *   Prefill Work: 5760 tokens.
        *   Decode Work: 640 tokens.
        *   Compute Cost Ratio: Prefill uses Compute, Decode uses Bandwidth.
        *   Effective "Time" cost:
            *   Prefill Time contribution: $5760 / 2888 \approx 2$ s (per stream, if exclusive).
            *   Decode Time contribution: $640 / (145/6) \approx 640 / 24 \approx 26.6$ s (per stream, if exclusive).
            *   Wait, this logic is flawed for parallel systems.
            *   Better approach: Calculate total system capacity.
            *   System can do $R_{prefill}$ prefills/sec and $R_{decode}$ decodes/sec.
            *   Actually, it's a pipeline.
            *   Let's use Little's Law or simple cycle time.
            *   Cycle Time per Task in System:
                *   The system processes 6 tasks concurrently.
                *   Each task takes $T_{sys}$ time to complete.
                *   During $T_{sys}$, each task does 1 Prefill and 1 Decode.
                *   Total Prefill work done by system in $T_{sys}$: $6 \times 5760$ tokens.
                *   Total Decode work done by system in $T_{sys}$: $6 \times 640$ tokens.
                *   Constraint 1 (Compute): $6 \times 5760 / T_{sys} \le R_{prefill\_max}$.
                    *   $34560 / T_{sys} \le 2888 \Rightarrow T_{sys} \ge 12$ s.
                *   Constraint 2 (Bandwidth): $6 \times 640 / T_{sys} \le R_{decode\_max}$.
                    *   $3840 / T_{sys} \le 145 \Rightarrow T_{sys} \ge 26.4$ s.
                *   So, $T_{sys} \approx 26.4$ s.
                *   Throughput: $6 \text{ tasks} / 26.4 \text{ s} = 0.227$ tasks/s.

*   **候选硬件 3: NVIDIA H100 80GB**
    *   VRAM: 80 GB. Same $N=6$ or 7.
    *   BW: 3350 GB/s (SXM).
    *   Compute: 989 TFLOPS (FP16).
    *   Max Decode Rate: $3350 / 14 \approx 239$ tokens/s.
    *   With $N=7$: $239/7 = 34$ tok/s/stream. Safe.
    *   With $N=7$:
        *   Prefill Constraint: $7 \times 5760 / T_{sys} \le 989e12 / 108e9 = 9157$ tok/s.
            *   $40320 / T_{sys} \le 9157 \Rightarrow T_{sys} \ge 4.4$ s.
        *   Decode Constraint: $7 \times 640 / T_{sys} \le 239$.
            *   $4480 / T_{sys} \le 239 \Rightarrow T_{sys} \ge 18.7$ s.
        *   $T_{sys} \approx 18.7$ s.
        *   Throughput: $7 / 18.7 = 0.374$ tasks/s.

*   **候选硬件 4: Mac Studio (M2 Ultra/M3 Ultra?)**
    *   Assume M2 Ultra 192GB Unified Memory.
    *   BW: 800 GB/s.
    *   Compute: Much lower than GPU. ~100+ TOPS (INT8)? Hard to compare directly with TFLOPS.
    *   Usually, Macs are good for inference due to high BW relative to compute, but absolute BW is low compared to A100/H100.
    *   Max Decode Rate: $800 / 14 \approx 57$ tokens/s (Total).
    *   Max Streams (VRAM 192GB): Can run many. $192-18=174$ GB. $174/8 = 21$ streams.
    *   But BW limits total throughput.
    *   If $N=21$:
        *   Decode per stream: $57/21 = 2.7$ tok/s. **FAILS** 20 tok/s constraint.
    *   To meet 20 tok/s/stream:
        *   $N_{max} = TotalRate / 20 = 57 / 20 = 2.85 \rightarrow 2$ streams.
    *   With $N=2$:
        *   Prefill Constraint: Compute is weak. Let's estimate M2 Ultra Prefill speed.
        *   M2 Ultra has 19 cores CPU + 60 GPU cores.
        *   Inference engine (MLX) optimizes well.
        *   Assume Prefill speed ~ 500-1000 tok/s for 27B?
        *   Let's say 800 tok/s.
        *   Prefill Constraint: $2 \times 5760 / T_{sys} \le 800 \Rightarrow T_{sys} \ge 14.4$ s.
        *   Decode Constraint: $2 \times 640 / T_{sys} \le 57 \Rightarrow T_{sys} \ge 22.4$ s.
        *   $T_{sys} \approx 22.4$ s.
        *   Throughput: $2 / 22.4 = 0.089$ tasks/s.
    *   Compare to 4090 (0.043), A100 (0.227), H100 (0.374).
    *   Mac is slower in throughput despite more RAM because BW is too low for high concurrency under strict latency constraint.

*   **Cost Comparison:**
    *   **RTX 4090**:
        *   Price: ~$1600 [E].
        *   Life: 3 years. Depreciation: $533/year.
        *   Power: 450W. Usage: 24/7? Or only when working?
        *   Assumption: Server runs 24/7 for availability, but energy cost scales with utilization?
        *   Usually, "Cost per task" assumes the machine is dedicated.
        *   Energy per year: $450W \times 24 \times 365 / 1000 = 3942$ kWh.
        *   Electricity price: $0.10/kWh [E]. Cost: $394/year.
        *   Total Annual Cost: $533 + 394 + O\&M (\$100) = \$1027$.
        *   Tasks per year: $0.043 \times 365 \times 24 \times 3600 = 1,396,000$ tasks?
        *   Wait. 0.043 tasks/s.
        *   Seconds in year: 31,536,000.
        *   Tasks: $0.043 \times 31,536,000 = 1,356,048$.
        *   Cost per Task: $1027 / 1,356,000 \approx \$0.00075$.

    *   **A100 80GB (Cloud or On-prem?)**:
        *   On-prem A100 is expensive.
        *   Price: ~$15,000 [E] (used/market fluctuation).
        *   Life: 3 years. Depreciation: $5000/year.
        *   Power: 400W (PCIe) or 400W (SXM).
        *   Energy: $400W \times 24 \times 365 / 1000 = 3504$ kWh.
        *   Cost: $350/year.
        *   Total Annual Cost: $5000 + 350 + 200 (O\&M) = \$5550$.
        *   Tasks per year: $0.227 \times 31,536,000 = 7,158,072$.
        *   Cost per Task: $5550 / 7,158,000 \approx \$0.00077$.
        *   Very close to 4090!

    *   **H100 80GB**:
        *   Price: ~$30,000 [E].
        *   Depreciation: $10,000/year.
        *   Power: 700W.
        *   Energy: $700 \times 8760 / 1000 = 6132$ kWh.
        *   Cost: $613/year.
        *   Total Annual Cost: $10,000 + 613 + 200 = \$10,813$.
        *   Tasks per year: $0.374 \times 31,536,000 = 11,794,464$.
        *   Cost per Task: $10,813 / 11,794,000 \approx \$0.00091$.
        *   H100 is more expensive per task due to high capex.

    *   **Wait, did I underestimate 4090?**
        *   4090 has 24GB. I assumed 1 stream.
        *   Is it possible to fit 2 streams?
        *   Weights 14GB. KV 8GB/stream.
        *   $14 + 2 \times 8 = 30$ GB > 24 GB. No.
        *   What if we quantize KV to INT4?
        *   KV 4GB/stream.
        *   $14 + 2 \times 4 = 22$ GB. Fits!
        *   Does INT4 KV degrade quality significantly? For "fixed task" with "judgment", maybe risky. But let's assume it works for the sake of comparison, or stick to FP16 for safety.
        *   If we stick to FP16, 4090 is strictly 1 stream.
        *   However, 4090 has much higher Compute density per dollar than A100?
        *   Let's re-evaluate A100 throughput.
        *   A100 PCIe vs SXM. PCIe A100 has 600GB/s BW.
        *   Max Decode: $600/14 = 42$ tok/s total.
        *   With 6 streams: $42/6 = 7$ tok/s. **FAILS** 20 tok/s constraint.
        *   So PCIe A100 can only run 2 streams ($42/2 = 21$ tok/s).
        *   If $N=2$:
            *   Prefill Constraint: $2 \times 5760 / T_{sys} \le 312e12/108e9 = 2888$ (Compute is same).
                *   $11520 / T_{sys} \le 2888 \Rightarrow T_{sys} \ge 4$ s.
            *   Decode Constraint: $2 \times 640 / T_{sys} \le 42$.
                *   $1280 / T_{sys} \le 42 \Rightarrow T_{sys} \ge 30.5$ s.
            *   $T_{sys} = 30.5$ s.
            *   Throughput: $2 / 30.5 = 0.065$ tasks/s.
            *   Cost per Task (PCIe A100):
                *   Price ~$10,000. Depr $3333/yr.
                *   Power 300W. Energy $2628$ kWh. $262/yr.
                *   Total: $3333+262+200 = 3795$.
                *   Tasks/yr: $0.065 \times 31.5M = 2,047,500$.
                *   Cost: $3795 / 2,047,500 = \$0.00185$.
        *   So PCIe A100 is worse than 4090.
        *   SXM A100 is better ($0.00077$) but requires datacenter setup (cooling, power supply, motherboard).
        *   4090 is plug-and-play.

    *   **What about multiple 4090s?**
        *   2x 4090.
        *   Total VRAM: 48GB.
        *   Weights loaded on both? Or split?
        *   Tensor Parallelism usually splits weights.
        *   If TP=2, each card holds 7GB weights.
        *   KV Cache can be replicated or split.
        *   This gets complex.
        *   Generally, scaling out consumer cards has diminishing returns due to NVLink absence (PCIe bottleneck) and software overhead.
        *   However, for 27B, TP=2 on 4090s is feasible.
        *   Let's assume 2x 4090 behaves like a weaker A100.
        *   BW: 2x 1TB/s = 2TB/s (Theoretical max, actual lower).
        *   Compute: 2x 82 TFLOPS = 164 TFLOPS.
        *   This is comparable to A100 SXM (2TB/s, 312 TFLOPS).
        *   So 2x 4090 might match A100 SXM performance?
        *   Cost: 2x $1600 = $3200.
        *   Depr: $1066/yr.
        *   Power: 2x 450W = 900W.
        *   Energy: $7884$ kWh. $788/yr.
        *   Total: $1066 + 788 + 200 = 2054$.
        *   If perf matches A100 SXM (0.227 tasks/s):
        *   Tasks/yr: 7.1M.
        *   Cost: $2054 / 7.1M = \$0.00028$.
        *   **This looks very promising!**

    *   **Refining the "Fixed Task" and "Pass Rate" aspect.**
        *   "目前的通过率不是 100%. 做不对的任务要重做."
        *   Let Pass Rate be $p$.
        *   Expected attempts per successful task = $1/p$.
        *   Cost per *successful* task = (Cost per attempt) / $p$.
        *   Since $p$ is constant for all hardware (assuming model accuracy is same, which depends on Quantization precision), it acts as a multiplier.
        *   If INT4 causes lower accuracy, $p$ drops, increasing cost.
        *   FP16 is safer.
        *   I will assume FP16 KV and INT4 Weights for all, keeping $p$ constant.

    *   **Final Candidates to Detail:**
        1.  Single RTX 4090 (24GB, FP16 KV, 1 stream).
        2.  Dual RTX 4090 (48GB effective, TP=2, ~2-3 streams?).
            *   With TP=2, weights split. KV can be larger.
            *   If we treat it as one big memory pool for KV (with comm overhead):
            *   Available KV: $48 - 14 - 4 = 30$ GB.
            *   Streams: $30/8 = 3.75 \rightarrow 3$ streams.
            *   Perf: Limited by PCIe interconnect.
            *   Likely slightly less than A100 SXM.
        3.  NVIDIA A100 80GB SXM (Datacenter).
        4.  Apple Mac Studio M2 Ultra 192GB (High RAM, Low BW).

    *   **Comparison Table Construction:**
        *   Need to fill COST factors.
        *   Need to ensure "Residual Value" is estimated.
        *   4090 Residual: 50% after 3 years [E].
        *   A100 Residual: 40% after 3 years [E] (Enterprise depreciation is faster).
        *   Mac Residual: 60% after 3 years [E] (Apple holds value).

    *   **Check Constraints again:**
        *   "Local run": Datacenter hardware (A100/H100) is often not considered "local" in the consumer sense, but "on-premise" is local. I will include A100 as an "On-Premise Server" option.
        *   "Common candidate hardware".
        *   I should probably focus on:
            1.  High-end Consumer GPU (RTX 4090).
            2.  Prosumer/Workstation (RTX 6000 Ada 48GB).
            3.  Datacenter (A100).
            4.  Unified Memory (Mac).

    *   **RTX 6000 Ada 48GB**:
        *   VRAM: 48GB.
        *   Price: ~$6800.
        *   BW: 960 GB/s.
        *   Compute: 91 TFLOPS.
        *   Weights: 14GB.
        *   KV Avail: $48-18=30$ GB.
        *   Streams: $30/8 = 3.75 \rightarrow 3$.
        *   Decode Rate: $960/14 = 68$ tok/s total.
        *   Per stream: $68/3 = 22.6$ tok/s. (Passes 20).
        *   Prefill Rate: $91e12/108e9 = 842$ tok/s.
        *   Prefill Constraint: $3 \times 5760 / T_{sys} \le 842 \Rightarrow T_{sys} \ge 20.5$ s.
        *   Decode Constraint: $3 \times 640 / T_{sys} \le 68 \Rightarrow T_{sys} \ge 28.2$ s.
        *   $T_{sys} = 28.2$ s.
        *   Throughput: $3 / 28.2 = 0.106$ tasks/s.
        *   Cost:
            *   Depr: $6800/3 = 2266$.
            *   Power: 300W. Energy $2628$ kWh. $262$.
            *   Total: $2266+262+200 = 2728$.
            *   Tasks/yr: $0.106 \times 31.5M = 3.3M$.
            *   Cost/Task: $2728 / 3.3M = \$0.00082$.

    *   **Comparison so far:**
        *   4090 (1 stream): $0.00075$.
        *   2x4090 (est): $0.00028$ (Very optimistic, need to verify TP overhead).
        *   RTX 6000 Ada: $0.00082$.
        *   A100 SXM: $0.00077$.
        *   Mac: $0.0015$ (Est).

    *   **Is 2x4090 realistic for "Local"?**
        *   Yes, a gaming PC or workstation with 2 GPUs.
        *   But software support for TP on consumer drivers is tricky (vLLM supports it, but PCIe latency hurts).
        *   If we discount 2x4090 perf by 20% due to comm overhead:
        *   New Throughput: $0.227 \times 0.8 = 0.18$ tasks/s.
        *   Cost/Task: $2054 / (0.18 \times 31.5M) = 2054 / 5.67M = \$0.00036$.
        *   Still cheaper than single 4090.

    *   **What about "Single Task" definition?**
        *   If the user means "I have a queue of tasks, how fast do I clear it?", then throughput is key.
        *   If the user means "I send one task, wait for result, send next", then Latency is key, and Cost is just (Power + Depr) / Tasks_per_year_if_running_24/7.
        *   But "Cost per task" usually implies amortized cost over the lifetime of the hardware, divided by total tasks processed.
        *   So maximizing throughput is the goal.

    *   **One more check: Pass Rate.**
        *   If pass rate is 50%, all costs double. Ranking remains same.

    *   **Structure of Response:**
        1.  Define Question.
        2.  Path: Identify Model Specs -> Identify Hardware Constraints (VRAM/BW/Compute) -> Estimate Throughput per HW -> Calculate Cost per HW -> Compare.
        3.  Contradictions: VRAM vs Compute trade-off. Consumer vs Datacenter Capex vs Opex.
        4.  Calculations: Show the table.
        5.  Conclusion: Recommend 2x4090 or Single 4090 depending on risk appetite. Given "Local", 2x4090 is a strong contender for lowest cost if stability is managed. Single 4090 is safest.

    *   **Wait, let's look at the "9:1:1" ratio again.**
        *   90% Cache Hit.
        *   This implies the KV Cache for the prefix is ALREADY in memory.
        *   My calculation of $VRAM_{kv}$ included the full 64K context.
        *   Is the "Prefix" part of the 64K? Yes.
        *   So the 8GB KV cache is for the *entire* 64K window.
        *   The fact that 90% is cached means we don't *compute* those tokens, but we still *store* their KV.
        *   So my VRAM estimate is correct.
        *   My Compute estimate (only calculating 9% + 1%) is correct.

    *   **Final Polish of Numbers:**
        *   I need to be precise with the "Standard" assumptions.
        *   Model: 27B, INT4 Weights, FP16 KV.
        *   Architecture: LLaMA-like, GQA.
        *   Hardware Prices: Use average market prices [E].
        *   Electricity: $0.10/kWh.

    *   **Self-Correction on 2x4090:**
        *   Managing 2 GPUs locally adds complexity (heat, PSU, case).
        *   Maintenance cost increases.
        *   I will present Single 4090 and A100 as primary, and mention Multi-GPU as a high-risk/high-reward optimization.
        *   Actually, the prompt asks for "Which one... lowest cost".
        *   I will conclude with the specific hardware that wins.

    *   **Drafting the Content:**
        *   **Section 0**: Definition.
        *   **Section 1**: Research Path.
            *   R-1: Model Resource Estimation.
            *   R-2: Hardware Candidate Selection.
            *   R-3: Performance Modeling (Throughput).
            *   R-4: Cost Modeling.
        *   **Section 2**: Contradictions.
            *   VRAM Capacity vs Concurrency.
            *   Capex vs Opex (4090 vs A100).
        *   **Section 3**: Calculations.
            *   Table comparing 4090, A100, RTX 6000 Ada.
        *   **Section 4**: Conclusion.
            *   Winner: RTX 4090 (Single) or 2x4090.
            *   Let's stick to **RTX 4090** as the robust "Local" answer, noting that multi-GPU is theoretically cheaper but practically harder.
            *   Wait, if 2x4090 is 50% cheaper, it's a significant difference.
            *   I will recommend **Dual RTX 4090** if the user has technical capability, otherwise **Single RTX 4090**.
            *   But the question asks "Which one...". Singular.
            *   I will argue that **RTX 4090** is the best *single-unit* choice, but **Dual 4090** is the *system* choice.
            *   Let's provide the analysis for both and pick the lowest.

    *   **Refining the "Pass Rate" impact**:
        *   It's a multiplier. I will note it doesn't change ranking.

    *   **Final Check of Rules**:
        *   Chinese output.
        *   5 sections.
        *   Draft first.
        *   Specific tags [F], [M], [S], [E].
        *   Cost breakdown required.

    *   **Ready to write.**
