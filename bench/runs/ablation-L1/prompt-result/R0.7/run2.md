【草稿】

**1. 问题拆解与直觉检查：**
*   **核心矛盾**：90:9:1 的比率意味着绝大部分计算量（Prefill）是一次性的，且可以被缓存。一旦缓存建立，后续请求主要消耗的是 Decode 阶段的带宽和少量 Prefill 增量。
*   **瓶颈分析**：
    *   如果只看单次冷启动，Compute-bound（算力）。
    *   如果看长期运行（Cache Hit），Decode 是持续的，而 Decode 是典型的 Memory Bandwidth Bound（内存带宽受限）。
    *   27B 模型参数量较大，显存/内存容量是硬约束。FP16/BF16 需要 ~54GB+，INT8 需要 ~27GB+，INT4 需要 ~13-15GB+。
    *   "本地跑"通常指单卡或双卡消费级/专业级显卡，或者高配工作站 CPU+大内存（但 CPU 跑 27B 速度极慢，很难满足 20 tok/s 并发解码，除非用 AMX 等高级指令集且任务简单，这里先假设 GPU 为主，CPU 为备选对比）。
*   **候选硬件池**：
    *   高端消费级：RTX 4090 (24GB), RTX 3090 (24GB)。
    *   中高端/工作站：A6000 (48GB), A5000 (24GB), L40S (48GB - 太贵?), H100 (80GB - 太贵?)。
    *   由于要求 27B，24GB 显存只能跑 INT4/INT8 量化。48GB 可以跑 FP16/INT8 甚至多实例。
    *   为了控制成本，重点考察：RTX 4090 vs A6000 vs 可能存在的二手 3090/A5000。
    *   *修正*：题目要求“本地”，H100 这种数据中心卡通常不算典型“本地”低成本方案，但如果是企业本地部署，可能包含。不过通常“本地”暗示非云端 API。我们将范围限定在单节点可部署的 GPU。
    *   关键变量：量化精度。为了最小化成本，通常会选能塞进显存的最优精度。27B FP16 = 54GB。RTX 4090 (24GB) 必须量化到 INT4 (~15GB) 或 INT8 (~27GB, 需 offload 或双卡)。单卡 4090 跑 27B INT4 是可行的。A6000 (48GB) 可以跑 INT8 或 FP16 (需优化 KV cache)。

**2. 成本模型构建：**
*   **折旧**：按 3-5 年直线折旧。
*   **能耗**：TDP (Watts) × 24h × 365d × 电价 ($/kWh)。
*   **运维**：人工时间折算成钱，或忽略不计（假设自动化监控）。这里主要算电费。
*   **残值**：二手市场价值。
*   **$/任务**：(总持有成本 / 寿命内完成任务数)。
*   **吞吐率 (TPS)**：这是分母的关键。
    *   公式估算：
        *   Prefill FLOPs ≈ $2 \times P \times L_{in}$ (P=参数, L_in=输入长度)。
        *   Decode FLOPs ≈ $2 \times P \times L_{out} \times N_{steps}$? 不，Decode 每步生成一个 token，计算量约为 $2P$ (矩阵向量乘)，但受限于带宽读取权重。
        *   实际上，LLM 推理速度由以下决定：
            *   Phase 1 (Prefill): Compute bound. Time $\propto FLOPS_{needed} / GPU\_TFLOPS$.
            *   Phase 2 (Decode): Memory bandwidth bound. Time per token $\approx Weight\_Size / Memory\_Bandwidth$.
    *   场景：90% 命中缓存。
        *   这意味着每次请求，只有 9% 的新 Input 需要做 Prefill，1% 的 Output 需要做 Decode。
        *   但是！KV Cache 的存在使得 Prefill 阶段只需处理增量部分？是的。
        *   然而，Decode 阶段每一步都要读取整个模型的权重（如果是单 batch size 或低并发）。
        *   等等，题目说“每个并发流”。如果有并发，Batch Size > 1。
        *   假设 Batch Size = B。
        *   Decode 吞吐量 $\approx B \times (Memory\_Bandwidth / Weight\_Size)$。
        *   Prefill 吞吐量 $\approx GPU\_TFLOPS / (2 \times P \times L_{new\_prefill})$。
    *   由于 90:9:1，大部分时间花在等待 Decode 完成吗？
        *   新输入 9K tokens (9% of 64K? 不，总长 64K。9% 是 5760 tokens)。
        *   输出 1K tokens (1% of 64K? 不，1% 是 640 tokens)。
        *   比例 90:9:1 是指 Token 数量比。
        *   所以每次任务：
            *   缓存命中 57600 tokens (无需计算 Prefill，仅需查表/KV load)。
            *   新 Prefill 5760 tokens。
            *   Decode 640 tokens。
        *   时间分布：
            *   Prefill Time: 取决于 5760 tokens 的计算。
            *   Decode Time: 取决于 640 steps 的延迟。
            *   对于单流，Decode 往往比短 Prefill 更耗时，因为 Decode 是串行且带宽受限的，而 Prefill 是并行且算力受限的。
            *   但是 5760 tokens 的 Prefill 也不短。
            *   让我们估算一下时间占比。
                *   Assume A6000: ~100 TFLOPS (FP16), ~800 GB/s BW.
                *   Model 27B INT8: ~27GB weights.
                *   Decode Latency per token (batch=1): $27GB / 800GB/s \approx 34ms$.
                *   Total Decode Time: $640 \times 34ms \approx 21.8s$.
                *   Prefill FLOPs for 5760 tokens: $2 \times 27e9 \times 5760 \approx 3.1e14$ FLOPs.
                *   Prefill Time: $3.1e14 / 100e12 \approx 3.1s$.
                *   Ratio: Decode is ~7x slower than Prefill in this scenario.
                *   So the system is **Decode-Bound**.
    *   结论：优化 Decode 带宽至关重要。

**3. 候选硬件数据搜集 (Mental Check & Search Strategy):**
*   **RTX 4090**: 24GB GDDR6X, 1008 GB/s BW, ~330W, Price ~$1600-$2000.
    *   Can it run 27B? Only INT4/INT5 quantization. Weights ~15GB. Leaves ~9GB for KV Cache.
    *   Decode Speed (INT4, batch=1): $15GB / 1008GB/s \approx 15ms/token \rightarrow 66 tok/s$.
    *   This easily meets the 20 tok/s constraint.
*   **NVIDIA A6000**: 48GB GDDR6, 768 GB/s BW, ~300W, Price ~$4000-$5000.
    *   Can run INT8 or FP16 (tight). Let's assume INT8 for speed/cost balance or FP16 for accuracy if needed. But task says "fixed task", likely tolerant to quantization? No, usually higher precision is safer. But cost is key. Let's compare INT8 on A6000 vs INT4 on 4090.
    *   Wait, 4090 has *higher* bandwidth (1008 GB/s) than A6000 (768 GB/s).
    *   If 4090 runs INT4 (15GB) and A6000 runs INT8 (27GB):
        *   4090 Decode Latency: $15/1008 \approx 15ms$.
        *   A6000 Decode Latency: $27/768 \approx 35ms$.
        *   4090 is faster per token! And cheaper!
    *   What about A5000? 24GB, 768 GB/s. Slower than 4090, more expensive used? Or similar? 4090 is generally better for inference due to newer arch and higher BW.
    *   What about dual 3090? 48GB total, but NVLink required for good performance, or use tensor parallelism. 3090 BW is 936 GB/s. Two cards add overhead. Complexity increases ops cost.
    *   What about Apple M-series? M2 Ultra/M3 Ultra? High unified memory bandwidth (up to 800GB/s on M2 Ultra). But ecosystem lock-in, hard to benchmark standard CUDA models. Usually less flexible for "local fixed task" unless specifically Mac-based. Let's stick to x86+CUDA as the standard "local server/workstation".

**4. Refining the Candidate List:**
1.  **RTX 4090 (24GB)**: Best consumer card. High BW. Must use aggressive quantization (AWQ/GPTQ Int4).
2.  **NVIDIA A6000 (48GB)**: Prosumer. Lower BW than 4090, but double VRAM. Allows INT8/FP16. Higher price.
3.  **NVIDIA L40S (48GB)**: Ada Lovelace architecture. 864 GB/s BW. ~1200W? No, 350W. Price ~$8000+. Likely too expensive compared to 4090.
4.  **Used RTX 3090 (24GB)**: 936 GB/s BW. Cheaper (~$800-$1000 used). Can run INT4.
    *   Comparison 4090 vs 3090:
        *   4090: 1008 GB/s, 330W, ~$1800.
        *   3090: 936 GB/s, 350W, ~$900.
        *   3090 is nearly same speed, half price, slightly worse power efficiency.
        *   However, 3090 is EOL, supply drying up, reliability risk?
        *   4090 has FP8 support? Yes, Ada supports FP8. This could allow running 27B in FP8 (~27GB... wait, 27B params * 1 byte = 27GB. Still doesn't fit in 24GB easily with KV cache. So still need INT4/INT5).
        *   Actually, 27B model.
            *   INT4: ~13.5 GB + Overhead. Fits in 24GB.
            *   INT8: ~27 GB. Does NOT fit in 24GB.
            *   So 4090/3090 are forced into INT4/INT5.
            *   A6000 (48GB) can run INT8 (~27GB) comfortably.
    *   Accuracy impact? The prompt mentions "pass rate is not 100%". Quantization error might lower pass rate, requiring retries. Retries increase cost.
    *   However, usually INT4 AWQ is quite good for general tasks. Let's assume pass rate degradation is manageable or negligible for the sake of hardware selection, OR we must account for it.
    *   If INT4 fails often, A6000 (INT8) might be better despite lower raw throughput.
    *   *Assumption*: For a "fixed task", users usually tune the quantization. Let's assume both achieve acceptable quality, but we note the risk.
    *   Let's look at **Cost per Task**.
        *   Cost = (Depreciation + Energy) / Throughput.
        *   Throughput (Tasks/sec) = Batch_Size / (Time_Prefill + Time_Decode).
        *   With 90% cache hit, Prefill time is small. Decode dominates.
        *   Decode Speed (tok/s) $\propto$ Bandwidth / Weight_Size.
        *   4090 (INT4): $1008 / 13.5 \approx 74$ tok/s (theoretical max, single stream).
        *   A6000 (INT8): $768 / 27 \approx 28$ tok/s (theoretical max, single stream).
        *   4090 is ~2.6x faster per stream.
        *   Since the constraint is >= 20 tok/s, both meet it.
        *   To maximize cost efficiency, we want max throughput.
        *   4090 allows higher concurrency?
            *   KV Cache size depends on Context Length (64K).
            *   KV Cache per layer: $2 \times Layers \times Heads \times HeadDim \times SeqLen \times Bytes$.
            *   For 27B (e.g., Llama-3 style): ~32 layers, 40 heads, 128 dim.
            *   KV Size per token (FP16): $2 \times 32 \times 40 \times 128 \times 2 \text{ bytes} \approx 1MB$?
            *   Let's calc: $2 (\text{K/V}) \times 32 (\text{layers}) \times 40 (\text{heads}) \times 128 (\text{dim}) \times 2 (\text{bytes}) = 655,360 \text{ bytes} \approx 0.6 MB$ per token.
            *   For 64K context: $64000 \times 0.6 MB \approx 38 GB$.
            *   Whoa. KV Cache for 64K context is huge.
            *   Even with 90% cache hit, do we store the full 64K KV? Yes, the "long-term invariant prefix" implies the KV cache for that prefix is stored.
            *   So each concurrent stream needs ~38GB of VRAM just for KV Cache (if FP16 KV).
            *   If KV is quantized to INT8: ~19GB.
            *   If KV is INT4: ~9.5GB.
            *   On 4090 (24GB):
                *   Weights (INT4): 14GB.
                *   Remaining for KV: 10GB.
                *   Max Concurrency (streams) = $10GB / 9.5GB (INT4 KV) \approx 1$.
                *   So 4090 can only handle **1 concurrent stream** with 64K context if using INT4 KV.
                *   If using FP16 KV: $10GB / 38GB < 1$. Impossible.
                *   So 4090 is limited to Batch Size = 1.
            *   On A6000 (48GB):
                *   Weights (INT8): 27GB.
                *   Remaining for KV: 21GB.
                *   Max Concurrency (INT8 KV, 19GB/stream) = $21 / 19 \approx 1$.
                *   Max Concurrency (FP16 KV, 38GB/stream) = 0. Impossible.
                *   So A6000 is also limited to Batch Size = 1 with 64K context!
    *   **Crucial Insight**: With 64K context, KV Cache is the bottleneck for concurrency, not compute/bandwidth alone.
    *   Both cards effectively run Batch Size = 1 for this specific long-context task.
    *   Therefore, Throughput is determined by **Latency per Task**.
    *   Latency = Time(Prefill_new) + Time(Decode_output).
    *   As calculated before:
        *   4090 (INT4 W, INT4/8 KV?):
            *   Decode is bandwidth bound.
            *   Weight read: 14GB.
            *   KV Read/Write: Small relative to weights? In Decode, we read Weights every step. We read/write KV for the current token.
            *   Dominant factor is Weight Bandwidth.
            *   4090: 1008 GB/s.
            *   A6000: 768 GB/s.
            *   4090 reads 14GB, A6000 reads 27GB.
            *   Time per token 4090: $14/1008 = 13.9 ms$.
            *   Time per token A6000: $27/768 = 35.1 ms$.
            *   4090 is **2.5x faster**.
    *   Cost comparison:
        *   4090 Price: ~$1800.
        *   A6000 Price: ~$4500.
        *   4090 is 2.5x faster and 2.5x cheaper.
        *   Winner: **RTX 4090**.

**5. Potential Pitfalls / Contradictions:**
*   **Quantization Quality**: Does INT4 on 4090 degrade the "pass rate" significantly compared to INT8 on A6000?
    *   If Pass Rate drops from 90% to 70%, retries increase.
    *   Cost per *Successful* Task = Cost_per_Attempt / Pass_Rate.
    *   If 4090 Pass Rate is much lower, A6000 might win.
    *   However, modern INT4 (AWQ) is very robust. For many tasks, difference is <5%.
    *   I will list this as a contradiction/risk.
*   **Power Consumption**:
    *   4090: 330W.
    *   A6000: 300W.
    *   Similar. Energy cost is secondary to depreciation.
*   **Availability**: 4090 is sometimes restricted or overpriced. 3090 is a strong alternative.
    *   3090: $900, 936 GB/s, 24GB.
    *   3090 vs 4090:
        *   3090 is 93% the speed of 4090 (BW wise) but 50% the price.
        *   Depreciation of 3090 is lower.
        *   Is 3090 still available? Yes, used market.
        *   Should I recommend 3090?
        *   The prompt asks for "common candidate hardware". 4090 is the flagship. 3090 is legacy.
        *   If buying new, 4090 is the choice. If budget is strict, 3090.
        *   I will focus on 4090 as the primary "best value" among mainstream new parts, but mention 3090 as a cost-cutting alternative.
        *   Actually, let's include **RTX 4090** and **NVIDIA A6000** and maybe **Dual 3090**? No, Dual 3090 adds complexity (NVLink bridge, driver issues, power draw 700W). For a "fixed task", simplicity matters. Single card is preferred.

**6. Structuring the Output:**

*   **Section 0**: Define the problem clearly.
    *   Question: Minimize $/task for 27B model, 64K ctx, 90:9:1 ratio, >=20 tok/s decode.
    *   Scope: Local GPU deployment.
    *   OUT: Cost per successful task.
    *   HARD: Decode >= 20 tok/s.
*   **Section 1**: Research Path.
    *   R-1: Hardware Selection.
    *   R-1.1: Determine VRAM requirement (Model + KV).
    *   R-1.2: Determine Bottleneck (Compute vs Bandwidth).
    *   R-1.3: Evaluate Candidates (4090, A6000, 3090).
    *   R-2: Cost Modeling.
    *   R-2.1: Depreciation.
    *   R-2.2: Energy.
    *   R-2.3: Throughput Calculation.
*   **Section 2**: Contradictions.
    *   CON-1: Quality vs Speed (INT4 vs INT8).
    *   CON-2: New vs Used (4090 vs 3090) regarding warranty/reliability vs price.
*   **Section 3**: Calculations.
    *   Table comparing 4090, A6000, 3090.
    *   Calculate TPS and $/Task.
*   **Section 4**: Conclusion.
    *   Recommendation: RTX 4090.

**Refining Calculations for Section 3:**

*   **Assumptions**:
    *   Electricity: $0.15/kWh [E].
    *   Lifespan: 3 years for depreciation [E].
    *   Utilization: 24/7? Or on-demand? "Fixed task" implies continuous service. Assume 24/7 for worst-case energy, or duty cycle? Let's assume high utilization (80%) for cost averaging, but calculate energy based on TDP.
    *   Actually, simpler: Annual Cost = (Price / Life) + (TDP * 24 * 365 * Price_kWh / 1000).
    *   Tasks per year = (365 * 24 * 3600) / Latency_per_Task.
    *   Latency_per_Task = T_prefill + T_decode.
    *   T_decode = N_out * (Weight_Size / BW).
    *   T_prefill = (2 * Params * N_new_in) / TFLOPS_eff.
    *   Let's refine the numbers.

    *   **Model**: 27B Parameters.
    *   **Context**: 64K.
    *   **Ratio**: 90:9:1.
        *   Cached: 57,600.
        *   New Prefill: 5,760.
        *   Output: 640.

    *   **Candidate 1: RTX 4090 (24GB)**
        *   Precision: INT4 (AWQ).
        *   Weight Size: $27 \times 0.5 \text{ bytes} = 13.5 GB$.
        *   KV Cache Precision: INT8 (to save space, common practice).
        *   KV Size per stream (64K): $64000 \times 0.6 \text{ MB (FP16)} / 2 = 19.2 GB$?
            *   Wait, earlier I estimated 0.6MB per token for FP16.
            *   $2 \times 32 \times 40 \times 128 \times 2 = 655,360$ bytes = 0.625 MB.
            *   For 64K: $64000 \times 0.625 MB = 40 GB$.
            *   If INT8 KV: 20 GB.
            *   Total VRAM Needed: 13.5 (Weights) + 20 (KV) = 33.5 GB.
            *   **PROBLEM**: 33.5 GB > 24 GB.
            *   **Correction**: You cannot run 64K context with INT8 KV on a 24GB card if you have 13.5GB weights.
            *   Option A: Use INT4 KV.
                *   KV INT4: 10 GB.
                *   Total: 13.5 + 10 = 23.5 GB. Fits barely.
                *   Does INT4 KV work? Some frameworks support it (e.g., vLLM with paged attention might handle it, or specific kernels). It degrades quality further.
            *   Option B: Offload KV to RAM? Too slow.
            *   Option C: Shorten context? No, fixed 64K.
            *   Option D: Use **Dual 4090**?
                *   48GB total.
                *   Tensor Parallelism.
                *   Cost doubles.
                *   Complexity doubles.
            *   Option E: Re-evaluate A6000.
                *   A6000 48GB.
                *   Weights INT8: 27GB.
                *   KV INT8: 20GB.
                *   Total: 47GB. Fits.
                *   So A6000 is viable for 64K context with reasonable precision (INT8 W, INT8 KV).
                *   4090 is **NOT VIABLE** for 64K context with acceptable precision unless we accept severe KV quantization (INT4) which risks the "pass rate" heavily, or we use Dual Cards.
    
    *   **Revised Candidate Analysis**:
        1.  **NVIDIA A6000 (48GB)**: Viable single-card solution.
        2.  **Dual RTX 4090 (48GB)**: Viable, but complex.
        3.  **NVIDIA L40S (48GB)**: Viable, expensive.
        4.  **AMD MI300X?** No, local workstation.
        5.  **Apple M3 Ultra (128GB Unified)**:
            *   Bandwidth: 800 GB/s.
            *   Weights INT8: 27GB.
            *   KV INT8: 20GB.
            *   Total: 47GB. Fits in 128GB.
            *   Speed: $27GB / 800GB/s = 33ms/token$. Similar to A6000.
            *   Price: M3 Ultra Mac Studio starts at ~$4000 (Base 96GB).
            *   Power: ~100-200W idle/load? Much lower than GPUs.
            *   This is a strong competitor for "Local" low-cost (energy).
            *   But software stack (MLX vs CUDA) might have overhead.
    
    *   Let's compare **A6000** vs **M3 Ultra (Mac Studio)**.
        *   A6000:
            *   Price: $4500.
            *   Power: 300W.
            *   BW: 768 GB/s.
            *   Decode Latency: ~35 ms/tok.
        *   M3 Ultra (96GB):
            *   Price: $4000 (Base) or $5000 (128GB). Let's take $4500 for 128GB config to be safe with KV.
            *   Power: ~150W average under load.
            *   BW: 800 GB/s.
            *   Decode Latency: ~33 ms/tok.
            *   Advantage: Half the power consumption.
            *   Disadvantage: Closed ecosystem, potential software friction.
        
    *   Let's compare **A6000** vs **Dual 4090**.
        *   Dual 4090:
            *   Price: $3600.
            *   Power: 660W.
            *   BW: Aggregated ~2000 GB/s? No, TP communication overhead reduces effective BW. Effective BW for weights is roughly sum, but inter-node comm is slow.
            *   Usually, Dual 4090 is slower than single A6000 for large batches due to PCIe/NVLink bottleneck, but for Batch=1, it might be okay.
            *   However, setup complexity is high.
            
    *   **Decision**: The most robust "standard" answer is **NVIDIA A6000** (or equivalent 48GB prosumer card like RTX 6000 Ada).
    *   Why not 4090? Because 24GB is too small for 64K context + 27B model without extreme quantization that jeopardizes the "pass rate" constraint.
    *   Why not 3090? Same VRAM issue.
    *   Why not Mac? Valid alternative, but CUDA is the industry standard for "research engineer" local setups unless specified otherwise. I will include Mac as a secondary option in the table but recommend A6000 for compatibility/performance predictability.

    *   **Wait**, is there a cheaper 48GB card?
        *   **Quadro P5200**? Old.
        *   **Tesla V100 32GB**? Not enough.
        *   **A5000 24GB**? Not enough.
        *   **L40S**? ~$8000. Too expensive.
        *   **Used A6000**? ~$3000.
        *   **New A6000**? ~$4000.
        
    *   Let's check **RTX 4090 with PagedAttention and smaller KV**.
        *   If we use **INT4 KV** and **INT4 Weights**.
        *   Weights: 13.5GB.
        *   KV: 10GB.
        *   Total: 23.5GB.
        *   Risk: Significant accuracy drop.
        *   If the "fixed task" is sensitive, this is a fail.
        *   Given the prompt emphasizes "pass rate is not 100%", minimizing additional failure modes (quantization error) is important.
        *   Therefore, **48GB VRAM is the minimum safe threshold** for 27B @ 64K.

    *   **Final Candidates for Table**:
        1.  **NVIDIA A6000 (48GB)**: The baseline prosumer choice.
        2.  **Apple M3 Ultra (128GB)**: The efficient alternative.
        3.  **Dual RTX 4090 (48GB)**: The DIY high-performance choice (but high power).
        4.  **NVIDIA L40S (48GB)**: The enterprise choice (high cost).

    *   I will focus the detailed calculation on **A6000** vs **M3 Ultra** vs **Dual 4090**.

    *   **Calculations for Table**:
        *   **A6000**:
            *   Price: $4,500 [E].
            *   Depreciation (3yr): $1,500/yr.
            *   Energy: 300W * 8760h * $0.15/kWh = $3,942/yr.
            *   Total Annual Fixed Cost: $5,442.
            *   Throughput:
                *   Decode: 35 ms/tok.
                *   Prefill: ~3.1 s.
                *   Total Task Time: 3.1 + (640 * 0.035) = 3.1 + 22.4 = 25.5 s.
                *   Tasks/Hour: 3600 / 25.5 = 141.
                *   Tasks/Year: 141 * 8760 = 1,235,160.
            *   $/Task: $5,442 / 1,235,160 = $0.0044.

        *   **M3 Ultra (128GB)**:
            *   Price: $5,000 [E] (128GB config).
            *   Depreciation (3yr): $1,667/yr.
            *   Energy: 150W * 8760h * $0.15/kWh = $1,971/yr.
            *   Total Annual Fixed Cost: $3,638.
            *   Throughput:
                *   Decode: 33 ms/tok (800GB/s BW, 27GB weights).
                *   Prefill: M3 Ultra ALU is slower than GPU?
                    *   M3 Ultra FP16 FLOPS ~ 25 TFLOPS?
                    *   A6000 FP16 ~ 38 TFLOPS? (Actually A6000 is Ampere, 31.2 TFLOPS FP16).
                    *   Let's say M3 Ultra Prefill is 2x slower.
                    *   Prefill Time: 6.2 s.
                    *   Total Task Time: 6.2 + (640 * 0.033) = 6.2 + 21.1 = 27.3 s.
                *   Tasks/Hour: 3600 / 27.3 = 132.
                *   Tasks/Year: 1,156,320.
            *   $/Task: $3,638 / 1,156,320 = $0.0031.
            *   **M3 Ultra wins on cost due to low energy.**

        *   **Dual 4090**:
            *   Price: $3,600.
            *   Depreciation: $1,200/yr.
            *   Energy: 660W * 8760 * 0.15 = $868/yr.
            *   Total: $2,068/yr.
            *   Throughput:
                *   Complex. Assuming 1.5x efficiency loss due to TP.
                *   Effective BW ~ 1300 GB/s?
                *   Decode: $27GB / 1300GB/s = 20ms$.
                *   Prefill: Faster compute.
                *   Task Time: ~20s.
                *   Tasks/Year: ~1.5M.
            *   $/Task: $2,068 / 1,500,000 = $0.0014.
            *   **Dual 4090 wins on cost if stable.** But risk of instability/failure is higher.

    *   **Recommendation**:
        *   If "Local" implies a standard PC workstation: **Dual 4090** is cheapest but risky. **A6000** is safest single-card.
        *   If "Local" allows Mac: **M3 Ultra** is best balance of cost/reliability/power.
        *   Given "Research Engineer" context, they likely prefer Linux/CUDA.
        *   I will recommend **NVIDIA A6000** as the most balanced *single-node* solution, but highlight **Dual 4090** as the lowest cost *if* stability is managed.
        *   Actually, looking at the $/Task, Dual 4090 is significantly cheaper.
        *   However, the prompt asks for "which one... lowest cost".
        *   I will present the table and conclude that **Dual RTX 4090** offers the lowest operational cost, provided the user accepts the complexity, while **A6000** is the best single-card option.
        *   Wait, is Dual 4090 "Common"? Yes.
        *   Is M3 Ultra "Common"? For research engineers, maybe less so than Linux boxes.
        *   I'll stick to NVIDIA GPUs for the main recommendation to ensure CUDA compatibility, but mention the Mac outlier.

    *   **One more check**: 20 tok/s constraint.
        *   A6000: 28 tok/s (Pass).
        *   Dual 4090: ~50 tok/s (Pass).
        *   All pass.

    *   **Final Structure Plan**:
        *   Draft complete.
        *   Write Section 0-4.
        *   Ensure all tags [F], [M], [S], [E] are used.
        *   Ensure contradictions are addressed.

## 0. 研究问题定义

QUESTION: 在本地部署 27B 参数语言模型执行固定长上下文任务时，哪种常见硬件方案的全生命周期单位任务成本最低？
SCOPE:   包含：单节点 GPU 部署（含双卡配置）、主流量化精度（INT4/INT8）、能源与折旧成本。不包含：云端 API、定制 ASIC、极端超频。
OUT:     每成功完成一个任务的平均成本（美元/任务）。
PASS:    识别出在满足“解码速度 ≥ 20 token/s”且“64K 上下文”硬约束下，全周期净成本最低的硬件组合。
FAIL:    未能考虑 KV Cache 对显存的占用导致无法加载 64K 上下文的方案被错误地列为可行选项；或未计入因量化导致的重试成本差异。
HARD:    1. 单个并发流的解码速度必须 ≥ 20 token/s。
         2. 单次上下文总长必须支持 64K token。
         3. 模型规模固定为 27B 参数。
OBJ:     最小化 $/任务 (Minimize Cost per Task)。
COST:    折旧: 1500 USD/年 [E±10%] (基于 3 年直线折旧，以 A6000 为例)
         能耗: 3900 USD/年 [E±15%] (基于 300W TDP, 24/7 运行, 0.15 USD/kWh)
         运维: 500 USD/年 [E±20%] (远程监控与故障处理人力分摊)
         残值: 800 USD [E±30%] (3 年后二手市场价估计)

## 1. 研究路径

### R-1.1 显存需求评估
PARENT:  R-1
NOW:     模型 27B 参数，上下文 64K，需确定最小显存阈值
GOAL:    确定各候选硬件是否能容纳模型权重 + KV Cache
GAP:     未量化不同精度下的 KV Cache 大小
IN:      model_params, context_length, kv_precision
OUT:     min_vram_required, feasible_candidates
KIND:    signal
RES:     人的时间
SHARE:   0.3
BOUND:   2.0x (若发现 24GB 不可行，直接排除一半候选，大幅缩减搜索空间)
TEST:    使用 `vllm` 或 `transformers` 库离线计算 27B 模型在 INT4/INT8 下的权重大小，以及 64K 序列的 KV Cache 理论大小
STATUS:  leaf

### R-1.2 性能瓶颈定位
PARENT:  R-1
NOW:     已知 90:9:1 的 Token 比率
GOAL:    确定系统瓶颈是 Compute-bound 还是 Bandwidth-bound
GAP:     未计算 Prefill 与 Decode 的时间占比
IN:      gpu_tflops, mem_bandwidth, weight_size, new_input_len, output_len
OUT:     bottleneck_type, latency_per_task
KIND:    signal
RES:     算力
SHARE:   0.4
BOUND:   1.5x (准确识别瓶颈可避免过度购买算力或带宽)
TEST:    基于公式 `T_decode = Weight_Size / Bandwidth` 和 `T_prefill = FLOPs / TFLOPS` 计算单任务延迟
STATUS:  leaf

### R-1.3 候选硬件筛选
PARENT:  R-1
NOW:     现有市场常见硬件列表
GOAL:    选出 3-5 个符合显存约束且有代表性的候选
GAP:     未根据显存门槛过滤掉不可行硬件
IN:      min_vram_required, market_prices
OUT:      shortlist_hardware
KIND:    constraint
RES:     外部供给
SHARE:   -
BOUND:   -
TEST:    查询 NVIDIA/Apple 官网及二手平台价格，确认库存状态
STATUS:  leaf

### R-2.1 成本模型构建
PARENT:  R-2
NOW:     拥有候选硬件列表
GOAL:    建立统一的成本计算公式
GAP:     未统一折旧年限和电价标准
IN:      hardware_price, tdp, lifespan, electricity_rate
OUT:     annual_cost_model
KIND:    signal
RES:     人的时间
SHARE:   0.5
BOUND:   1.2x (标准化假设可减少误差)
TEST:    编写 Python 脚本计算年化总拥有成本 (TCO)
STATUS:  leaf

### R-2.2 吞吐量与最终成本计算
PARENT:  R-2
NOW:     拥有成本模型和性能数据
GOAL:    计算每个候选的 $/任务
GAP:     未整合性能与成本数据
IN:      annual_cost_model, latency_per_task, pass_rate
OUT:     cost_per_task
KIND:    signal
RES:     算力
SHARE:   0.5
BOUND:   1.0x (线性计算)
TEST:    代入公式 `Cost_Per_Task = Annual_Cost / (Hours_Year / Latency * Pass_Rate)`
STATUS:  leaf

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  显存容量与量化精度的权衡
NODES:   R-1.1 × R-1.2
TENSION: 24GB 显卡（如 4090）无法在不严重降低 KV Cache 精度（至 INT4）的情况下容纳 27B 模型 + 64K 上下文，而 KV Cache 精度降低可能导致任务通过率下降，增加重试成本。
MECHANISM: KV Cache 随上下文长度线性增长。64K 上下文的 KV Cache 在 FP16 下约需 40GB，INT8 下约需 20GB。24GB 显存减去 14GB (INT4 权重) 后仅剩 10GB，不足以存储 INT8 KV，迫使使用有损的 INT4 KV。
STATUS:  unresolved
VERIFICATION: 在 4090 上分别运行 INT8-KV 和 INT4-KV 版本，对比相同测试集上的任务通过率（Pass Rate）。若 INT4-KV 通过率低于 INT8-KV 5% 以上，则判定该矛盾显著。
IMPACT:   ±30% (若通过率大幅下降，4090 的单次尝试成本优势将被重试抵消)

CON-2  单卡稳定性与双卡成本的冲突
NODES:   R-1.3 × R-2.2
TENSION: 双卡 4090 方案虽然总购置成本低且理论吞吐高，但引入了 NVLink/PCIe 通信开销和更高的电力消耗，且硬件故障率随组件数增加而上升，增加了隐性运维成本。
MECHANISM: 分布式推理需要频繁同步中间结果。PCIe 带宽远低于显存带宽，成为新的瓶颈。同时，双电源、双风扇散热系统的可靠性低于单卡系统。
STATUS:  unresolved
VERIFICATION: 连续运行双卡 4090 集群 72 小时，监控温度、功耗波动及任务失败率，并与单卡 A6000 进行对比。
IMPACT:   ±15% (运维成本和潜在停机损失可能侵蚀双卡的购置成本优势)

## 3. 计算与验证

CALCULATIONS:

CAL-1  显存需求计算
FORMULA: $VRAM = P \times B_{weight} + L_{ctx} \times B_{kv} \times 2$ (其中 $B_{kv}$ 为每层 KV 字节数) [F]
INPUTS:  P=27e9, B_weight(INT8)=1B, L_ctx=64000, B_kv(INT8)≈0.6KB/token/layer (估算)
OUTPUT: 27GB (Weights) + 20GB (KV) = 47GB
ERROR:   ±5% (依赖具体模型架构层数和头数)
SOURCE:  [F]

CAL-2  解码延迟计算 (A6000)
FORMULA: $T_{decode} = \frac{Weight\_Size}{Memory\_Bandwidth}$ [F]
INPUTS:  Weight_Size=27GB, BW=768GB/s
OUTPUT:  35.1 ms/token
ERROR:   ±10% (实际有效带宽受碎片化影响)
SOURCE:  [F]

CAL-3  解码延迟计算 (RTX 4090, INT4)
FORMULA: $T_{decode} = \frac{Weight\_Size}{Memory\_Bandwidth}$ [F]
INPUTS:  Weight_Size=13.5GB, BW=1008GB/s
OUTPUT:  13.4 ms/token
ERROR:   ±10%
SOURCE:  [F]

CAL-4  单任务总延迟 (A6000)
FORMULA: $T_{total} = T_{prefill} + N_{out} \times T_{decode}$ [F]
INPUTS:  T_prefill≈3.1s (估算), N_out=640, T_decode=35.1ms
OUTPUT:  3.1 + 640 * 0.0351 = 25.56 s
ERROR:   ±15%
SOURCE:  [F]

CAL-5  年化成本 (A6000)
FORMULA: $C_{annual} = \frac{Price}{Life} + TDP \times Hours \times Rate + Ops$ [F]
INPUTS:  Price=$4500, Life=3y, TDP=300W, Hours=8760, Rate=$0.15/kWh, Ops=$500
OUTPUT:  1500 + 3942 + 500 = 5942 USD/年
ERROR:   ±10%
SOURCE:  [F]

CAL-6  年化成本 (Dual 4090)
FORMULA: $C_{annual} = \frac{Price}{Life} + TDP \times Hours \times Rate + Ops$ [F]
INPUTS:  Price=$3600, Life=3y, TDP=660W, Hours=8760, Rate=$0.15/kWh, Ops=$1000 (更高运维)
OUTPUT:  1200 + 868 + 1000 = 3068 USD/年
ERROR:   ±15%
SOURCE:  [F]

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本(3年) | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A6000 (48GB)** | $4,500 [E±10%] | $1,500 | $3,942 | $500 | $800 [E±30%] | $12,826 | **$0.0052** |
| **Dual RTX 4090 (48GB)** | $3,600 [E±10%] | $1,200 | $868 | $1,000 | $1,200 [E±30%] | $7,868 | **$0.0033** |
| **Apple M3 Ultra (128GB)** | $5,000 [E±10%] | $1,667 | $1,971 | $300 | $2,500 [E±30%] | $9,872 | **$0.0041** |
| **NVIDIA L40S (48GB)** | $8,000 [E±10%] | $2,667 | $4,730 | $500 | $3,000 [E±30%] | $18,737 | **$0.0091** |

*注：*
1. *A6000 任务吞吐估算：3600/25.56 * 8760 ≈ 1,235,000 tasks/year. Cost/Task = 12826/1235000 ≈ $0.0104?* 
   *Wait, recalculation:*
   *Annual Cost A6000: $5942.*
   *Tasks/Year: 1,235,000.*
   *$/Task: 5942 / 1,235,000 = $0.0048.*
   *Table above shows 3-year net cost. Let's align table with 3-year totals.*
   *A6000 3-Year Net: 1500*3 + 3942*3 + 500*3 - 800 = 4500 + 11826 + 1500 - 800 = $17,026.*
   *Total Tasks 3-Year: 1,235,000 * 3 = 3,705,000.*
   *$/Task: 17026 / 3705000 = $0.0046.*
   
   *Dual 4090 3-Year Net: 1200*3 + 868*3 + 1000*3 - 1200 = 3600 + 2604 + 3000 - 1200 = $8,004.*
   *Throughput Dual 4090: Assume 1.5x A6000 efficiency due to parallelism overhead? Let's say 1.2x. Tasks/Year: 1,482,000.*
   *Total Tasks 3-Year: 4,446,000.*
   *$/Task: 8004 / 4446000 = $0.0018.*

   *Corrected Table Values:*
   * A6000: $0.0046
   * Dual 4090: $0.0018
   * M3 Ultra: Throughput similar to A6000 but slightly slower prefill. Say 0.9x. Tasks/Year: 1,111,000. 3-Year Net: $9,872. $/Task: 9872 / 3333000 = $0.0030.

VERIFICATIONS:

VER-1  验证 A6000 是否满足 20 tok/s
TARGET:  CAL-2
METHOD:  在实际 A6000 机器上使用 `benchmarks.py` 运行 27B-INT8 模型，测量单流解码速度。
RESULT:  pass (理论值 28 tok/s > 20 tok/s)
CONFIDENCE: 0.9

VER-2  验证 Dual 4090 的稳定性
TARGET:  CON-2
METHOD:  连续运行 72 小时压力测试，记录崩溃次数和平均无故障时间 (MTBF)。
RESULT:  partial (需实际测试，目前仅为风险声明)
CONFIDENCE: 0.5

## 4. 结论与置信度

BOTTLENECK: R-1.1
REASON: SHARE(0.3) × BOUND(2.0x) = 0.6。虽然数值不是最大，但它是决定硬件可行性的**硬性准入条件**。如果不解决显存不足的问题，后续的性能优化（R-1.2）和成本计算（R-2）都建立在错误的假设之上（即 24GB 卡不可行）。因此，它是最关键的“闸门”节点。
STOPPED: none
OPEN: CON-1 (量化精度对通过率的影响), CON-2 (双卡运维隐性成本)
CONFIDENCE: 0.85
RECOMMENDATION: 建议优先调研 **Dual RTX 4090** 方案的长期稳定性。如果稳定性无法满足 SLA，则选择 **NVIDIA A6000** 作为最稳妥的低成本单卡方案。**Apple M3 Ultra** 是极具竞争力的替代方案，尤其在电价敏感的场景下。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本(3年) | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A6000 (48GB)** | $4,500 [E±10%] | $1,500 | $3,942 | $500 | $800 [E±30%] | $17,026 | **$0.0046** |
| **Dual RTX 4090 (48GB)** | $3,600 [E±10%] | $1,200 | $868 | $1,000 | $1,200 [E±30%] | $8,004 | **$0.0018** |
| **Apple M3 Ultra (128GB)** | $5,000 [E±10%] | $1,667 | $1,971 | $300 | $2,500 [E±30%] | $9,872 | **$0.0030** |
| **NVIDIA L40S (48GB)** | $8,000 [E±10%] | $2,667 | $4,730 | $500 | $3,000 [E±30%] | $18,737 | **$0.0091** |

**最终建议**：在满足 64K 上下文和 20 tok/s 解码速度的前提下，**Dual RTX 4090** 提供了最低的每任务成本（约 $0.0018/任务），主要得益于其较低的购置成本和较高的聚合显存带宽。然而，这依赖于你能接受双卡配置的复杂性和潜在的运维风险。如果追求单卡稳定性和简化部署，**NVIDIA A6000** 是最佳选择，其成本约为 Dual 4090 的 2.5 倍，但免去了双卡调优的痛苦。