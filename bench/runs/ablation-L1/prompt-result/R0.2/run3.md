# R0.2 run3

【草稿】

**1. 问题拆解与物理直觉**
*   **任务特征**：27B 模型，64K 上下文。关键比例是 90:9:1 (Cache Hit : Prefill : Decode)。
    *   这意味着大部分时间（90%）是在做 "KV Cache 读取" 和 "Attention 计算"（如果前缀不变，通常只读 KV，或者重算 Attention 但跳过 QKV 投影？不，通常 Prefix Caching 意味着 KV Cache 常驻显存，Prefill 阶段只处理新的 9% token，Decode 阶段每步都要读整个 64K 的 KV Cache）。
    *   **瓶颈分析**：
        *   **Prefill (9%)**: 计算密集型 (Compute-bound)。需要高 FLOPS。
        *   **Decode (1%)**: 内存带宽密集型 (Memory-bandwidth bound)。每一步解码都需要从 HBM/DRAM 中读取整个模型的权重 + 整个 Context 的 KV Cache。
        *   **Prefix (90%)**: 这里有个歧义。“缓存命中的 token”是指这部分 token 不需要重新计算 Prefill，但其 KV Cache 必须存在于显存中供 Decode 阶段使用。
        *   **KV Cache 大小估算**：
            *   假设 FP16/BF16 精度。
            *   27B 模型，层数 L，隐藏维度 D，头数 H。
            *   单 token KV size ≈ $2 \times L \times D_{head} \times N_{heads}$ (Key + Value)。
            *   粗略估算：Llama-3-70B 是 80 层，D=8192, H=64。27B 可能是类似架构或更小。假设 L~32-40, D~4096-5120。
            *   单 token KV (FP16) $\approx 2 \times 40 \times 4096 \times 2$ bytes? 不对，GQA/MQA 会减少。假设标准 MHA，单 token KV $\approx 2 \times L \times D \times 2$ bytes (K and V are same dim as hidden usually per head group).
            *   更简单的估算：经验法则，FP16 下，1M token 的 KV cache 大约占用 1GB - 2GB 显存（取决于模型具体配置）。
            *   64K token = 0.064 M。所以 KV Cache $\approx 0.064 \times 1.5 \text{ GB} \approx 0.1 \text{ GB}$。
            *   等等，这个估算太小了。让我重新查一下。
            *   Llama-2-70B: 80 layers, 8192 dim, 64 heads. KV per token per layer = $2 \times 8192 \times 2$ bytes (FP16) = 32KB. Total per token = $80 \times 32\text{KB} = 2.56 \text{MB}$.
            *   64K tokens $\times 2.56 \text{MB/token} = 163,840 \text{MB} \approx 160 \text{GB}$.
            *   如果是 GQA (Grouped Query Attention)，比如 8 groups，则 KV 缩小到 1/8，约 20GB。
            *   27B 模型（如 Yi-27B, Command-R+ 等）通常使用 GQA。假设 KV Cache 占用在 20GB - 40GB 之间（FP16）。
            *   模型权重本身：27B $\times$ 2 bytes (FP16) = 54GB。如果是 INT8，则是 27GB。INT4 则是 ~14GB。
            *   **总显存需求**：
                *   FP16: Weights (54GB) + KV (30GB avg) + Activations/Overhead (~5GB) $\approx$ 90GB+。这需要 A100/H100 级别的大显存卡，或者多卡。
                *   INT8: Weights (27GB) + KV (15GB if quantized or stays FP16?) + Overhead $\approx$ 50-60GB。RTX 4090 (24GB) 跑不了单卡。A5000 (24GB) 也跑不了。需要 A6000 (48GB) 勉强，或者两张 4090/A5000。
                *   INT4: Weights (14GB) + KV (15-30GB, KV usually kept higher precision for accuracy) + Overhead $\approx$ 40-50GB。
    *   **并发流限制**：
        *   每个流独立占用一份 KV Cache。
        *   如果并发数为 N，总 KV 占用 = $N \times \text{KV}_{single}$。
        *   解码速度 $\ge$ 20 tok/s/stream。
        *   Decode 吞吐量公式：$\text{Throughput} \approx \frac{\text{Memory Bandwidth}}{\text{Bytes Read Per Step}}$.
        *   Bytes Read Per Step $\approx \text{Model Weights Size} + \text{Total KV Cache Size of Active Streams}$.
        *   注意：Weights 被所有 stream 共享读取（如果 batch 在一起），但 KV 是每个 stream 独立的。
        *   对于大 Batch 的 Decode，主要瓶颈是 KV Cache 的读取带宽。
        *   设 $B_w$ 为权重字节数，$B_{kv}$ 为单流 KV 字节数，$N$ 为并发数。
        *   每步读取数据量 $D_{read} \approx B_w + N \times B_{kv}$ (假设 Weight 只需读一次，KV 需读 N 次)。
        *   每流速度 $v = \frac{\text{Bandwidth}}{D_{read}} \times N$ ? 不对。
        *   总 Token/s = $\frac{\text{Bandwidth}}{B_w + N \times B_{kv}}$.
        *   每流速度 $v_{stream} = \frac{\text{Total Throughput}}{N} = \frac{\text{Bandwidth}}{N(B_w + N \times B_{kv})}$.
        *   约束：$v_{stream} \ge 20$.
        *   即 $\frac{\text{Bandwidth}}{N B_w + N^2 B_{kv}} \ge 20$.

**2. 候选硬件调研**
*   **本地常见高端 GPU**：
    1.  **NVIDIA RTX 4090**: 24GB VRAM, ~1 TB/s BW, ~$1600-$2000 [E].
    2.  **NVIDIA RTX A6000 / A5000**: 48GB / 24GB VRAM, ~1 TB/s (A6000 is ~1.1TB/s), ~$4000-$6000 [E].
    3.  **NVIDIA L40S / A100 / H100**: 数据中心级，本地部署较少见但“常见候选”可能包含二手或租赁。
        *   A100 80GB: ~1.6 TB/s BW, ~$8k-$15k used [E].
        *   H100 80GB: ~3.35 TB/s BW, ~$25k-$30k [E].
    4.  **Apple M2 Ultra / M3 Ultra**: 统一内存，带宽极高 (M2U ~800GB/s, M3U ~800GB/s? No, M2 Max is 400GB/s, M2 Ultra is 800GB/s). 容量可达 128GB-192GB. 价格 ~$4k-$10k [E].
    5.  **AMD MI300X**: 192GB HBM3, 5.3 TB/s BW, ~$10k-$15k [E]. (较新，本地少见，但作为高性能候选值得考虑).

*   **筛选**：
    *   4090 (24GB): 无法容纳 27B INT8 + 64K KV (即使 GQA, 27GB weights + 20GB KV > 24GB). 除非用 INT4 且 KV 压缩极狠，或者 offload (速度慢，违反 20tok/s). 排除单卡 4090 方案，除非双卡 NVLink (4090 不支持 NVLink). 双卡 PCIe 延迟高，Decode 性能差。
    *   A6000 (48GB): 可容纳 INT8 (27GB W + 15GB KV + 5GB Other = 47GB). 极限运行。并发数 N 受限于 KV 空间。
    *   A100 80GB: 舒适区。可跑 FP16 或 INT8 高并发。
    *   M2 Ultra (128GB): 统一内存，带宽 800GB/s。比 A100 (1.6TB/s) 慢一半，但容量大。
    *   MI300X: 带宽之王，容量大。

**3. 成本模型构建**
*   **OBJ**: Cost per successful task.
*   **Cost Factors**:
    *   CapEx (Hardware Purchase).
    *   OpEx (Electricity).
    *   Depreciation (Assume 3 years straight-line to 0 or residual value).
    *   Retry Factor: Pass rate $P < 1$. Expected attempts per success = $1/P$.
*   **Task Time Calculation**:
    *   One task consists of:
        1.  Prefill 9% of 64K = 5760 tokens.
        2.  Decode 1% of 64K = 640 tokens.
    *   Wait, the prompt says "Context total length 64K". "Cache hit : New input : Output = 90:9:1".
    *   This implies the *input* to the model for this specific request has a structure where 90% is already cached (from previous turns or system prompt), 9% is new prompt, 1% is generation.
    *   So, computational work per task:
        *   Prefill: Process 9% of 64K = 5,760 tokens.
        *   Decode: Generate 1% of 64K = 640 tokens.
    *   Time taken $T_{task}$:
        *   $T_{prefill} = \frac{5760}{\text{Prefill Speed}}$.
        *   $T_{decode} = \frac{640}{\text{Decode Speed}}$.
    *   Since we are constrained by Decode Speed $\ge 20$ tok/s, let's assume we operate at the limit or slightly above to maximize throughput/cost efficiency? No, cost is per task. Slower speed means more electricity time but maybe lower hardware cost? Usually, faster hardware costs more. We need to find the sweet spot.
    *   However, "Local running" usually implies fixed hardware. The variable is which hardware to buy.
    *   Electricity cost is proportional to Power Draw $\times$ Time.
    *   Depreciation is fixed per year, divided by tasks/year.
    *   Tasks/year depends on Throughput. Higher throughput -> More tasks/day -> Lower depreciation cost per task. But higher power draw during peak load? Or just idle power when not running?
    *   Assumption: Server runs 24/7. Idle power vs Active power.
    *   Let's simplify: Cost = $(\text{Depreciation Rate} + \text{Power Cost}) / \text{Tasks per Day} \times (\text{Attempts per Success})$.
    *   Actually, Power Cost is usually dominant in long-term, but Depreciation is huge upfront.
    *   Let's estimate Tasks per Second (TPS) for each hardware.
    *   TPS = $\min(\text{Prefill Limit}, \text{Decode Limit})$.
    *   Given 9:1 ratio, Prefill is 9x the tokens of Decode.
    *   If Decode speed is 20 tok/s/stream, and we have N streams...
    *   Let's look at the bottleneck.
    *   For large contexts (64K), Decode is heavily memory bandwidth limited.
    *   Prefill is compute limited.
    *   Ratio of Compute to Memory access changes with batch size.
    *   At high batch sizes (high concurrency), Decode becomes the bottleneck because KV cache reading scales linearly with N, while Weights are shared.
    *   Let's calculate Max Concurrent Streams $N_{max}$ based on VRAM.
    *   Then calculate Throughput at $N_{max}$.
    *   Then calculate Cost/Task.

**4. Detailed Hardware Comparison**

**Candidate A: NVIDIA RTX 4090 (24GB)**
*   Can it run 27B?
    *   Quantization required. INT4/INT8.
    *   INT4 Weights: ~14GB.
    *   KV Cache (FP16/GQA): ~20-30GB.
    *   Total > 24GB. **Impossible** for single card without offloading. Offloading kills speed (<20 tok/s likely).
    *   **Verdict**: Excluded due to Hard Constraint (VRAM).

**Candidate B: NVIDIA RTX A6000 (48GB)**
*   Price: ~$5,000 [E] (Used/New market fluctuation).
*   VRAM: 48GB.
*   Precision: INT8 (Weights 27GB).
*   Remaining for KV: $48 - 27 - 5 (\text{overhead}) = 16$ GB.
*   KV per stream (64K, GQA, FP16): Estimated ~20-25GB?
    *   Recalculate KV size carefully.
    *   Model: 27B. Assume architecture similar to Llama-3-70B scaled down or Yi-27B.
    *   Yi-27B: 60 layers? No, let's check typical 27B specs.
    *   Common 27B models: `Yi-27B` (60 layers, 4096 dim, 64 heads, GQA 8).
    *   KV per token: $2 \times \text{Layers} \times \text{Dim} \times \text{Bytes}$. With GQA, Dim is effectively reduced for KV storage? No, GQA reduces number of KV heads.
    *   Standard MHA: KV dim = Hidden Dim.
    *   GQA: KV dim = Hidden Dim / Group Size? No, KV heads are fewer.
    *   Formula: $2 \times L \times D_{kv} \times 2$ (bytes).
    *   If GQA 8, $D_{kv}$ effective width is smaller?
    *   Let's use a rule of thumb: 1M tokens $\approx$ 1-2 GB for 7B, scaling linearly with params? No, scaling with Layers x Dim.
    *   7B (32 layers, 4096 dim): 1M tokens $\approx$ 1GB.
    *   27B (60 layers, 4096 dim?): Layers increased by ~2x. So 1M tokens $\approx$ 2GB.
    *   64K tokens = 0.064M.
    *   KV per stream $\approx 0.064 \times 2 \text{GB} = 0.128 \text{GB}$?
    *   **WAIT**. My previous estimate was 160GB for 70B.
    *   70B (80 layers, 8192 dim).
    *   27B (60 layers, 4096 dim).
    *   Ratio: $(60/80) \times (4096/8192) = 0.75 \times 0.5 = 0.375$.
    *   So KV for 27B is ~37.5% of 70B.
    *   70B @ 64K $\approx$ 160GB (MHA).
    *   27B @ 64K (MHA) $\approx$ 60GB.
    *   27B @ 64K (GQA 8) $\approx$ 60 / 8 = **7.5 GB**.
    *   Okay, this changes everything.
    *   If KV is only 7.5GB per stream, then A6000 (48GB) can fit:
        *   Weights (INT8): 27GB.
        *   Overhead: 5GB.
        *   Remaining: 16GB.
        *   Max Streams $N = 16 / 7.5 \approx 2$ streams.
    *   Is that enough?
    *   Decode Speed Check:
        *   Bandwidth A6000: ~1.1 TB/s.
        *   Bytes read per step: Weights (27GB) + N*KV (2*7.5=15GB) = 42GB.
        *   Steps/sec = $1100 \text{GB/s} / 42 \text{GB} \approx 26$ steps/sec.
        *   Throughput = 26 tokens/sec (total).
        *   Per stream = $26 / 2 = 13$ tok/s.
        *   **FAIL**: 13 < 20.
    *   Try FP16 Weights? No, doesn't fit.
    *   Try INT4 Weights?
        *   Weights: 14GB.
        *   Remaining: $48 - 14 - 5 = 29$ GB.
        *   Max Streams $N = 29 / 7.5 \approx 3$ streams.
        *   Bytes read: 14GB (W) + 3*7.5 (KV) = 36.5GB.
        *   Steps/sec = $1100 / 36.5 \approx 30$ steps/sec.
        *   Per stream = $30 / 3 = 10$ tok/s. **FAIL**.
    *   Why so slow? Because reading 14GB of weights every single token is expensive.
    *   To get 20 tok/s per stream, we need higher bandwidth or smaller weights/KV footprint.
    *   Or, does the "Cache Hit" part mean we don't read weights for those 90%? No, weights are always read for attention/output projections. Only the *compute* of prefill is saved. The *memory read* of weights happens every decode step.
    *   Conclusion: A6000 is likely too weak for 20 tok/s *per stream* at 27B scale with 64K context, unless we use very aggressive quantization (INT4 KV? risky) or multi-GPU.
    *   Multi-GPU 2xA6000? Complexity increases.

**Candidate C: NVIDIA A100 80GB SXM/PCIe**
*   Price: Used ~$8,000 [E].
*   VRAM: 80GB.
*   Bandwidth: 1.6 TB/s (PCIe) or 2.0 TB/s (SXM). Let's use 1.6 TB/s conservative.
*   Precision: INT8.
*   Weights: 27GB.
*   Remaining for KV: $80 - 27 - 5 = 48$ GB.
*   Max Streams $N = 48 / 7.5 = 6$ streams.
*   Decode Speed Check:
    *   Bytes read: 27GB (W) + 6*7.5 (KV) = 72GB.
    *   Steps/sec = $1600 / 72 \approx 22$ steps/sec.
    *   Per stream = $22 / 6 \approx 3.6$ tok/s. **FAIL HARD**.
*   What? Why is it so low?
    *   Ah, the formula $Steps/sec = BW / (W + N \cdot KV)$ gives *Total* Tokens/sec.
    *   Total Throughput = 22 tokens/sec.
    *   If I have 6 streams, each gets $22/6 = 3.6$ tokens/sec.
    *   The constraint is **20 tokens/sec PER STREAM**.
    *   This implies I cannot batch heavily if the model is large relative to bandwidth.
    *   Let's check N=1.
    *   Bytes read: 27 + 7.5 = 34.5GB.
    *   Steps/sec = $1600 / 34.5 \approx 46$ steps/sec.
    *   Per stream (N=1) = 46 tok/s. **PASS**.
    *   So A100 can do 1 stream at 46 tok/s.
    *   Can it do 2 streams?
    *   Bytes read: 27 + 15 = 42GB.
    *   Steps/sec = $1600 / 42 \approx 38$ steps/sec.
    *   Per stream = $38 / 2 = 19$ tok/s. **FAIL** (Just under 20).
    *   So A100 effectively supports **1 concurrent stream** satisfying the 20 tok/s constraint.
    *   Throughput = 1 task / ($T_{prefill} + T_{decode}$).
    *   $T_{decode} = 640 / 46 \approx 14$ sec.
    *   $T_{prefill}$: Compute bound.
        *   A100 FP16 TFLOPS: ~312 TFLOPS (dense).
        *   Ops for Prefill 5760 tokens:
            *   MatMul ops $\approx 2 \times \text{Params} \times \text{Tokens}$.
            *   $2 \times 27 \times 10^9 \times 5760 \approx 3.1 \times 10^{14}$ FLOPs.
            *   Time = $3.1 \times 10^{14} / (312 \times 10^{12}) \approx 1$ sec.
            *   Plus overhead, say 2-3 sec.
    *   Total Task Time $\approx 14 + 3 = 17$ sec.
    *   Throughput $\approx 1/17 \approx 0.06$ tasks/sec $\approx 5184$ tasks/day.

**Candidate D: Apple M2 Ultra (128GB Unified Memory)**
*   Price: ~$4,000 - $5,000 (Mac Studio) [E].
*   VRAM: 128GB (Shared).
*   Bandwidth: 800 GB/s.
*   Precision: INT8 (llama.cpp/vLLM support).
*   Weights: 27GB.
*   Remaining: $128 - 27 - 10 (\text{OS/App}) = 91$ GB.
*   Max Streams $N$: Limited by speed, not VRAM.
*   Check N=1:
    *   Bytes read: 27 + 7.5 = 34.5GB.
    *   Steps/sec = $800 / 34.5 \approx 23$ steps/sec.
    *   Per stream = 23 tok/s. **PASS**.
*   Check N=2:
    *   Bytes read: 27 + 15 = 42GB.
    *   Steps/sec = $800 / 42 \approx 19$ steps/sec.
    *   Per stream = $19 / 2 = 9.5$ tok/s. **FAIL**.
*   So M2 Ultra also supports **1 concurrent stream**.
*   Throughput comparison with A100:
    *   M2U Decode Speed: 23 tok/s vs A100 46 tok/s.
    *   M2U is slower in Decode.
    *   Prefill: M2U Neural Engine/Core performance?
        *   M2U Core FP16 perf is lower than A100 Tensor Cores.
        *   A100 Prefill ~1-2s. M2U might be ~5-10s.
    *   Task Time M2U: $640/23 (27s) + 10s (Prefill) \approx 37s$.
    *   Throughput M2U: $1/37 \approx 0.027$ tasks/sec $\approx 2300$ tasks/day.
    *   A100 Throughput: ~5000 tasks/day.

**Candidate E: AMD MI300X (192GB)**
*   Price: ~$10,000 - $12,000 [E].
*   VRAM: 192GB.
*   Bandwidth: 5.3 TB/s.
*   Precision: INT8.
*   Weights: 27GB.
*   Check N=1:
    *   Bytes read: 34.5GB.
    *   Steps/sec = $5300 / 34.5 \approx 153$ steps/sec.
    *   Per stream = 153 tok/s. **PASS**.
*   Check N=4:
    *   Bytes read: 27 + 4*7.5 = 57GB.
    *   Steps/sec = $5300 / 57 \approx 93$ steps/sec.
    *   Per stream = $93 / 4 \approx 23$ tok/s. **PASS**.
*   Check N=5:
    *   Bytes read: 27 + 37.5 = 64.5GB.
    *   Steps/sec = $5300 / 64.5 \approx 82$ steps/sec.
    *   Per stream = $82 / 5 \approx 16.4$ tok/s. **FAIL**.
*   So MI300X supports **4 concurrent streams**.
*   Throughput:
    *   Since they are parallel, total throughput is roughly 4x single stream (minus some weight sharing inefficiency, but we modeled that).
    *   Actually, the "Steps/sec" calculated IS the total token generation rate for the batch.
    *   Total Decode Rate = 93 tokens/sec.
    *   Task Decode Time (for 4 parallel tasks) = $640 / 93 \approx 6.9$ sec.
    *   Prefill: 4 tasks in parallel?
        *   Compute bound. A100 does 1 task in 2s. MI300X compute is comparable or better? MI300X FP16 ~1300 TFLOPS? (Roughly 4x A100?).
        *   If 4x compute, 4 tasks take ~2s (same wall time as 1 task on A100 if fully parallel).
        *   Let's assume Prefill Wall Time $\approx 2-3$ sec.
    *   Total Task Cycle Time (Wall clock for 4 tasks) $\approx 6.9 + 3 = 10$ sec.
    *   Tasks per cycle = 4.
    *   Throughput = $4 / 10 = 0.4$ tasks/sec.
    *   Daily Tasks = $0.4 \times 86400 \approx 34,560$ tasks/day.

**5. Cost Calculation**

Let's define the costs.
*   **Pass Rate (P)**: Assume 80% (Typical for complex tasks). Attempts = 1.25.
*   **Electricity**: $0.15/kWh [E]$.
*   **Depreciation**: 3 years (1095 days). Residual 0.

**Scenario 1: A100 80GB**
*   Price: $8,000.
*   Depreciation/Day: $8000 / 1095 \approx \$7.30$.
*   Power: Idle 50W, Active 300W (Avg 200W?). Let's say 250W average during task processing.
*   Energy/Day: $0.25 \text{kW} \times 24 \text{h} = 6 \text{kWh}$.
*   Elec Cost/Day: $6 \times 0.15 = \$0.90$.
*   Total Fixed Cost/Day: $7.30 + 0.90 = \$8.20$.
*   Tasks/Day: 5,000.
*   Cost/Task (before retry): $8.20 / 5000 = \$0.00164$.
*   Cost/Success: $0.00164 / 0.8 = \$0.00205$.

**Scenario 2: M2 Ultra**
*   Price: $4,500.
*   Depreciation/Day: $4500 / 1095 \approx \$4.11$.
*   Power: Avg 100W.
*   Energy/Day: $2.4 \text{kWh}$.
*   Elec Cost/Day: $2.4 \times 0.15 = \$0.36$.
*   Total Fixed Cost/Day: $4.11 + 0.36 = \$4.47$.
*   Tasks/Day: 2,300.
*   Cost/Task: $4.47 / 2300 = \$0.00194$.
*   Cost/Success: $0.00194 / 0.8 = \$0.00242$.
*   *Comparison*: A100 is cheaper per success despite higher capex, because throughput is much higher.

**Scenario 3: MI300X**
*   Price: $12,000.
*   Depreciation/Day: $12000 / 1095 \approx \$10.96$.
*   Power: Avg 400W (High perf).
*   Energy/Day: $9.6 \text{kWh}$.
*   Elec Cost/Day: $9.6 \times 0.15 = \$1.44$.
*   Total Fixed Cost/Day: $10.96 + 1.44 = \$12.40$.
*   Tasks/Day: 34,560.
*   Cost/Task: $12.40 / 34560 = \$0.000359$.
*   Cost/Success: $0.000359 / 0.8 = \$0.000448$.

**Winner**: MI300X is significantly cheaper per task due to massive bandwidth enabling high concurrency.

**Wait, is MI300X "Common"?**
The prompt asks for "Common candidate hardware".
MI300X is enterprise/datacenter. "Local running" might imply a workstation or lab setting.
Is A100 common? Yes, widely available used.
Is H100 common? Getting more common.
What about **Dual A100**?
*   Price: $16,000.
*   Bandwidth: 3.2 TB/s (if PCIe link is fast enough for KV transfer? No, KV is local to GPU if split? No, usually tensor parallel or pipeline parallel. For Decode, KV must be accessible. If TP, KV is sharded. Bandwidth is sum of both HDMs).
*   Effectively 2x A100 performance for Decode (roughly).
*   Throughput ~10,000 tasks/day.
*   Cost/Day: $16000/1095 + 2 \times 0.9 \approx 14.6 + 1.8 = 16.4$.
*   Cost/Task: $16.4 / 10000 = 0.00164$. Same as single A100. Linear scaling.
*   So adding cards scales cost linearly with throughput.
*   Therefore, the *unit economics* depend on the *efficiency* of the single node (or cluster).
*   MI300X has better unit economics than A100 because its bandwidth-per-dollar is higher?
    *   A100: $8000 for 1.6 TB/s. Cost/TBps = $5000.
    *   MI300X: $12000 for 5.3 TB/s. Cost/TBps = $2264.
    *   MI300X is much more efficient in terms of Bandwidth/$.
    *   Since Decode is BW-bound, MI300X wins.

**Are there other options?**
*   **H100**: $25,000 for 3.35 TB/s. Cost/TBps = $7462. Worse than MI300X.
*   **L40S**: 48GB, 864 GB/s. Too slow.
*   **RTX 6000 Ada**: 48GB, 1 TB/s. Similar to A6000.

**Conclusion**: Among "common" high-end local/datacenter GPUs, the one with the highest **Memory Bandwidth per Dollar** will win for this memory-bound workload.
Currently, **AMD MI300X** offers the best Bandwidth/$ ratio for large capacity.
If "Local" strictly excludes Datacenter chips (no liquid cooling, no 8-pin EPS, etc.), then **A100 80GB** (used) is the standard benchmark.
However, MI300X is increasingly appearing in edge/lab environments.
Let's assume the user can acquire a used A100 or a new MI300X.
I will compare A100 vs MI300X.

**Refining the "Pass Rate" and "Retry"**
If pass rate is low, retries increase cost linearly. It doesn't change the ranking of hardware, only the absolute value.

**Final Check on Constraints**
*   27B Model.
*   64K Context.
*   90:9:1 Split.
*   20 tok/s per stream.

My calculation showed:
*   A100: 1 stream max.
*   MI300X: 4 streams max.

This 4x concurrency advantage drives the cost down by ~4x (minus slight efficiency loss).

**Drafting the Response**

## 0. Research Question Definition
QUESTION: Determine the lowest cost-per-successful-task hardware for running a 27B LLM with 64K context and high prefix caching.
SCOPE: Includes hardware acquisition, electricity, depreciation. Excludes labor, software licensing.
OUT: USD per successful task.
PASS: Identify one hardware option with min cost.
FAIL: Unable to satisfy 20 tok/s/stream constraint.
HARD:
1. VRAM >= Model Weights + Max_Concurrent_KV_Cache.
2. Decode_Throughput_Per_Stream >= 20 tokens/s.
OBJ: Minimize (CapEx/Lifetime_Tasks + OpEx_Lifetime) / Successful_Tasks.
COST:
Depreciation: $X/yr [E]
Energy: $Y/yr [E]
Ops: $0 (assumed automated) [E]
Residual: $0 [E]

## 1. Research Path
Nodes:
1. Estimate Model Footprint (Weights + KV).
2. Evaluate Candidate Hardware Specs (BW, VRAM, Price).
3. Calculate Max Concurrency (N) for each HW.
4. Calculate Throughput (Tasks/Day).
5. Calculate Unit Cost.

## 2. Contradictions
CON-1: High Concurrency vs. Per-Stream Speed.
TENSION: Increasing N increases total throughput but decreases per-stream speed due to shared bandwidth.
MECHANISM: KV cache reads scale with N, competing with weight reads.
STATUS: Resolved by finding optimal N.

CON-2: Compute vs. Memory Bound.
TENSION: Prefill is compute bound, Decode is memory bound.
MECHANISM: Different bottlenecks dictate different hardware metrics (FLOPS vs BW).
STATUS: Resolved by noting Decode dominates time (640 tokens vs 5760 tokens, but Decode is serial and BW-limited, Prefill is parallel). Actually, Prefill 5760 tokens takes ~1-2s. Decode 640 tokens takes ~15s (A100). Decode is the bottleneck.

## 3. Calculations & Verifications
(Calculations from draft)

## 4. Conclusion
BOTTLENECK: Memory Bandwidth.
RECOMMENDATION: AMD MI300X (if available/budget allows) or NVIDIA A100 80GB (most common/available). Given "Common", A100 is safer bet, but MI300X is technically superior for this specific workload profile. I will present A100 as the "Standard Local/Datacenter" choice and MI300X as the "Optimal Performance" choice. The prompt asks for "Which one". I should pick the winner.
If I must pick ONE "common" one, A100 is more ubiquitous. But MI300X is becoming common in AI labs.
Let's look at the price again.
A100: $8k.
MI300X: $12k.
If MI300X gives 4x throughput, its cost/task is 1/4th.
$12k / 4 = $3k equivalent. Much cheaper.
So MI300X is the answer.

Wait, is MI300X "Local"?
It requires a server chassis.
If "Local" means "On-premise in a lab", yes.
If "Local" means "Desktop PC", no.
Given "Research Engineer" persona, on-prem server is acceptable.

I will structure the output now.

One detail: **Prefix Caching**.
Does vLLM/SGLang support this efficiently on these cards? Yes.
Does it save Compute? Yes.
Does it save Memory? No, KV is still there.
So my memory calc holds.

Final check on KV size.
If KV is smaller (e.g., due to GQA being more aggressive), N could be higher.
Even if N doubles, the BW/$ logic holds. MI300X has ~3x the BW of A100 for 1.5x the price. It wins.

Table:
| Scheme | Purchase [E] | Dep/Year | Energy/Year | Ops/Year | Residual [E±%] | Net Cost | $/Task |
|---|---|---|---|---|---|---|---|
| A100 80G | $8,000 | $2,666 | $108 | $0 | $0 | $2,774 | $0.00055 (Annual) -> Wait, per task.
| MI300X | $12,000 | $4,000 | $173 | $0 | $0 | $4,173 | ...

Recalc annual costs.
A100: Dep 8000/3 = 2666. Elec 0.25kW * 24 * 365 * 0.15 = 328. Total 2994.
Tasks/Year: 5000 * 365 = 1,825,000.
$/Task: 2994 / 1.825M = $0.00164.

MI300X: Dep 12000/3 = 4000. Elec 0.4kW * 24 * 365 * 0.15 = 525. Total 4525.
Tasks/Year: 34560 * 365 = 12,614,400.
$/Task: 4525 / 12.6M = $0.00036.

MI300X is ~4.5x cheaper.

Confidence: High on physics, Medium on exact pricing (fluctuates).

STOPPED: None.
OPEN: None.

Output generation.
