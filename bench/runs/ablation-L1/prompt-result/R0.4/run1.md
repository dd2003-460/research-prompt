# R0.4 run1

【草稿】

**1. 问题拆解与核心逻辑推导：**
*   **任务画像分析**：
    *   总上下文长度：64K tokens。
    *   比例分布：Cache (90%) : New Input (9%) : Output (1%)。
    *   具体数量：
        *   Cache Hit: $64,000 \times 0.9 = 57,600$ tokens。
        *   New Input (Prefill): $64,000 \times 0.09 = 5,760$ tokens。
        *   Output (Decode): $64,000 \times 0.01 = 640$ tokens。
    *   **关键洞察**：这是一个典型的“长前缀、短输出”场景。计算量主要集中在 Prefill（新输入的处理）和 Decode（自回归生成）。由于 Cache 命中，那 57.6K token 不需要重新进行矩阵乘法（FLOPs），只需要从显存/内存中读取 KV Cache。
    *   **瓶颈判断**：
        *   Prefill 阶段是 Compute-Bound（算力受限），因为要处理 5.7K 个新 token 的注意力机制和前馈网络。
        *   Decode 阶段是 Memory-Bandwidth-Bound（带宽受限），因为每生成一个 token，都需要读取整个模型权重以及当前序列的 KV Cache。
        *   但是，题目给了硬约束：**每个并发流的解码速度不低于 20 token/s**。
        *   如果只跑 1 个并发流，Decode 速度取决于单卡带宽。对于 27B 模型，FP16 下约 54GB，INT8 下约 27GB，INT4 下约 13-15GB。
        *   假设使用 A100/H100 或 RTX 4090/5090 等常见硬件。
        *   我们需要找到一个平衡点：既能满足 20 tok/s 的延迟要求，又能最大化吞吐量（从而降低单任务成本）。
        *   通常，增加并发数会提高 GPU 利用率，但会降低单流速度。因此，“最低成本”往往出现在**刚好满足 20 tok/s 延迟约束的最大并发数**对应的硬件配置上，或者是**单流速度远超 20 tok/s 从而允许极高并发**的高带宽硬件上。

**2. 候选硬件调研（基于 2024-2025 市场现状）：**
*   **消费级旗舰**：NVIDIA RTX 4090 (24GB VRAM), NVIDIA RTX 5090 (32GB VRAM, 未上市但已泄露规格，暂按 4090 为主，5090 作为未来选项或忽略，因目前不可买，或者用预估)。考虑到“本地跑”，4090 是最常见的顶级消费卡。
*   **专业/数据中心级**：NVIDIA A100 80GB (PCIe or SXM), NVIDIA H100 (SXM/PCIe), NVIDIA L40S (48GB).
*   **Apple Silicon**：Mac Studio M2 Ultra / M3 Ultra (Unified Memory). 统一内存带宽很高，且容量大，适合大模型推理。
*   **CPU+GPU 混合**：不太现实，效率太低。

**3. 精度选择策略：**
*   27B 模型在 FP16 下需要 ~54GB 显存 + KV Cache。KV Cache 大小取决于 Batch Size 和 Sequence Length。
*   64K Context 的 KV Cache 很大。
    *   以 LLaMA-3 架构为例，GQA (Grouped Query Attention) 可以减少 KV Cache。假设 GQA 组数为 8。
    *   KV Cache per token $\approx \frac{2 \times \text{num\_layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes}}{\text{group\_size}}$.
    *   粗略估算：27B 模型层数约 48-64 层。假设 48 层，KV Heads 8 (如果是 GQA 8:1)，Head Dim 128。
    *   Per token KV size $\approx 48 \times 8 \times 128 \times 2 \text{ (K and V)} \times 2 \text{ bytes (FP16)} / 8 \text{ (groups? no, heads are already reduced)}$.
    *   更简单的算法：LLaMA 2 70B FP16 KV cache is ~1MB/token for 4K context? No, that's total.
    *   Let's use a standard estimator: For a 30B model, FP16 KV cache is roughly 0.5 - 1 MB per 1K tokens depending on architecture.
    *   If 64K context, KV Cache could be 32GB - 64GB just for one stream in FP16! This is huge.
    *   Therefore, **Quantization is mandatory** to fit into consumer cards or even single data center cards unless using extremely efficient GQA/MQA.
    *   INT8 or INT4 weights will significantly reduce weight memory, but KV Cache often remains FP16 or FP8 for accuracy during attention calculation, though some frameworks allow quantized KV. Let's assume KV Cache is FP16 for safety, or we need a card with massive VRAM.
    *   Wait, if KV Cache is the bottleneck, **Bandwidth** is king for Decode.
    *   To achieve 20 tok/s, we need to read the entire model + KV cache every step.
    *   Time per token $\le 50ms$.
    *   Data to read = Weights + KV Cache.
    *   If Weights = 15GB (INT4), KV Cache = 10GB (for 64K ctx, quantized?), Total = 25GB.
    *   Required Bandwidth = 25GB / 0.05s = 500 GB/s.
    *   RTX 4090 Bandwidth = 1 TB/s. So 1 stream is easy.
    *   Can we run multiple streams?
    *   If we run $N$ streams, bandwidth requirement scales linearly.
    *   Max Streams $N \approx \text{Card BW} / \text{Per-Stream BW Req}$.
    *   Cost per task = Hardware Depreciation / Total Tasks Completed.
    *   Total Tasks = Throughput $\times$ Time.
    *   Throughput increases with N until bandwidth saturates.

**4. 矛盾与张力初步识别：**
*   **矛盾 1**：高并发 vs 低延迟。为了降低成本（分摊硬件折旧），我们希望高吞吐（高并发）。但硬约束限制单流速度 $\ge 20$ tok/s。这限制了最大并发数。
*   **矛盾 2**：显存容量 vs 带宽。大容量显存（如 A100 80G）允许更大的 Batch Size 或更长的 Context without swapping，但单价高、功耗高。小容量高带宽卡（如 4090）便宜，但可能装不下巨大的 KV Cache，导致 OOM 或 Swap（性能暴跌）。
*   **矛盾 3**：精度损失 vs 性能提升。INT4 量化能大幅降低显存占用和带宽需求，但可能导致准确率下降，进而导致“做不对的任务重做”率上升，实际有效成本增加。

**5. 计算框架搭建：**
*   **Cost Model**:
    *   $C_{total} = C_{hardware\_annual} + C_{energy\_annual} + C_{ops}$
    *   $C_{task} = C_{total} / (\text{Throughput}_{eff})$
    *   $\text{Throughput}_{eff} = \text{Base Throughput} \times (1 - P_{retry})$
    *   $P_{retry}$ depends on Precision.

*   **Hardware Candidates**:
    1.  **RTX 4090**: ~$1600, 24GB VRAM, 1 TB/s BW, ~450W.
    2.  **A100 80GB PCIe**: ~$8000-$10000 (Used/Cloud equiv), 80GB HBM2e, 2 TB/s BW, ~300W (PCIe) or 400W (SXM). Let's use PCIe for local setup realism.
    3.  **Mac Studio M2 Ultra**: ~$4000 (192GB config), Unified Memory ~800GB/s BW, ~100-150W (System avg).
    4.  **L40S**: ~$8000, 48GB GDDR6, 864 GB/s BW, ~350W.

*   **Feasibility Check (Memory)**:
    *   Model 27B INT4: ~14GB.
    *   KV Cache for 64K tokens:
        *   Assume LLaMA-like arch. Layers=48, KV_Heads=8 (GQA), Head_Dim=128.
        *   Bytes per token = $2 \times 48 \times 8 \times 128 \times 2 \text{ (FP16)} = 196,608$ Bytes $\approx 0.19$ MB.
        *   Wait, 8 KV heads means the K and V tensors have shape [Batch, 8, SeqLen, 128].
        *   Size = $2 \times 8 \times 64000 \times 128 \times 2 \text{ bytes} = 262,144,000$ Bytes $\approx 250$ MB per stream?
        *   Let's re-calculate carefully.
        *   Formula: $Size = 2 \times N_{layers} \times N_{kv\_heads} \times D_{head} \times S_{len} \times B_{type}$.
        *   $N_{layers}=48$, $N_{kv\_heads}=8$, $D_{head}=128$, $S_{len}=64000$, $B_{type}=2$ (FP16).
        *   $Size = 2 \times 48 \times 8 \times 128 \times 64000 \times 2 = 12,582,912,000$ Bytes $\approx 11.7$ GB.
        *   Okay, ~12GB per stream for KV Cache.
        *   Total Memory per Stream = Weights (shared) + KV Cache (unique).
        *   In batch inference, Weights are shared.
        *   VRAM Usage = $Weight\_Size + Batch\_Size \times KV\_Size$.
        *   For 4090 (24GB):
            *   Weights INT4 = 14GB.
            *   Remaining for KV = 10GB.
            *   Max Batch Size = $10GB / 12GB < 1$.
            *   **Problem**: 4090 cannot even run 1 stream of 64K context with INT4 weights and FP16 KV Cache comfortably if overhead exists. It might fit 1 stream tightly, but no room for batching.
            *   If we quantize KV Cache to INT8/FP8? Size halves to ~6GB.
            *   Then Batch Size = $10GB / 6GB \approx 1.6$. Still only 1 concurrent stream effectively.
            *   If we use MQA (Multi-Query Attention)? KV Heads = 1.
            *   Size = $2 \times 48 \times 1 \times 128 \times 64000 \times 2 = 1.57$ GB.
            *   Most modern 27B models (like Yi, Qwen2, etc.) use GQA. Some older ones might use MHA (KV Heads = Num Heads). If MHA, KV Cache is huge. Assuming GQA is standard now.
            *   Let's assume GQA 8:1. 12GB KV is correct.
            *   So 4090 is limited to **Concurrency = 1**.
            *   Decode Speed on 4090 for 1 stream:
                *   Read Weights (14GB) + KV (12GB) = 26GB.
                *   BW = 1 TB/s.
                *   Theoretical Max TPS = $1000 GB/s / 26 GB \approx 38$ tok/s.
                *   Constraint: $\ge 20$ tok/s. Pass.
                *   Actual TPS likely ~30-35 due to overhead.
                *   Throughput = 35 tasks/hour (since 1 stream).
    *   For A100 80GB (2TB/s BW):
        *   Weights INT4 = 14GB.
        *   Remaining for KV = 66GB.
        *   Max Batch Size = $66GB / 12GB = 5.5 \rightarrow 5$ streams.
        *   Decode Speed for 5 streams:
            *   Total Read = Weights (14GB, read once per step? No, in kernel fusion, weights are read once per batch step usually, or broadcast. But in simple sequential execution, it's complex. In vLLM/TGI, PagedAttention helps.
            *   Approximation: Bandwidth is shared.
            *   Total Data per Step for Batch=5: Weights (14GB) + 5*KV (60GB) = 74GB.
            *   Time per step = $74GB / 2000 GB/s = 0.037 s$.
            *   TPS per stream = $1 / 0.037 \approx 27$ tok/s.
            *   Constraint: $\ge 20$ tok/s. Pass.
            *   Total Throughput = $5 \text{ streams} \times 27 \text{ tok/s} \times 3600 \text{ s/h} / 640 \text{ tokens/task} \approx 760$ tasks/hour.
            *   Compare 4090: 35 tasks/hour.
            *   A100 is ~21x faster throughput.
            *   Cost: A100 is ~6x more expensive than 4090 ($8k vs $1.6k).
            *   Energy: A100 300W vs 4090 450W. A100 uses less power for much higher throughput.
            *   Conclusion: A100 seems better for cost-per-task if concurrency > 1 is possible.

    *   What about Mac Studio M2 Ultra (800GB/s BW, 192GB RAM)?
        *   Weights INT4 = 14GB.
        *   Remaining for KV = 178GB.
        *   Max Batch Size = $178GB / 12GB = 14$ streams.
        *   Decode Speed for 14 streams:
            *   Total Data = 14GB (Weights) + 14*12GB (KV) = 14 + 168 = 182GB.
            *   Time per step = $182GB / 800 GB/s = 0.2275 s$.
            *   TPS per stream = $1 / 0.2275 \approx 4.4$ tok/s.
            *   Constraint: $\ge 20$ tok/s. **FAIL**.
            *   Reduce Batch Size to meet 20 tok/s.
            *   Need Time/step $\le 0.05 s$.
            *   Max Data/Step = $0.05 s \times 800 GB/s = 40 GB$.
            *   $14 + 12 \times B \le 40 \Rightarrow 12B \le 26 \Rightarrow B \le 2$.
            *   So Max Batch = 2.
            *   Throughput = $2 \text{ streams} \times 20 \text{ tok/s} \times 3600 / 640 \approx 225$ tasks/hour.
            *   Compare A100 (760 tasks/hr) and 4090 (35 tasks/hr).
            *   Mac is in the middle.
            *   Cost: Mac $4000. A100 $8000. 4090 $1600.
            *   Let's normalize by Task Cost.

**6. Refining the Calculation:**

Let's define "Task" as generating 640 output tokens.
Time per task for 1 stream = $640 / \text{TPS}$.
Throughput (Tasks/Sec) = $\text{Batch Size} / \text{Time per Step} / \text{Tokens per Task}$?
No.
Throughput (Tokens/Sec) = $\text{Batch Size} \times \text{TPS per stream}$.
Wait, if TPS per stream drops as Batch increases, Total Tokens/Sec might stay constant or drop.
Actually, Total Tokens/Sec $\approx \text{Effective Bandwidth} / \text{Bytes per Token per Stream}$.
If Weights are shared, adding streams adds KV load but not Weight load.
So Total Throughput should increase with Batch Size until KV dominates bandwidth.

Let's calculate **Cost per Million Output Tokens** (CPMT).
Then adjust for Retry Rate.

**Candidate 1: RTX 4090**
*   Price: $1,600 [E]
*   Lifespan: 3 years [E]
*   Annual Depreciation: $533.
*   Power: 450W.
*   Annual Energy Cost (24/7, $0.10/kWh): $450 \times 24 \times 365 \times 0.1 / 1000 = \$394$.
*   Total Annual Fixed Cost: $533 + 394 = \$927$.
*   Performance:
    *   Batch Size = 1 (due to VRAM limit).
    *   TPS = 35 [E] (Conservative estimate for 27B INT4 on 4090 with 64K ctx).
    *   Tasks/Hour = $(35 \times 3600) / 640 = 196$ tasks/hour.
    *   Wait, earlier I said 35. Let's stick with 35.
    *   Annual Tasks = $196 \times 24 \times 365 = 1,724,520$ tasks/year.
*   Base Cost per Task = $\$927 / 1,724,520 = \$0.000537$.
*   Retry Factor: Assume INT4 accuracy loss causes 10% retry rate. Effective Tasks = $1.72M \times 0.9 = 1.55M$.
*   Adjusted Cost = $\$927 / 1,552,000 = \$0.000597$.

**Candidate 2: A100 80GB (PCIe)**
*   Price: $8,000 [E] (Used market/Spot cloud equivalent for local purchase assumption). Let's say $9,000 new-ish.
*   Lifespan: 3 years.
*   Annual Depreciation: $3,000.
*   Power: 300W (PCIe).
*   Annual Energy Cost: $300 \times 24 \times 365 \times 0.1 / 1000 = \$263$.
*   Total Annual Fixed Cost: $3,263$.
*   Performance:
    *   Batch Size = 5 (from previous calc, fits in 80GB).
    *   TPS per stream = 27 [E].
    *   Total Throughput (Tasks/Hour) = $(5 \times 27 \times 3600) / 640 = 760$ tasks/hour.
    *   Annual Tasks = $760 \times 24 \times 365 = 6,657,600$ tasks/year.
*   Base Cost per Task = $\$3,263 / 6,657,600 = \$0.000490$.
*   Retry Factor: Assume INT4 accuracy is same as 4090 (10% retry).
*   Adjusted Cost = $\$3,263 / (6.65M \times 0.9) = \$0.000544$.

**Comparison so far:**
*   4090: $0.000597
*   A100: $0.000544
*   A100 is slightly cheaper per task.

**Candidate 3: Mac Studio M2 Ultra (192GB)**
*   Price: $4,000 [E].
*   Lifespan: 5 years (Apple longevity).
*   Annual Depreciation: $800.
*   Power: 150W (System avg under load).
*   Annual Energy Cost: $150 \times 24 \times 365 \times 0.1 / 1000 = \$131$.
*   Total Annual Fixed Cost: $931$.
*   Performance:
    *   Batch Size = 2 (to meet 20 TPS constraint).
    *   TPS per stream = 20 (constrained).
    *   Total Throughput (Tasks/Hour) = $(2 \times 20 \times 3600) / 640 = 225$ tasks/hour.
    *   Annual Tasks = $225 \times 24 \times 365 = 1,971,000$ tasks/year.
*   Base Cost per Task = $\$931 / 1,971,000 = \$0.000472$.
*   Retry Factor:
    *   Apple Silicon often runs FP16 or INT8 more efficiently/stably? Or does it support INT4 well?
    *   If we can run FP16 weights on Mac (54GB fits in 192GB), accuracy is higher.
    *   Let's assume FP16 on Mac.
    *   Weights = 54GB.
    *   KV Cache (FP16) = 12GB/stream.
    *   Available for KV = $192 - 54 = 138GB$.
    *   Max Batch = $138 / 12 = 11$.
    *   Bandwidth Limit:
        *   Data/Step = $54 + 11 \times 12 = 54 + 132 = 186GB$.
        *   BW = 800 GB/s.
        *   Time/Step = $186 / 800 = 0.23 s$.
        *   TPS = $1/0.23 = 4.3$ tok/s. **FAIL** 20 TPS constraint.
    *   Need to reduce Batch to meet 20 TPS.
    *   Target Time/Step = 0.05s. Max Data = 40GB.
    *   $54 + 12B \le 40 \Rightarrow 12B \le -14$. Impossible.
    *   **Conclusion**: You cannot run 27B FP16 on M2 Ultra and meet 20 TPS even with Batch=1 because Weight loading alone (54GB) takes $54/800 = 0.067s > 0.05s$.
    *   So Mac MUST use Quantized Weights (INT4/INT8) to meet latency.
    *   If INT4 Weights (14GB):
        *   $14 + 12B \le 40 \Rightarrow B \le 2$.
        *   So Batch=2 is max.
        *   Accuracy with INT4 on CPU/GPU hybrid? Usually acceptable, maybe 5% retry rate instead of 10%.
    *   Adjusted Cost = $\$931 / (1.97M \times 0.95) = \$0.000497$.

**Revised Comparison:**
*   4090 (INT4, B=1): $0.000597
*   A100 (INT4, B=5): $0.000544
*   Mac (INT4, B=2): $0.000497

It looks like **Mac Studio** might be the winner due to high unified memory allowing decent batching at low power, despite lower raw bandwidth than A100, because the fixed cost is lower than A100 and energy is negligible.

**Wait, check A100 Batch Size again.**
Is Batch=5 realistic for 64K context?
KV Cache = 12GB per stream.
5 Streams = 60GB KV.
Weights = 14GB.
Total = 74GB. Fits in 80GB.
However, fragmentation and OS overhead might make this tight. Maybe Batch=4 is safer.
If Batch=4:
Data/Step = $14 + 4 \times 12 = 62GB$.
Time/Step = $62 / 2000 = 0.031s$.
TPS = 32.
Throughput = $4 \times 32 \times 3600 / 640 = 720$ tasks/hr.
Annual Tasks = $6,292,800$.
Cost = $3263 / (6.29M \times 0.9) = $0.000575$.
This makes A100 worse than Mac.

What if we use **H100**?
Price: $25,000+. Too expensive for "local" unless enterprise. Exclude.

What if we use **2x RTX 4090**?
Price: $3,200.
VRAM: 48GB.
Can we split model? Yes.
Or run 2 instances?
If we run 2 independent instances on 2 cards:
Each card handles 1 stream (due to 24GB limit).
Total Throughput = 2x Single Card.
Fixed Cost = 2x Single Card Depreciation + Energy.
Depreciation: $1,066/yr.
Energy: $788/yr.
Total Fixed: $1,854.
Throughput: $196 \times 2 = 392$ tasks/hr.
Annual Tasks: $3.45M$.
Retry: 10%.
Adjusted Cost: $1854 / (3.45M \times 0.9) = $0.000597.
Same as single card efficiency, just scaled. No gain in cost-per-task due to linearity, except if we can do tensor parallelism to get higher TPB?
Tensor Parallelism on 2x 4090 for 27B:
Communication overhead over PCIe is high. Usually slower than single card for small models. Not recommended.

**Is there a better candidate?**
**NVIDIA L40S**?
Price: $8,000.
VRAM: 48GB.
BW: 864 GB/s.
Power: 350W.
Depreciation: $2,666/yr.
Energy: $306/yr.
Fixed: $2,972.
Performance:
Weights INT4: 14GB.
Remaining KV: 34GB.
Max Batch: $34/12 = 2$.
Data/Step: $14 + 2 \times 12 = 38GB$.
Time/Step: $38 / 864 = 0.044s$.
TPS: 22.5. (Passes 20).
Throughput: $2 \times 22.5 \times 3600 / 640 = 253$ tasks/hr.
Annual Tasks: $2.21M$.
Retry: 10%.
Adjusted Cost: $2972 / (2.21M \times 0.9) = $0.00149.
Much worse than Mac/A100/4090. Low bandwidth is the killer.

**Conclusion Trend:**
Mac Studio M2 Ultra appears to be the most cost-effective for this specific "Long Context, High Latency Tolerance (20tps), Moderate Throughput" workload due to its massive unified memory enabling higher batch sizes within the latency constraint, combined with low power consumption and moderate price.

**Critical Check: Does Mac M2 Ultra actually sustain 20 TPS with INT4 27B?**
Theoretical max TPS for 1 stream (INT4):
Read 14GB Weights + 12GB KV = 26GB.
BW 800GB/s.
Ideal TPS = $800/26 = 30$ TPS.
With Batch 2:
Read 14GB Weights + 24GB KV = 38GB.
Ideal TPS = $800/38 = 21$ TPS.
This is very close to the 20 TPS limit. Real-world overhead (kernel launch, memory copy) might push it below 20.
If it drops to 18 TPS, it fails the hard constraint.
Risk: High.
If it fails, Mac is out.

If Mac fails, what's next?
A100 with Batch=4 was $0.000575.
4090 was $0.000597.
They are close.
But A100 has higher reliability and professional drivers.
However, 4090 is much cheaper upfront.

Let's look at **RTX 5090** (Expected late 2024/2025).
32GB VRAM.
BW: ~1.8 TB/s (Estimate).
Price: $2,000 (Estimate).
If available, it would dominate.
But the prompt says "common candidates". 5090 is not common yet.
I will stick to 4090, A100, Mac.

**Refining the "Retry" Assumption:**
The prompt says "pass rate is not 100%".
If I use FP16, pass rate is higher.
Can any card run FP16 27B?
Only A100 80GB and Mac 192GB.
4090 cannot run FP16 27B (54GB > 24GB).
So 4090 is forced to INT4/INT8.
A100 can run FP16.
Mac can run FP16.

**Scenario A: All INT4**
*   4090: $0.000597
*   A100: $0.000575 (Batch 4)
*   Mac: $0.000497 (Batch 2, Risky)

**Scenario B: A100 & Mac run FP16**
*   **A100 FP16**:
    *   Weights: 54GB.
    *   KV (FP16): 12GB/stream.
    *   Remaining for KV: $80 - 54 = 26GB$.
    *   Max Batch: $26/12 = 2$.
    *   Data/Step: $54 + 2 \times 12 = 78GB$.
    *   BW: 2000 GB/s.
    *   Time/Step: $78/2000 = 0.039s$.
    *   TPS: 25.6. (Passes 20).
    *   Throughput: $2 \times 25.6 \times 3600 / 640 = 288$ tasks/hr.
    *   Annual Tasks: $2.51M$.
    *   Retry: Assume FP16 has 2% retry rate.
    *   Adjusted Cost: $3263 / (2.51M \times 0.98) = $0.00133.
    *   Much worse than INT4 A100 because throughput dropped significantly (Batch 2 vs Batch 4-5).
    *   So **A100 should run INT4** to maximize throughput, accepting the 10% retry rate. The cost savings from higher throughput outweigh the retry penalty.

*   **Mac FP16**:
    *   Weights: 54GB.
    *   KV: 12GB/stream.
    *   Remaining: $192 - 54 = 138GB$.
    *   Max Batch: $138/12 = 11$.
    *   Data/Step: $54 + 11 \times 12 = 186GB$.
    *   BW: 800 GB/s.
    *   Time/Step: $186/800 = 0.23s$.
    *   TPS: 4.3. **FAIL**.
    *   Must reduce Batch.
    *   Need TPS >= 20 -> Time <= 0.05s -> Data <= 40GB.
    *   $54 + 12B \le 40 \rightarrow$ Impossible.
    *   So Mac **cannot** run FP16 27B within latency constraint.
    *   Mac must also run INT4.

**Final Standings (All INT4):**
1.  **Mac Studio M2 Ultra**: Lowest Cost (~$0.0005). Risk: Latency stability.
2.  **A100 80GB**: Middle Cost (~$0.000575). Stable.
3.  **RTX 4090**: Highest Cost (~$0.0006). Cheapest CapEx.

**Wait, did I miss something about "Local"?**
Usually "Local" implies consumer hardware. A100 is data center gear, but people do buy them for home labs. Mac is definitely local. 4090 is definitely local.
If "Local" strictly excludes Data Center GPUs (due to noise/power/support), then it's Mac vs 4090.
In that case, **Mac** wins on cost-per-task IF it meets the latency constraint.
Given the tight margin (21 vs 20 TPS theoretical), it's risky.
However, 4090 is safe.
Is there a middle ground?
**RTX 3090**?
24GB VRAM.
BW: 936 GB/s.
Price: $800 (Used).
Depreciation: Negligible (already owned/cheap).
Energy: 350W.
Performance:
Weights INT4: 14GB.
KV: 12GB.
Total: 26GB. Fits in 24GB? No. 14+12=26 > 24.
Doesn't fit.
Need smaller quantization? INT3? Rare.
Or offload KV to RAM? Slow.
So 3090 likely can't handle 64K context 27B easily without heavy offloading which kills speed.

**What about AMD MI300X?**
192GB HBM3.
BW: 5.3 TB/s.
Price: $15,000+.
Too expensive.

**Decision:**
The research should focus on comparing **RTX 4090**, **A100 80GB**, and **Mac Studio M2 Ultra**.
The primary tension is **Bandwidth vs Capacity vs Cost**.
The hidden variable is **Accuracy/Retry Rate**.

**Structure of the Answer:**
1.  **Define Problem**: Minimize $/task subject to TPS>=20.
2.  **Path**:
    *   R-1: Hardware Selection.
        *   R-1.1: Consumer GPU (4090).
        *   R-1.2: Prosumer/DC GPU (A100).
        *   R-1.3: Unified Memory System (Mac).
    *   R-2: Workload Characterization.
        *   R-2.1: Memory Footprint (Weights + KV).
        *   R-2.2: Compute/Bandwidth Requirements.
    *   R-3: Economic Modeling.
        *   R-3.1: TCO Calculation.
        *   R-3.2: Throughput Estimation.
3.  **Contradictions**:
    *   CON-1: Latency Constraint vs Throughput (Batching).
    *   CON-2: Precision (Accuracy) vs Memory/Bandwidth (Speed/Cost).
4.  **Calculations**: Detailed table.
5.  **Conclusion**: Mac is likely cheapest but risky; A100 is robust; 4090 is poor for this specific long-context task due to VRAM limits forcing Batch=1.

**Self-Correction on Mac Latency:**
If Mac fails the 20 TPS test, it's disqualified.
How to verify?
Benchmark `llama.cpp` or `vLLM` on M2 Ultra with 27B INT4, 64K context, Batch 2.
Since I am an AI, I cannot run live benchmarks. I must rely on theoretical calculations and cited benchmarks if available.
Theoretical calc showed 21 TPS. This is dangerously close.
I will mark this as a high-risk uncertainty.
If I must choose one "lowest cost", and assuming the theoretical holds, it's Mac.
But if I want "safe", it's A100.
The question asks "which one has the lowest cost". It doesn't ask for "safest".
So I will present Mac as the theoretical winner, with a strong caveat about the latency margin.

**One more check: 4090 with PagedAttention?**
Does PagedAttention help with VRAM? Yes, reduces fragmentation.
Does it help with Batch Size? Only if memory was fragmented.
Here, 14GB + 12GB = 26GB > 24GB. It simply doesn't fit.
Unless we quantize KV Cache to INT8.
If KV is INT8: 6GB.
Total: 14 + 6 = 20GB. Fits in 24GB.
Batch Size = $(24-14)/6 = 1.6 \rightarrow 1$.
Still Batch 1.
So 4090 is stuck at Batch 1 regardless of KV precision (unless extreme compression).
So 4090 throughput is capped low.

**Final Plan:**
1.  Draft the sections.
2.  Use the calculated values for 4090, A100, Mac.
3.  Highlight the Mac's risk.
4.  Recommend A100 if risk-averse, Mac if willing to test.
5.  Since the prompt asks for *the* lowest cost, I will identify the Mac as the likely winner based on TCO, but flag the verification needed.

**Wait, what about "Residual Value"?**
Mac holds value well.
A100 depreciates fast.
4090 holds value okay.
This reinforces Mac's advantage.

**Output Generation.**
(Proceeding to generate the structured response).
