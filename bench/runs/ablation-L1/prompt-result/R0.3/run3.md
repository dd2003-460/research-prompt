# R0.3 run3

【草稿】

**1. 问题拆解与直觉判断**
*   **核心指标**：每完成一个“成功”任务的成本（$/Success Task）。
*   **成本构成**：硬件折旧 + 能耗 + 运维 - 残值。由于是本地跑，电力单价通常较低，但硬件折旧是大头。
*   **负载特征**：
    *   Context: 64K tokens.
    *   Ratio: Cache Hit (90) : New Input (9) : Output (1).
    *   这意味着每次请求中，只有 9% 的新 token 需要计算 Prefill，1% 的 token 需要 Decode。
    *   但是，Prompt 说“上下文里有一段长期不变的前缀，可以被缓存”。如果前缀被完全缓存，那么新输入 token 可能只是很少的一部分？
    *   仔细看比例：Cache Hit : New Input : Output = 90 : 9 : 1。
    *   Total Tokens per request = 64K.
    *   New Input Tokens = $64000 \times 0.09 = 5760$ tokens.
    *   Output Tokens = $64000 \times 0.01 = 640$ tokens.
    *   Cache Hit Tokens = $64000 \times 0.90 = 57600$ tokens.
    *   等等，通常 "Context Length" 指的是总长度。如果 64K 是单次上下文总长，那么每次推理涉及 64K 个 token 的处理。
    *   关键点：**Prefill** 阶段处理的是“非缓存”的部分吗？还是说 Prefill 处理整个 Prompt？
    *   现代推理引擎（如 vLLM, TensorRT-LLM）支持 Prefix Caching。如果前缀命中，Prefill 只需要计算新增的 9% 部分。
    *   Decode 阶段生成 1% 的输出。
    *   所以，计算量主要在于：
        1.  Prefill 5760 tokens (Compute Bound)。
        2.  Decode 640 tokens (Memory Bandwidth Bound)。
    *   约束：Decode speed >= 20 tokens/s per stream.
    *   这意味着每个流在 Decode 阶段耗时至少 $640 / 20 = 32$ 秒。
    *   Prefill 耗时取决于算力。

*   **候选硬件调研**：
    *   需要跑 27B 模型。
    *   精度选择：FP16/BF16 需要 ~54GB+ VRAM (权重) + KV Cache。
    *   INT8/INT4 量化可以显著降低显存占用和带宽需求。
    *   常见本地高性能硬件：
        1.  **NVIDIA RTX 4090** (24GB GDDR6X, ~1TB/s BW, ~82 TFLOPS FP16, ~164 TOPS INT8). Price ~$1600.
        2.  **NVIDIA A100 40GB/80GB** (HBM2e, ~1.5-2TB/s BW, ~312 TFLOPS FP16). Used price ~$10k-$15k.
        3.  **NVIDIA H100/H800** (HBM3, ~3TB/s BW, ~990 TFLOPS FP16). Price ~$25k+. (可能太贵，且功耗高).
        4.  **AMD MI300X** (192GB HBM3, ~5.3TB/s BW, ~1300 TFLOPS FP16). Price ~$15k? (需确认价格).
        5.  **Mac Studio M2/M3 Ultra** (Unified Memory, ~800GB/s BW). Price ~$4k-$9k. (软件生态差，速度慢，不适合高并发或严格延迟要求，除非单流).
        6.  **Multi-GPU Consumer**: 2x 4090 or 4x 4090.

*   **初步筛选**：
    *   27B 模型在 BF16 下约 54GB。单卡 4090 (24GB) 跑不了 BF16 全量，必须量化到 INT8 (~27GB) 或 INT4 (~14GB)。
    *   如果是 INT4，KV Cache 也需要空间。64K context 的 KV Cache 很大。
    *   假设使用 INT8 量化 (GPTQ/AWQ)。权重 ~27GB。
    *   KV Cache: 27B LLaMA 架构，Layer=32, Head=32, Head_Dim=128.
        *   Per Token KV Size = $2 \text{ (K+V)} \times 32 \text{ layers} \times 32 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes (FP16)} = 512 \text{ KB/token}$.
        *   64K tokens -> $64000 \times 512 \text{ KB} = 32 \text{ GB}$ per stream!
        *   这太大了。单张 4090 (24GB) 连模型都装不下（27GB INT8），更别提 KV Cache。
        *   所以，**单张 4090 无法运行 64K context 的 27B 模型**，即使量化到 INT4。
        *   INT4 权重 ~14GB. KV Cache 如果用 FP16 还是 32GB. 总共 46GB. 4090 不行.
        *   必须用更大的显存卡，或者多卡。
        *   候选调整为：
            1.  **2x RTX 4090** (48GB total). Can fit INT4 weights + limited KV cache. But bandwidth is split across PCIe/NVLink (4090 has no NVLink for tensor parallelism in most consumer drivers, only P2P which is slow). Actually, 4090s can do P2P but it's not recommended for TP due to latency. Better for PP (Pipeline Parallelism).
            2.  **RTX 6000 Ada** (48GB GDDR6, ~960 GB/s). Price ~$6800. Fits INT8 weights + some KV.
            3.  **A100 80GB** (HBM2e, ~2TB/s). Price ~$12k. Fits BF16 or INT8 easily. High bandwidth.
            4.  **MI300X** (192GB HBM3, ~5.3TB/s). Price ~$15k-$18k. Massive bandwidth.
            5.  **L40S** (48GB GDDR6, ~864 GB/s). Price ~$4k-$5k.

*   **瓶颈分析**：
    *   **Decode 阶段**：Memory Bandwidth Bound.
        *   Time per token $\approx \frac{\text{Model Size}}{\text{Bandwidth}}$.
        *   For 27B INT8 (~27GB):
            *   On 4090 (1TB/s effective maybe less due to overhead): $27 / 1000 \approx 27$ ms/token. Speed ~37 tok/s. Meets 20 tok/s constraint.
            *   On A100 (2TB/s): $27 / 2000 \approx 13.5$ ms/token. Speed ~74 tok/s.
            *   On MI300X (5.3TB/s): $27 / 5300 \approx 5$ ms/token. Speed ~200 tok/s.
        *   Wait, the model size for decode is the weight matrix accessed per token.
        *   Constraint: 20 tok/s.
        *   If we run multiple streams, bandwidth is shared.
        *   Let $N$ be number of concurrent streams.
        *   Total Bandwidth Demand $\approx N \times \text{Weight Size}$.
        *   Max $N \approx \frac{\text{Total Bandwidth}}{\text{Weight Size} \times \text{Latency Limit}}$.
        *   Latency Limit = 50ms (for 20 tok/s).
        *   So $N_{max} \approx \frac{BW}{Size \times 0.05}$.
        *   For 4090 (1 TB/s, 27GB INT8): $N_{max} \approx 1000 / (27 \times 0.05) \approx 740$?? No, this formula assumes pure memory bound and perfect overlap. In reality, compute also takes time.
        *   Let's use FLOPS for Prefill and BW for Decode.

    *   **Prefill 阶段**：Compute Bound.
        *   Compute required for 5760 new tokens.
        *   FLOPs $\approx 2 \times P \times L_{new}$ (roughly). $P=27 \times 10^9$.
        *   $2 \times 27e9 \times 5760 \approx 311 \text{ GFLOPs}$.
        *   Time on 4090 (164 TOPS INT8): $311e9 / 164e12 \approx 1.9$ ms. Very fast.
        *   Time on CPU? Too slow.
        *   So Prefill is negligible compared to Decode for this specific ratio (9:1 input:output).
        *   Wait, is Prefill really that small?
        *   Standard attention prefill is $O(L^2)$ for quadratic attention, but FlashAttention makes it $O(L)$.
        *   Even with linear complexity, 5760 tokens is small compared to 64K.
        *   The dominant cost is **Decode** because we generate 640 tokens, each requiring reading the entire model weights (or a portion if sharded) from VRAM.
        *   Therefore, **Bandwidth is the primary bottleneck** for throughput/cost efficiency in Decode-heavy scenarios.

*   **Cost Model**:
    *   Cost per Success Task = $(\text{Hardware Depreciation} + \text{Energy} + \text{Ops}) / (\text{Tasks per Hour} \times \text{Pass Rate})$.
    *   Tasks per Hour depends on Throughput.
    *   Throughput $\approx N_{streams} \times (\text{Output Tokens} / \text{Time per Stream})$.
    *   Time per Stream $\approx T_{prefill} + T_{decode}$.
    *   $T_{decode} = 640 \times t_{token}$.
    *   $t_{token} \propto 1/BW$.
    *   So Throughput $\propto BW$.
    *   Cost per Task $\propto \frac{\text{CapEx}/\text{LifeSpan} + \text{Power}}{\text{Throughput}} \propto \frac{\text{CapEx} + \text{Power}}{BW}$.
    *   We need to minimize $\frac{\text{Annualized Cost}}{BW}$.

*   **Candidate Comparison (Rough)**:
    1.  **RTX 4090** ($1600, 450W, 1TB/s BW, 24GB VRAM).
        *   Problem: Cannot hold 27B INT8 + 64K KV Cache for even 1 stream?
        *   Weight INT8: 27GB. > 24GB. **FAIL**.
        *   Must use INT4. Weight: 14GB. KV Cache (1 stream, 64K, FP16): 32GB. Total 46GB. **FAIL**.
        *   Must quantize KV Cache too? Or use smaller context? Prompt says 64K fixed.
        *   So single 4090 is out unless we use extreme quantization (INT2?) which hurts accuracy/pass rate.
        *   Let's assume we **must** use a card with >48GB VRAM or multi-GPU.

    2.  **2x RTX 4090** ($3200, 900W, 2TB/s aggregate? No, PCIe bottleneck. Effective BW for TP is low. For PP, it's sequential.
        *   Pipeline Parallelism: Split model into 2 parts. Each part runs on one GPU.
        *   Latency adds up.
        *   Cost: $3200. Power: 900W.
        *   This is messy. Let's look at single high-capacity cards first.

    3.  **RTX 6000 Ada** ($6800, 300W, 960 GB/s, 48GB).
        *   Fits INT8 (27GB) + KV Cache (up to ~20GB -> ~40K tokens? No, 32GB needed for 64K).
        *   Tight. Might need INT4 weights (14GB) + FP16 KV (32GB) = 46GB. Fits in 48GB barely.
        *   Or INT8 weights + INT8 KV?
        *   Let's assume it fits.
        *   Metric: Cost/BW.
        *   CapEx/Year (3yr): $2266.
        *   Energy/Year (24/7, $0.1/kWh): $300 \times 24 \times 365 \times 0.1 = \$2628$.
        *   Total Annual: ~$4900.
        *   BW: 960 GB/s.
        *   Ratio: $4900 / 960 \approx 5.1$.

    4.  **A100 80GB** ($12000, 400W, 2039 GB/s).
        *   Fits easily.
        *   CapEx/Year (3yr): $4000.
        *   Energy/Year: $400 \times 24 \times 365 \times 0.1 = \$3504$.
        *   Total Annual: ~$7500.
        *   BW: 2039 GB/s.
        *   Ratio: $7500 / 2039 \approx 3.68$.
        *   Better than 6000 Ada.

    5.  **MI300X** ($16000, 750W, 5300 GB/s).
        *   Fits easily.
        *   CapEx/Year (3yr): $5333.
        *   Energy/Year: $750 \times 24 \times 365 \times 0.1 = \$6570$.
        *   Total Annual: ~$11900.
        *   BW: 5300 GB/s.
        *   Ratio: $11900 / 5300 \approx 2.24$.
        *   Much better.

    6.  **H100 SXM** ($28000, 700W, 3350 GB/s).
        *   CapEx/Year: $9333.
        *   Energy/Year: $6132$.
        *   Total: $15465.
        *   BW: 3350.
        *   Ratio: $15465 / 3350 \approx 4.6$.
        *   Worse than MI300X due to higher price and lower BW density per dollar.

    7.  **Used A100 40GB**?
        *   Can't fit 27B + 64K KV comfortably.

    8.  **Mac Studio M2 Ultra 192GB** ($4000, 100W, 800 GB/s).
        *   Unified memory allows huge context.
        *   CapEx/Year: $1333.
        *   Energy/Year: $87.
        *   Total: $1420.
        *   BW: 800 GB/s.
        *   Ratio: $1420 / 800 \approx 1.77$.
        *   **Wait.** Is this viable?
        *   Software stack: llama.cpp works well on Mac.
        *   Performance: 800 GB/s theoretical. Real world for LLM inference is often 50-70% of theoretical.
        *   If real BW is 400 GB/s.
        *   Ratio becomes $1420 / 400 = 3.55$. Still competitive with A100.
        *   However, the constraint "Decode speed >= 20 token/s".
        *   On Mac, 27B model.
        *   Time per token $\approx 27GB / 400GB/s \approx 67ms$. Speed ~15 tok/s. **FAILS CONSTRAINT**.
        *   If using INT4 on Mac: Weight 14GB.
        *   Time per token $\approx 14GB / 400GB/s \approx 35ms$. Speed ~28 tok/s. **PASSES**.
        *   But does INT4 pass the "task correctness" check? Pass rate might drop.
        *   The prompt says "current pass rate is not 100%". It implies we should maintain current accuracy level or account for drops.
        *   Assuming we must use a precision that maintains acceptable pass rate. Usually FP16/INT8 is safe. INT4 is risky.
        *   If we stick to INT8/FP16 equivalent quality:
            *   Mac M2 Ultra (192GB) running 27B FP16 (54GB).
            *   BW 800 GB/s (real ~400-500).
            *   Tok/s: $54 / 0.45 \approx 120ms$ -> 8 tok/s. **FAILS**.
            *   So Mac is likely out for strict 20 tok/s constraint with high-quality precision.

    9.  **Re-evaluating MI300X vs A100**.
        *   MI300X has significantly higher bandwidth.
        *   Is the price accurate? AMD MI300X is around $15k-$18k.
        *   A100 80GB used is around $10k-$12k.
        *   A100 is older, more available.
        *   Let's check **L40S** again.
        *   L40S: $4500, 350W, 864 GB/s, 48GB.
        *   Fits INT8 (27GB) + KV (limited).
        *   CapEx/Year: $1500.
        *   Energy/Year: $306.
        *   Total: $1806.
        *   BW: 864.
        *   Ratio: $1806 / 864 \approx 2.09$.
        *   This looks very promising! Cheaper than A100, decent BW.
        *   Does it meet 20 tok/s?
        *   INT8 27B. BW 864 GB/s. Real world ~70% -> 600 GB/s.
        *   Tok/s: $27 / 0.6 = 45$ tok/s. **PASSES**.
        *   Can it handle concurrency?
        *   If we run 1 stream: 45 tok/s.
        *   If we run 2 streams: Bandwidth shared. 22.5 tok/s each. **PASSES**.
        *   If we run 3 streams: 15 tok/s each. **FAILS**.
        *   So max 2 streams.
        *   Throughput: 2 streams * 45 tok/s / 640 tokens/task = 0.14 tasks/sec = 504 tasks/hour.
        *   Cost per task: $1806 / (504 \times 365) \approx \$0.0099$.

    10. **Compare L40S vs A100 80GB**.
        *   A100 80GB:
            *   BW 2039 GB/s. Real ~1400 GB/s.
            *   Tok/s (1 stream): $27 / 1.4 = 19$ tok/s. **BARELY FAILS/MEETS**.
            *   Wait, 27GB / 1.4 TB/s = 19.2 ms/token. 52 tok/s.
            *   My previous calc: $27 / 0.0014 s = 19277$ tokens/s? No.
            *   $27 \times 10^9 \text{ bytes} / (1.4 \times 10^{12} \text{ bytes/s}) = 0.019 \text{ s} = 19 \text{ ms}$.
            *   Speed = $1 / 0.019 = 52$ tok/s.
            *   Max streams: $52 / 20 = 2.6$. So 2 streams.
            *   Throughput: 2 streams * 52 tok/s / 640 = 0.1625 tasks/sec = 585 tasks/hour.
            *   Cost: $7500 / (585 \times 365) \approx \$0.0035$.
            *   **A100 is cheaper per task!**
            *   Why? Because although A100 costs more, its bandwidth is much higher, allowing more parallelism or faster decoding.
            *   Wait, L40S had 2 streams at 45 tok/s. A100 has 2 streams at 52 tok/s.
            *   A100 annual cost $7500. L40S annual cost $1806.
            *   A100 throughput 585/hr. L40S throughput 504/hr.
            *   Cost/L40S: $1806 / 184000 \approx 0.0098$.
            *   Cost/A100: $7500 / 213000 \approx 0.0035$.
            *   A100 is ~3x cheaper per task.

    11. **Check MI300X again**.
        *   BW 5300 GB/s. Real ~3700 GB/s.
        *   Tok/s (1 stream): $27 / 3.7 = 7.3$ ms -> 137 tok/s.
        *   Max streams: $137 / 20 = 6.8$. So 6 streams.
        *   Throughput: 6 * 137 / 640 = 1.28 tasks/sec = 4600 tasks/hour.
        *   Cost: $11900 / (4600 \times 365) \approx \$0.0007$.
        *   **MI300X is significantly cheaper per task.**

    12. **Is there a cheaper high-BW option?**
        *   **Used V100**? Low BW. No.
        *   **P40**? No.
        *   **RTX 3090**? 24GB, 936 GB/s. Can't fit model.
        *   **Dual 3090**? 48GB. 1872 GB/s aggregate (PCIe bottleneck).
            *   Price $2000. Power 700W.
            *   CapEx/Year: $666. Energy: $613. Total: $1279.
            *   Effective BW for TP is poor. Let's say 50% of sum -> 900 GB/s.
            *   Similar to L40S but slower compute.
            *   Likely worse than L40S.

    13. **Conclusion Direction**:
        *   MI300X seems the winner among high-end professional GPUs due to massive HBM3 bandwidth and reasonable power consumption relative to performance.
        *   A100 is the runner-up, widely available.
        *   L40S is a budget option but loses to A100/MI300X on $/task due to lower bandwidth efficiency.

    14. **Refining the "Pass Rate" Factor**:
        *   The prompt mentions "pass rate is not 100%".
        *   If we use a lower precision (e.g., INT4 on MI300X), we might get more throughput (smaller model size -> faster decode).
        *   But if pass rate drops from 90% to 70%, the cost per *successful* task increases by factor of $0.9/0.7 \approx 1.28$.
        *   Doubling the speed (via INT4) offsets this.
        *   However, the problem asks to judge based on "common candidate hardware". Usually, this implies standard deployment practices. I will assume standard INT8/FP16 precision that maintains the current pass rate, as changing precision changes the "model" effectively. I will note this sensitivity.

    15. **Drafting the Research Path**:
        *   R-1: Define Workload Metrics.
        *   R-2: Identify Hardware Candidates.
        *   R-3: Estimate Performance (BW/Compute).
        *   R-4: Estimate Costs (CapEx/OpEx).
        *   R-5: Calculate $/Task.

    16. **Checking Constraints**:
        *   "Decode speed >= 20 token/s".
        *   Must ensure selected hardware meets this at the maximum concurrent load we calculate for throughput.

    17. **Specific Calculation for MI300X**:
        *   Model: 27B, INT8. Size ~27 GB.
        *   KV Cache: 64K context.
        *   Need to verify KV Cache size fits in 192GB VRAM with multiple streams.
        *   Per stream KV: 32 GB (FP16).
        *   Weights: 27 GB.
        *   Remaining for KV: $192 - 27 = 165$ GB.
        *   Max Streams by Memory: $165 / 32 \approx 5$ streams.
        *   Max Streams by Bandwidth (20 tok/s limit):
            *   BW 5300 GB/s.
            *   Per token read: 27 GB.
            *   Time per token: $27 / 5300 = 5.09$ ms.
            *   Speed: 196 tok/s.
            *   Max streams: $196 / 20 = 9.8$.
        *   Bottleneck is Memory Capacity (5 streams), not Bandwidth.
        *   So, 5 concurrent streams.
        *   Actual Speed per stream: 196 tok/s (since BW is abundant).
        *   Wait, if 5 streams run concurrently, they share bandwidth.
        *   Total BW demand: $5 \times 27 \text{ GB} / t$.
        *   If we want 20 tok/s per stream:
            *   Time for 1 token = 50 ms.
            *   Data moved per stream per token = 27 GB.
            *   Total data moved per 50ms = $5 \times 27 = 135$ GB.
            *   Required BW = $135 \text{ GB} / 0.05 \text{ s} = 2700$ GB/s.
            *   MI300X has 5300 GB/s. So it can support 5 streams at 20 tok/s easily.
            *   Can it support more?
            *   Memory limits to 5 streams (due to KV cache).
            *   So Throughput = 5 streams.
            *   Time per task = 640 tokens / 196 tok/s? No.
            *   With 5 streams, do they run at full speed?
            *   Yes, because required BW (2700) < Available BW (5300).
            *   So each stream runs at ~196 tok/s.
            *   Time per task = $640 / 196 = 3.26$ seconds.
            *   Throughput = $5 / 3.26 = 1.53$ tasks/sec = 5500 tasks/hour.
            *   Cost: $11900 / (5500 \times 365) \approx \$0.00059$.

    18. **Recalculate A100 80GB**:
        *   VRAM 80 GB.
        *   Weights 27 GB.
        *   KV remaining: 53 GB.
        *   Max Streams by Memory: $53 / 32 \approx 1$ stream. (Only 1 full 64K context fits).
        *   So A100 80GB can only run **1 concurrent stream** for 64K context.
        *   Speed:
            *   BW 2039 GB/s.
            *   Req BW for 1 stream @ 20 tok/s: $27 / 0.05 = 540$ GB/s.
            *   Available 2039. Plenty.
            *   Actual Speed: $27 / 2.039 \text{ TB/s} \approx 13$ ms/token -> 76 tok/s.
            *   Time per task: $640 / 76 = 8.4$ seconds.
            *   Throughput: $1 / 8.4 = 0.119$ tasks/sec = 428 tasks/hour.
            *   Cost: $7500 / (428 \times 365) \approx \$0.0048$.
        *   **Correction**: A100 80GB is memory-limited to 1 stream. MI300X is memory-rich (192GB), allowing 5 streams.
        *   This changes the comparison drastically. MI300X is ~8x cheaper per task.

    19. **What about 2x A100 80GB?**
        *   Cost: $15000. Power: 800W.
        *   CapEx/Year: $5000. Energy: $7000. Total: $12000.
        *   Total VRAM: 160 GB.
        *   Weights replicated? Or sharded?
        *   If sharded (TP=2), each GPU holds 13.5 GB weights.
        *   KV Cache is usually replicated or partitioned.
        *   If KV is partitioned, each GPU handles half the KV.
        *   Effective VRAM for KV per stream: 32 GB total.
        *   Max Streams: $(160 - 27) / 32 \approx 4$ streams.
        *   Bandwidth: Aggregated 4078 GB/s.
        *   Req BW for 4 streams @ 20 tok/s: $4 \times 540 = 2160$ GB/s.
        *   Available 4078. OK.
        *   Speed: Limited by communication overhead between GPUs. Assume 80% efficiency.
        *   Effective BW ~3200 GB/s.
        *   Time per token: $27 / 3.2 = 8.4$ ms -> 119 tok/s.
        *   Time per task: $640 / 119 = 5.38$ s.
        *   Throughput: $4 / 5.38 = 0.74$ tasks/sec = 2670 tasks/hour.
        *   Cost: $12000 / (2670 \times 365) \approx \$0.0012$.
        *   Still worse than MI300X ($0.00059).

    20. **Final Candidate Selection**:
        *   Winner: **AMD MI300X**.
        *   Runner-up: **NVIDIA A100 80GB** (if AMD not available).
        *   Budget: **NVIDIA L40S** (if cost is critical and volume is low, but $/task is higher).

    21. **Structure the Output**:
        *   Section 0: Definition.
        *   Section 1: Path nodes (Workload -> HW Specs -> Perf Est -> Cost Est).
        *   Section 2: Contradictions (e.g., Bandwidth vs Capacity, Precision vs Pass Rate).
        *   Section 3: Calculations (The table comparing MI300X, A100, L40S).
        *   Section 4: Conclusion.

    22. **Double Check Numbers**:
        *   MI300X Price: $15,000 [E].
        *   A100 80GB Price: $11,000 [E] (Used market fluctuates).
        *   L40S Price: $4,500 [E].
        *   Electricity: $0.10/kWh [E].
        *   Lifespan: 3 years [E].
        *   Depreciation: Straight line to 0? Or residual value?
        *   Residual Value:
            *   GPU resale after 3 years is low. Maybe 20%.
            *   Net CapEx = (Price - Residual) / Life.
            *   MI300X: $(15000 - 3000)/3 = 4000$.
            *   A100: $(11000 - 2200)/3 = 2933$.
            *   L40S: $(4500 - 900)/3 = 1200$.
        *   Recalculating Costs with Residual:
            *   MI300X: CapEx 4000 + Energy 6570 = 10570.
            *   A100: CapEx 2933 + Energy 3504 = 6437.
            *   L40S: CapEx 1200 + Energy 306 = 1506.
        *   Throughputs (re-calculated with memory limits):
            *   MI300X (5 streams): 5500 tasks/hr. Cost: $10570 / (5500*365) = 0.00052$.
            *   A100 (1 stream): 428 tasks/hr. Cost: $6437 / (428*365) = 0.00041$.
            *   **WAIT.** A100 is cheaper per task now?
            *   $0.00041 < 0.00052$.
            *   Let's re-check A100 throughput.
            *   A100 80GB. 1 stream.
            *   Speed 76 tok/s.
            *   Time 8.4s.
            *   Throughput 428/hr.
            *   Cost 6437/yr.
            *   $/Task = 6437 / 156220 = 0.000412$.
            *   MI300X:
            *   5 streams.
            *   Speed 196 tok/s.
            *   Time 3.26s.
            *   Throughput 5500/hr.
            *   Cost 10570/yr.
            *   $/Task = 10570 / 2007500 = 0.000526$.
            *   **A100 is cheaper.**
            *   Why? Because MI300X is over-provisioned for the *concurrency* needed to saturate its bandwidth within the memory limit?
            *   No, MI300X runs 5 streams. A100 runs 1.
            *   MI300X costs almost double A100 annually.
            *   MI300X produces 12.8x throughput.
            *   So it should be cheaper.
            *   $10570 / 5500 = 1.92$ $/hour.
            *   $6437 / 428 = 15.04$ $/hour.
            *   Ah, I divided by hours per year incorrectly in the mental math?
            *   $/Task = \text{Annual Cost} / (\text{Tasks/Hour} \times 24 \times 365)$.
            *   A100: $6437 / (428 \times 8760) = 6437 / 3,749,280 = 0.0017$.
            *   MI300X: $10570 / (5500 \times 8760) = 10570 / 48,180,000 = 0.000219$.
            *   **MI300X is 8x cheaper per task.**
            *   My previous manual division was sloppy.
            *   Let's stick to the rigorous calculation in Section 3.

    23. **Final Check on L40S**:
        *   Cost 1506/yr.
        *   Throughput: 2 streams.
        *   Speed 45 tok/s.
        *   Time 14.2s.
        *   Throughput 2 / 14.2 = 0.14 tasks/sec = 504 tasks/hr.
        *   $/Task = 1506 / (504 \times 8760) = 1506 / 4,415,040 = 0.00034$.
        *   **L40S is cheaper than MI300X?**
        *   $0.00034 < 0.000219$? No. $0.00034 > 0.000219$.
        *   MI300X is still cheaper.
        *   Wait, $1.92 $/hr (MI300X) vs $3.4 $/hr (L40S).
        *   MI300X wins.

    24. **One more candidate: RTX 6000 Ada 48GB**.
        *   Cost: CapEx $(6800-1360)/3 = 1813$. Energy 2628. Total 4441.
        *   VRAM 48GB. Weights 27. KV 21GB.
        *   Max Streams: $21/32 < 1$. Only 0.66 streams.
        *   So it can only run **1 stream** if we reduce context? No, context is fixed 64K.
        *   It cannot fit 1 stream of 64K context with INT8 weights + FP16 KV.
        *   It would need INT4 weights (14GB) + FP16 KV (32GB) = 46GB. Fits.
        *   But INT4 reduces pass rate.
        *   If we assume INT4 is acceptable (risky), let's see.
        *   BW 960 GB/s.
        *   Weight 14GB.
        *   Speed: $14 / 0.96 = 14.5$ ms -> 68 tok/s.
        *   Time: $640 / 68 = 9.4$ s.
        *   Throughput: 1 / 9.4 = 0.106 tasks/sec = 383 tasks/hr.
        *   $/Task = 4441 / (383 \times 8760) = 4441 / 3,355,080 = 0.00132$.
        *   Worse than MI300X.

    25. **Conclusion**: MI300X is the best cost-per-successful-task.

## 0. 研究问题定义

QUESTION: 在本地部署 27B 规模语言模型执行固定任务时，哪种常见候选硬件的每成功完成任务成本最低？
SCOPE:   包含主流数据中心级及高端消费级 GPU；不包含 CPU 集群、专用 ASIC（如 TPU/Pod 租赁视为云服务而非本地硬件购置）；考虑 3 年生命周期。
OUT:     美元 / 成功任务 ($/Success Task)
PASS:    找到一种硬件方案，其单位成本低于所有其他候选方案，且满足解码速度 >= 20 token/s 的硬约束。
FAIL:    所有候选方案均无法满足解码速度约束，或无法确定唯一最优解。
HARD:    1. 解码速度 >= 20 token/s (每并发流)
         2. 单次上下文长度 = 64K tokens
         3. 任务通过率保持当前水平（隐含精度不能过度损失，默认采用 INT8/FP16 等效质量）
OBJ:     最小化 每成功完成任务的全周期净成本
COST:    折旧: 见各硬件估算 [E]
         能耗: 基于 24/7 运行，电价 $0.10/kWh [E]
         运维: 忽略不计（本地环境，无额外人力）[E]
         残值: 3 年后预计保值率 20% [E]

## 1. 研究路径

### R-1.1  工作负载特征量化
PARENT:  ROOT
NOW:     已知比例 9:9:1，总长 64K
GOAL:    得到单次任务的 Prefill FLOPs, Decode Memory Access Bytes, KV Cache Size
GAP:     需要将比例转换为绝对 token 数，并计算对应的计算量和内存占用
IN:      context_length, ratio_cache, ratio_input, ratio_output, model_params
OUT:     prefill_tokens, decode_tokens, kv_cache_bytes_per_stream, model_weight_bytes
KIND:    signal
RES:     人的时间
SHARE:   0.10
BOUND:   1.0x (线性映射，无优化空间)
TEST:    手动计算验证公式
STATUS:  leaf

### R-1.2  候选硬件规格采集
PARENT:  ROOT
NOW:     列表为空
GOAL:    获取 MI300X, A100 80GB, L40S, RTX 6000 Ada 的关键参数 (VRAM, BW, FLOPS, Price, Power)
GAP:     需要最新的市场价格和官方技术规格
IN:      candidate_list
OUT:     hw_specs_table
KIND:    signal
RES:     数据
SHARE:   0.20
BOUND:   1.0x
TEST:    对比厂商官网 Datasheet
STATUS:  leaf

### R-1.3  性能与并发能力估算
PARENT:  ROOT
NOW:     未知吞吐量
GOAL:    计算每种硬件在满足 20 tok/s 约束下的最大并发流数 (N_max) 和总吞吐量 (Tasks/Hour)
GAP:     需要结合内存容量限制和带宽限制取最小值
IN:     hw_specs_table, kv_cache_bytes_per_stream, model_weight_bytes
OUT:     n_streams_max, throughput_tasks_per_hour
KIND:    blackbox
RES:     算力
SHARE:   0.30
BOUND:   ? (依赖具体实现的调度效率，估计误差 ±20%)
TEST:    运行 vLLM benchmark 脚本模拟该负载 (若无法实测，则使用理论峰值 x 0.7 效率因子)
STATUS:  split

#### R-1.3.1  内存容量限制检查
PARENT:  R-1.3
NOW:     -
GOAL:    确定由 VRAM 决定的最大并发流数 N_mem
GAP:     (VRAM - Weight_Size) / KV_Cache_Size
IN:      hw_vram, model_weight_bytes, kv_cache_bytes_per_stream
OUT:     n_mem
KIND:    constraint
RES:     -
SHARE:   -
BOUND:   1.0x
TEST:    计算除法
STATUS:  leaf

#### R-1.3.2  带宽与延迟限制检查
PARENT:  R-1.3
NOW:     -
GOAL:    确定由带宽和 20 tok/s 延迟决定的最大并发流数 N_bw
GAP:    Total_BW / (Weight_Size / Latency_Limit)
IN:     hw_bandwidth, model_weight_bytes, latency_limit_s
OUT:     n_bw
KIND:    signal
RES:     -
SHARE:   -
BOUND:   1.0x
TEST:    计算除法
STATUS:  leaf

### R-1.4  成本模型构建
PARENT:  ROOT
NOW:     仅有购置价
GOAL:    计算年化总拥有成本 (TCO_annual)
GAP:     需要折旧、电费、残值计算
IN:     hw_price, hw_power_watts, lifespan_years, electricity_rate, residual_rate
OUT:     tco_annual_usd
KIND:    signal
RES:     钱
SHARE:   0.25
BOUND:   1.0x
TEST:    财务公式计算
STATUS:  leaf

### R-1.5  单位成本综合评估
PARENT:  ROOT
NOW:     -
GOAL:    计算 $/Success Task
GAP:    TCO_annual / (Throughput_Hourly * Hours_Per_Year * Pass_Rate)
IN:     tco_annual_usd, throughput_tasks_per_hour, pass_rate
OUT:     cost_per_success_task
KIND:    signal
RES:     -
SHARE:   0.15
BOUND:   1.0x
TEST:    最终除法运算
STATUS:  leaf

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  带宽密度与内存容量的权衡
NODES:   R-1.3.1 × R-1.3.2
TENSION: 高带宽芯片（如 MI300X）往往伴随高内存容量，但低成本芯片（如 L40S）虽便宜却受限于内存容量导致并发数极低，从而摊薄了低购置价的优点。
MECHANISM: 对于长上下文（64K）场景，KV Cache 占用巨大。小显存芯片只能跑 1-2 个流，无法充分利用其相对较高的带宽/价格比，导致吞吐量断崖式下跌。
STATUS:  resolved
VERIFICATION: 比较 L40S (2 streams) 和 MI300X (5 streams) 的最终 $/Task，确认 MI300X 胜出。
IMPACT:   决定了选型从“最便宜”转向“最高效”，影响 OBJ 约 -80% (成本降低)。

CON-2  精度与通过率的隐性耦合
NODES:   R-1.1 × R-1.5
TENSION: 为了提升吞吐量，可能倾向于使用 INT4 量化以减小模型体积和 KV Cache 压力，但这会降低任务通过率，增加重做成本。
MECHANISM: 提示词指出“通过率不是 100%”，暗示准确率敏感。若量化导致通过率从 90% 降至 70%，有效吞吐量需除以 1.28。
STATUS:  unresolved
VERIFICATION: 在一日内，使用 INT4 权重的模型对测试集进行抽样评估，对比 FP16/INT8 的通过率差异。若差异 > 5%，则排除 INT4 方案。
IMPACT:   若 INT4 不可用，OBJ 会上升 20-30%；若可用，OBJ 可能下降 50%。

## 3. 计算与验证

CALCULATIONS:

CAL-1  单次任务资源需求
FORMULA: 
  prefill_tokens = 64000 * 0.09 = 5760
  decode_tokens = 64000 * 0.01 = 640
  kv_cache_size = 64000 * (2 * 32 * 32 * 128 * 2 bytes) = 32 GB (FP16)
  weight_size_int8 = 27 GB
INPUTS:  R-1.1 OUT
OUTPUT:  5760 tokens prefill, 640 tokens decode, 32 GB KV/stream, 27 GB Weight
ERROR:   ±0% (定义值)
SOURCE:  [F]

CAL-2  硬件规格与年化成本 (假设 3 年寿命, 20% 残值, $0.1/kWh)
FORMULA:
  Annual_CapEx = (Price * (1 - Residual)) / 3
  Annual_Energy = Power_kW * 24 * 365 * 0.1
  TCO_Annual = Annual_CapEx + Annual_Energy
INPUTS:  R-1.2 OUT
OUTPUT:  
  MI300X: CapEx=$4000, Energy=$6570, TCO=$10570
  A100 80G: CapEx=$2933, Energy=$3504, TCO=$6437
  L40S: CapEx=$1200, Energy=$306, TCO=$1506
ERROR:   ±10% (价格波动)
SOURCE:  [E]

CAL-3  最大并发流数与吞吐量
FORMULA:
  N_max = min( floor((VRAM - Weight) / KV), floor(BW / (Weight / 0.05s)) )
  Throughput_Hourly = N_max * (Decode_Tokens / (Weight / BW_effective)) * 3600
  (Assume BW_effective = 0.7 * BW_theoretical for realistic overhead)
INPUTS:  R-1.3 OUT
OUTPUT:
  **MI300X**:
    N_mem = (192 - 27) / 32 = 5
    N_bw = 5300 / (27 / 0.05) = 5300 / 540 = 9.8 -> 9
    N_max = 5
    Speed = 27 / (5300 * 0.7) = 0.0073 s/tok = 137 tok/s
    Time_Task = 640 / 137 = 4.67 s
    Throughput = 5 / 4.67 * 3600 = 3854 tasks/hr
    
  **A100 80GB**:
    N_mem = (80 - 27) / 32 = 1.6 -> 1
    N_bw = 2039 / 540 = 3.7 -> 3
    N_max = 1
    Speed = 27 / (2039 * 0.7) = 0.0189 s/tok = 52 tok/s
    Time_Task = 640 / 52 = 12.3 s
    Throughput = 1 / 12.3 * 3600 = 292 tasks/hr
    
  **L40S**:
    N_mem = (48 - 27) / 32 = 0.65 -> 0 (Cannot fit 1 stream with FP16 KV!)
    *Correction*: L40S 48GB. Weight 27GB. KV 32GB. Total 59GB > 48GB.
    L40S **cannot run** this workload with standard INT8/FP16 precision.
    Must use INT4 Weight (14GB) + INT8 KV? Or smaller context?
    Prompt fixes context.
    If INT4 Weight (14GB) + FP16 KV (32GB) = 46GB. Fits.
    But INT4 risks pass rate.
    Let's assume INT4 is NOT allowed due to CON-2 risk.
    Thus L40S is **INFEASIBLE** for this specific workload profile.
    
    *Re-evaluating RTX 6000 Ada (48GB)*: Same issue. INFEASIBLE.
    
    *Re-evaluating A100 40GB*:
    N_mem = (40 - 27) / 32 = 0.4. INFEASIBLE.

    *So candidates are limited to Large VRAM cards.*
    **H100 80GB**:
    N_mem = (80 - 27) / 32 = 1.
    N_bw = 3350 / 540 = 6.
    N_max = 1.
    Speed = 27 / (3350 * 0.7) = 0.0115 s/tok = 86 tok/s.
    Time_Task = 640 / 86 = 7.44 s.
    Throughput = 1 / 7.44 * 3600 = 483 tasks/hr.
    TCO: CapEx $(28000*0.8)/3 = 7466. Energy $6132. TCO $13598.
    $/Task = 13598 / (483 * 8760) = 0.0032.

    **MI300X** remains the only high-throughput feasible option in this class.

ERROR:   ±15% (Efficiency factor uncertainty)
SOURCE:  [F]

VERIFICATIONS:

VER-1  验证 MI300X 并发可行性
TARGET:  CAL-3 MI300X N_max=5
METHOD:  检查 5 个流的 KV Cache 总和 (160GB) + 权重 (27GB) = 187GB < 192GB VRAM.
RESULT:  pass
CONFIDENCE: 0.95

VER-2  验证 A100 单流瓶颈
TARGET:  CAL-3 A100 N_max=1
METHOD:  检查 2 个流的 KV Cache (64GB) + 权重 (27GB) = 91GB > 80GB VRAM.
RESULT:  pass
CONFIDENCE: 0.95

## 4. 结论与置信度

BOTTLENECK: R-1.3.1 (内存容量限制)
REASON: SHARE(R-1.3)=0.30 * BOUND(R-1.3.1)=1.0x. 虽然 BANDWIDTH 也很重要，但在 64K 长上下文场景下，VRAM 成为硬性截断点，直接限制了并发数 N，进而限制了吞吐量分母。
STOPPED: none
OPEN: CON-2 (INT4 精度与通过率的关系未实测)
CONFIDENCE: 0.85
RECOMMENDATION: 建议采购 AMD MI300X。若供应链受限，次选 NVIDIA H100/A100 集群（需多卡互联），但成本将显著上升。务必在正式部署前进行 INT8 精度下的通过率回归测试。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本 | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AMD MI300X** | $15,000 | $4,000 | $6,570 | $0 | $3,000 (20%) | $10,570 | **$0.00031** |
| NVIDIA A100 80GB | $11,000 | $2,933 | $3,504 | $0 | $2,200 (20%) | $6,437 | $0.00252 |
| NVIDIA H100 80GB | $28,000 | $7,466 | $6,132 | $0 | $5,600 (20%) | $13,598 | $0.00320 |
| NVIDIA L40S | $4,500 | $1,200 | $306 | $0 | $900 (20%) | $1,506 | **Infeasible** (VRAM不足) |
| NVIDIA RTX 6000 Ada | $6,800 | $1,813 | $2,628 | $0 | $1,360 (20%) | $4,441 | **Infeasible** (VRAM不足) |

*(注：$/验收任务 = TCO_Annual / (Throughput_Hourly * 8760)。L40S 和 6000 Ada 因显存不足以容纳 27B 模型 + 64K 上下文的 KV Cache 而被标记为不可行，除非大幅牺牲精度。)*
