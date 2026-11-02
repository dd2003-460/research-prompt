# R0.5 run3

【草稿】

**1. 核心问题拆解与直觉判断**
*   **任务特征**：27B 模型，64K 上下文。关键比例是 Cache:Input:Output = 90:9:1。
    *   这意味着大部分计算量在于处理那 9% 的新输入（Prefill），而不是输出（Decode）。
    *   但是，硬约束是“每个并发流解码速度 >= 20 token/s”。这是一个吞吐量/延迟混合约束。
    *   由于有缓存，Prefill 的计算量大幅减少（只需计算新的 9% + 加载 KV Cache）。
    *   Decode 阶段是内存带宽受限（Memory Bound）还是算力受限（Compute Bound）？对于长上下文 Decode，通常受限于 HBM 带宽读取权重和 KV Cache。
*   **成本构成**：
    *   硬件购置价（一次性）。
    *   电费（持续）。
    *   运维/人力（持续，本地部署可能较低，但需考虑监控等）。
    *   折旧（时间成本）。
    *   残值（回收）。
    *   **关键变量**：单位任务的完成时间（决定能耗和折旧分摊）和成功率（决定实际需要的总任务次数）。
*   **候选硬件调研方向**：
    *   **消费级显卡**：RTX 4090 (24GB), RTX 3090 (24GB)。显存小，跑 27B 模型可能需要量化（INT8/INT4）且上下文长度受限，或者需要多卡。单张 4090 跑 27B INT8 大约需要 27-30GB 显存（含 KV Cache），24GB 不够，必须双卡或极限量化。双卡互联开销大。
    *   **专业工作站卡**：A6000 (48GB), L40S (48GB), A100 (40/80GB)。单卡可容纳 27B FP16/INT8 及较长上下文。L40S 性价比高，A100 贵但稳定。
    *   **服务器级 GPU**：H100/H800。太贵，除非高并发，否则单任务成本极高。
    *   **Apple Silicon**：M1 Ultra / M2 Ultra (64GB/128GB 统一内存)。带宽极高（800GB/s+），适合推理。但生态兼容性、并发能力需考量。
    *   **CPU/专用 ASIC**：排除，速度慢，不满足 20 tok/s 并发要求。

**2. 初步筛选与瓶颈分析**
*   **显存容量约束**：
    *   27B 模型参数：~54GB (FP16) 或 ~27GB (INT8) 或 ~14GB (INT4)。
    *   KV Cache：64K 上下文。假设 Batch Size=1（因为关注单流速度，虽然题目说“每个并发流”，暗示可能有多个流，但为了简化先算单流极限，再考虑并发下的资源争抢）。
    *   KV Cache 大小估算：$2 \times N_{layers} \times N_{heads} \times D_{head} \times SeqLen \times BytesPerElement$。
    *   以 LLaMA-2 70B 为参考，27B 类似结构。Layer=32, Head=32, D_head=128? 不，LLaMA 2 70B 是 80 layers。27B 可能是 Mistral 类或 LLaMA 2 13B 的放大版？假设标准 Transformer。
    *   粗略估算：64K tokens 的 KV Cache 在 FP16 下非常大。
        *   $KV = 2 \times 32 \text{ layers} \times 32 \text{ heads} \times 128 \text{ dim} \times 64 \times 1024 \text{ tokens} \times 2 \text{ bytes}$
        *   $KV \approx 2 \times 32 \times 32 \times 128 \times 65536 \times 2 \approx 3.4 \times 10^9 \text{ bytes} \approx 3.4 \text{ GB}$?
        *   等等，这个公式有点简略。更准确的估算：每层每头每 token 占 $D_{head} \times 2$ (K and V) bytes。
        *   如果是 LLaMA-2 70B 架构缩小版：Layers=40, Heads=40, Dim=128?
        *   让我们用经验法则：FP16 下，1K context 约占 0.5-1GB 显存（取决于模型具体配置）。64K context 可能需要 32-64GB 显存仅用于 KV Cache！
        *   如果 KV Cache 需要 32GB，加上模型权重 27GB (INT8)，总共需要 59GB 显存。
        *   结论：**单张 24GB 显卡（4090/3090）绝对无法运行 64K 上下文的 27B 模型（即使 INT8）**。必须使用大显存卡（48GB+）或 CPU 卸载（极慢，不满足 20tok/s）或多卡聚合。
        *   因此，候选硬件锁定在：**48GB+ 显存的 GPU** 或 **统一内存 > 64GB 的 Apple Mac**。

*   **候选列表修正**：
    1.  **NVIDIA RTX 4090 x2 (NVLink)**: 48GB 总显存。NVLink 带宽高。成本高。
    2.  **NVIDIA RTX A6000**: 48GB GDDR6。单卡。性价比中等。
    3.  **NVIDIA L40S**: 48GB GDDR6。新卡，能效比好。
    4.  **Apple M1 Ultra (64GB/128GB)**: 统一内存。带宽 800GB/s。无需 NVLink，软件栈完善（MLX/CoreML）。
    5.  **NVIDIA A100 80GB**: 企业级，昂贵。

*   **性能估算 (20 tok/s 约束)**：
    *   Decode 阶段主要瓶颈是读取权重和 KV Cache。
    *   所需带宽 $BW_{req} \approx \frac{ModelSize + KVCacheSize}{Latency}$。
    *   Target Latency = 1/20 s = 0.05s/token。
    *   ModelSize (INT8) $\approx 27$ GB。
    *   KVCache (64K, FP16/INT8?) 假设 KV Cache 也量化到 INT8 以节省显存和带宽，则约 1.7GB。若 FP16 则 3.4GB。
    *   总数据量 $\approx 30$ GB。
    *   $BW_{req} \approx 30 \text{ GB} / 0.05 \text{ s} = 600 \text{ GB/s}$。
    *   **带宽检查**：
        *   RTX 4090: 1 TB/s (GDDR6X)。单卡带宽够，但显存不够。双卡 NVLink 有效带宽降低，且跨卡通信开销大。
        *   RTX A6000: 760 GB/s (GDDR6)。带宽勉强够 ($760 > 600$)，但余量很小，且是共享总线，实际吞吐会打折。
        *   L40S: 864 GB/s (GDDR6)。带宽充足。
        *   Apple M1 Ultra: 800 GB/s。带宽充足。
        *   A100: 1.55 TB/s (HBM2e)。带宽非常充足。

*   **成本模型构建**：
    *   定义一个“验收任务”：成功完成一次请求。
    *   通过率 $P$ (例如 0.9)。平均尝试次数 $N = 1/P$。
    *   单次尝试耗时 $T_{task}$。
    *   能耗 $E = P_{power} \times T_{task}$。
    *   折旧 $D = C_{hw} / L_{life}$。
    *   单位成本 $Cost_{unit} = \frac{C_{hw} + E_{total} + O_{maint}}{N_{tasks\_per\_year}}$? 不，应该是 $Cost_{per\_successful\_task} = \frac{\text{Total Annual Cost}}{\text{Annual Successful Tasks}}$。
    *   或者更直接：$Cost_{task} = \text{Energy Cost}_{attempt} + \text{Depreciation Cost}_{attempt} + \text{Ops Cost}_{attempt}$。
    *   由于任务是固定的，我们比较的是**每成功完成任务的边际成本**。
    *   假设年运行小时数 $H_{year} = 8000$ 小时 (24/7 高负载)。
    *   每小时完成任务数 $R = 3600 / T_{task}$。
    *   每年成功任务数 $Y = R \times H_{year} \times P$。
    *   年总成本 $C_{annual} = C_{depr} + C_{energy} + C_{ops}$。
    *   单位成本 $C_u = C_{annual} / Y$。

**3. 详细数据搜集与计算准备**

*   **硬件价格 [E/S]**:
    *   RTX 4090: ~$1600-$2000 USD. 双卡系统主板/CPU/电源等额外 ~$1000. Total ~$4000-$5000.
    *   RTX A6000: ~$4000-$5000 USD. 系统配套 ~$1000. Total ~$5000-$6000.
    *   L40S: ~$8000-$10000 USD. 系统配套 ~$1000. Total ~$9000-$11000.
    *   Apple M1 Ultra 128GB: ~$15000-$20000 USD (Mac Pro). 或 M1 Max 64GB (可能不够 KV Cache 空间?). M1 Ultra 64GB ~$12000.
    *   A100 80GB: ~$15000-$20000 USD. 系统配套 ~$2000. Total ~$17000-$22000.

*   **功耗 [S/E]**:
    *   RTX 4090: 450W per card. System idle/load variance. Avg load power ~500W per card + 100W overhead. Dual card ~1100W.
    *   RTX A6000: 300W per card. System ~400W.
    *   L40S: 350W per card. System ~450W.
    *   Apple M1 Ultra: 100W-200W peak, avg ~150W.
    *   A100: 400W (PCIe) or 550W (SXM). PCIe version cheaper power. System ~600W.

*   **性能 (Token/s)**:
    *   这是最关键的未知数。需要使用基准测试数据或理论估算。
    *   搜索 "LLaMA 2 13B/70B inference latency 4090/A6000/M1 Ultra".
    *   已知事实：
        *   4090 跑 13B FP16 可达 100+ tok/s (短上下文)。
        *   27B 模型比 13B 慢约 2x (计算量翻倍)。
        *   长上下文 (64K) 会增加 KV Cache 读取开销，但主要是 Prefill 变慢。Decode 速度主要看带宽。
        *   如果带宽满足 600GB/s 需求，理论上都能达到 20 tok/s。
        *   实际中，框架优化（vLLM, llama.cpp, MLX）差异巨大。
        *   Apple Silicon 在长上下文推理上表现优异，因为内存带宽高且无 PCIe 瓶颈。
        *   NVIDIA 需要良好的 CUDA 实现。llama.cpp 对 4090 优化较好。

*   **矛盾点预判**：
    *   **显存 vs 速度**：4090 快但显存小，需双卡，增加复杂度和故障率。A6000/L40S 单卡大显存，速度快，但贵。
    *   **苹果 vs N卡**：苹果便宜（相对 A100/L40S），功耗低，但并发扩展性差（单节点限制），且二手市场流动性不如 N卡。
    *   **精度选择**：INT8 vs FP16。FP16 显存占用翻倍。27B FP16 = 54GB。单张 48GB 卡放不下 FP16 27B + 64K KV Cache。必须 INT8 或 INT4。
    *   **INT8 精度损失**：可能导致通过率 $P$ 下降。题目说“目前的通过率不是 100%”，暗示精度敏感。如果量化导致 $P$ 从 0.95 降到 0.80，重试次数增加，成本上升。这是一个重要的非线性因素。

**4. 细化研究路径**

*   **R-1: 硬件候选集确定**
    *   R-1.1: 显存容量门槛筛选 (>= 60GB effective for 27B INT8 + 64K KV).
    *   R-1.2: 带宽门槛筛选 (>= 600 GB/s theoretical).
    *   R-1.3: 生态/驱动成熟度筛选.
*   **R-2: 性能建模**
    *   R-2.1: 理论最大吞吐计算 (基于带宽).
    *   R-2.2: 实际基准数据搜集 (Search S).
    *   R-2.3: 并发影响评估 (Batching efficiency).
*   **R-3: 成本建模**
    *   R-3.1: 购置成本分解.
    *   R-3.2: 能耗成本计算.
    *   R-3.3: 折旧与残值估算.
    *   R-3.4: 通过率敏感度分析 (Impact of Quantization on Pass Rate).

**5. 关键计算草稿**

*   **Scenario A: 2x RTX 4090**
    *   Price: $4500 [E].
    *   Power: 1100W [S].
    *   Bandwidth: Effective ~800GB/s (NVLink bottleneck)? Or just memory bandwidth sum? For decode, reading weights from both cards is tricky. If tensor parallelism, communication overhead. Let's assume effective BW drops by 20%. 1TB/s * 0.8 = 800GB/s.
    *   Speed: Likely meets 20 tok/s easily. Maybe 30-40 tok/s.
    *   Risk: High failure rate due to dual card setup.

*   **Scenario B: 1x RTX A6000**
    *   Price: $5500 [E].
    *   Power: 400W [S].
    *   Bandwidth: 760 GB/s [S].
    *   Speed: Close to limit. Might be 15-25 tok/s depending on optimization. Risky if <20.
    *   Reliability: High (single card).

*   **Scenario C: 1x L40S**
    *   Price: $10000 [E].
    *   Power: 450W [S].
    *   Bandwidth: 864 GB/s [S].
    *   Speed: Comfortable. 30+ tok/s.
    *   Cost: High upfront.

*   **Scenario D: Apple M1 Ultra 128GB**
    *   Price: $15000 [E] (Mac Pro). Or used M1 Ultra board? Hard to buy standalone. Let's stick to whole system.
    *   Power: 150W [E].
    *   Bandwidth: 800 GB/s [S].
    *   Speed: Very good for single stream.
    *   Limitation: Only one instance. Cannot scale concurrency easily without buying more Macs. But question asks for "local run", implying a single node solution for the task stream. If concurrency is low, this is great.

*   **Comparison Metric**: $/Success Task.
    *   Assume 1000 tasks/day.
    *   Time per task: 10 seconds (for 10 output tokens at 20 tok/s? No, 1 output token takes 50ms. 10 tokens = 0.5s. Plus Prefill. Prefill 9% of 64K = 5.7K tokens. Prefill speed depends on Compute. 4090 prefill is fast. Let's say Prefill 0.2s. Total ~0.7s/task attempt).
    *   Wait, "Single context total length 64K". "Cache hit : New Input : Output = 90:9:1".
    *   New Input = 0.09 * 64K = 5760 tokens.
    *   Output = 0.01 * 64K = 640 tokens.
    *   Ah! Output is 640 tokens, not 10.
    *   Decode time = 640 tokens * (1/20 s/token) = 32 seconds.
    *   Prefill time: 5760 tokens. Compute bound.
        *   FLOPs for Prefill $\approx 2 \times P \times L_{new}$.
        *   $P = 27 \times 10^9$. $L_{new} = 5760$.
        *   $FLOPs \approx 3 \times 10^{14}$.
        *   4090 TFLOPS (FP16): ~80 TFLOPS. Time = $3e14 / 80e12 \approx 3.75$ s.
        *   A6000 TFLOPS: ~38 TFLOPS. Time = $3e14 / 38e12 \approx 7.9$ s.
        *   M1 Ultra TFLOPS: ~22.5 TFLOPS (GPU). Time = $3e14 / 22.5e12 \approx 13.3$ s.
    *   Total Attempt Time:
        *   4090: 3.75 (prefill) + 32 (decode) = 35.75 s.
        *   A6000: 7.9 (prefill) + 32 (decode, assuming BW allows 20tok/s) = 39.9 s. (If A6000 is slower in decode, say 15 tok/s, then 42s. Constraint is min 20. So A6000 might fail constraint or operate at edge).
        *   M1 Ultra: 13.3 (prefill) + 32 (decode) = 45.3 s.
    *   Energy per attempt:
        *   4090: 1.1 kW * 35.75 s = 39.3 Wh.
        *   A6000: 0.4 kW * 39.9 s = 16.0 Wh.
        *   M1 Ultra: 0.15 kW * 45.3 s = 6.8 Wh.
    *   Depreciation:
        *   Life 3 years.
        *   4090 Set: $4500 / 3 = $1500/year.
        *   A6000 Set: $5500 / 3 = $1833/year.
        *   M1 Ultra Set: $15000 / 5 (longer life?) = $3000/year.
    *   Tasks per year:
        *   Assume 8000 hours running.
        *   Throughput (attempts/hour) = 3600 / T_attempt.
        *   Success rate P=0.9.
        *   4090: 3600/35.75 = 100 attempts/hr. 100 * 8000 * 0.9 = 720,000 successes/year.
        *   A6000: 3600/39.9 = 90 attempts/hr. 90 * 8000 * 0.9 = 648,000 successes/year.
        *   M1 Ultra: 3600/45.3 = 79 attempts/hr. 79 * 8000 * 0.9 = 568,800 successes/year.
    *   Unit Cost Calculation ($/success):
        *   Electricity cost: $0.10/kWh.
        *   4090:
            *   Energy Cost: 39.3 Wh * 100/hr * 8000 hr = 31,440 kWh = $3144.
            *   Depreciation: $1500.
            *   Ops: $0 (assumed).
            *   Total: $4644.
            *   Per Success: 4644 / 720,000 = $0.0064.
        *   A6000:
            *   Energy Cost: 16.0 Wh * 90/hr * 8000 hr = 11,520 kWh = $1152.
            *   Depreciation: $1833.
            *   Total: $2985.
            *   Per Success: 2985 / 648,000 = $0.0046.
        *   M1 Ultra:
            *   Energy Cost: 6.8 Wh * 79/hr * 8000 hr = 4294 kWh = $429.
            *   Depreciation: $3000.
            *   Total: $3429.
            *   Per Success: 3429 / 568,800 = $0.0060.

    *   **Surprise Result**: A6000 looks cheapest due to lower depreciation than M1 Ultra and lower energy than 4090, despite slightly higher energy than M1. Wait, M1 Ultra depreciation is high because it's expensive. If I use a cheaper M1 Max 64GB?
        *   M1 Max 64GB: ~$4000 (MacBook Pro/Mac Mini).
        *   Bandwidth: 400 GB/s.
        *   Does 400 GB/s support 20 tok/s for 27B?
        *   Required BW = 600 GB/s. 400 < 600. **M1 Max will likely fail the 20 tok/s constraint.**
        *   So M1 Ultra is the only Apple option that fits, but it's expensive.

    *   What about **Used A100 80GB**?
        *   Price: $8000 (used).
        *   Power: 400W.
        *   BW: 1.5 TB/s.
        *   Prefill: Fast.
        *   Decode: Very fast.
        *   Depreciation: $8000/3 = $2666.
        *   Energy: Low.
        *   This might be competitive with A6000.

    *   **Re-evaluating A6000 vs 2x4090**:
        *   2x4090 has higher throughput (faster prefill), so more tasks per year.
        *   But higher energy and similar depreciation.
        *   The "Pass Rate" factor: If 2x4090 uses INT8 and A6000 uses INT8, same pass rate.
        *   If A6000 can run FP16? No, 54GB model doesn't fit in 48GB. Both must be INT8.

    *   **Conclusion Draft**:
        *   Winner likely **RTX A6000** or **Used A100**.
        *   A6000 is safer bet for availability and standard drivers.
        *   Need to verify A6000 decode speed strictly. If it dips below 20 tok/s, it fails hard constraint.
        *   If A6000 fails, next best is L40S (more expensive) or 2x4090 (higher energy/complexity).

**6. Finalizing the Structure for Output**

*   **Section 0**: Define Question, Scope, Out, Pass/Fail, Hard Constraints, Obj, Cost Factors.
*   **Section 1**: Research Path.
    *   R-1: Hardware Screening.
    *   R-2: Performance Validation.
    *   R-3: Cost Modeling.
*   **Section 2**: Contradictions.
    *   CON-1: Bandwidth vs Capacity (4090 issue).
    *   CON-2: Upfront Cost vs Operational Efficiency (Apple vs Nvidia).
*   **Section 3**: Calculations.
    *   Table comparing 4090x2, A6000, L40S, M1 Ultra.
    *   Show Bottleneck shift.
*   **Section 4**: Conclusion.
    *   Recommend A6000 (or Used A100 if budget allows risk).
    *   Confidence level.

Let's refine the "Used A100" option. It's a blackbox in terms of price stability and supply. I will include it as a candidate but mark it with higher uncertainty.

One detail: "Cache hit... 90%". This implies the prompt engineering is already done. We just process the delta.
My calculation of Prefill time assumed processing 9% of 64K. Correct.

Let's double check the A6000 bandwidth constraint.
Model: 27B INT8 = 27GB.
KV Cache: 64K.
If KV Cache is INT8: ~1.7GB.
Total Read per Token: 28.7 GB.
Target: 20 tok/s -> 574 GB/s.
A6000 Peak: 760 GB/s.
Efficiency: Usually 60-70% of peak for inference.
Effective BW: 760 * 0.65 = 494 GB/s.
494 < 574. **A6000 might FAIL the 20 tok/s constraint!**
This is a critical finding.
If A6000 fails, we need something with higher bandwidth or smaller model footprint.
L40S: 864 GB/s * 0.65 = 561 GB/s. Still borderline.
4090: 1008 GB/s * 0.8 (NVLink penalty?) = 800 GB/s. Safe.
M1 Ultra: 800 GB/s * 0.8 = 640 GB/s. Safe.
A100: 1550 GB/s * 0.7 = 1085 GB/s. Very Safe.

So, A6000 is risky. L40S is risky.
Safe options: 2x4090, M1 Ultra, A100.

Comparing Safe Options:
1.  **2x4090**: Cheap hardware, High Energy.
2.  **M1 Ultra**: Expensive hardware, Low Energy.
3.  **A100**: Very Expensive hardware, Medium Energy.

Let's re-calculate costs with the "Fail" scenario for A6000 removed or marked as high risk.
Actually, if A6000 is too slow, it's disqualified.
So candidates are 2x4090, L40S (if optimized well), M1 Ultra, A100.

Between 2x4090 and M1 Ultra:
2x4090: $4500 capex, High OpEx.
M1 Ultra: $15000 capex, Low OpEx.
Break-even point?
Difference in Capex: $10,500.
Difference in Annual Energy: 4090 ($3144) - M1 ($429) = $2715 savings per year.
Payback period: 10500 / 2715 = 3.8 years.
If hardware life is 3 years, M1 Ultra does NOT pay back its premium in energy savings within the life cycle.
Therefore, **2x4090 is cheaper over 3 years**.

What about L40S?
Price $10000. Energy ~$1500/year.
vs 2x4090 ($4500, $3144/year).
Capex diff: $5500.
Opex diff: $1644/year.
Payback: 3.3 years.
Also does not pay back within 3 years.

So, **2x4090** seems the winner for lowest total cost of ownership (TCO) over 3 years, provided it meets the speed constraint.
Does 2x4090 meet speed?
Effective BW ~800 GB/s. Required 574 GB/s. Yes, safe margin.
Prefill is fast.

Final Check: Is there a single card option?
RTX 3090 Ti 24GB? No, not enough VRAM.
RTX 6000 Ada 48GB? Price ~$6800. BW 960 GB/s.
Effective BW 624 GB/s. Safe.
Price $6800 + $1000 sys = $7800.
Energy: 300W * 40s = 12kWh/year? No, 300W * 100hr * 8000h?
Wait, 6000 Ada is faster than A6000, similar to L40S but cheaper?
Actually, RTX 6000 Ada is often priced similarly to A6000 or slightly higher.
Let's stick to the most common "candidate" list: 4090, A6000, L40S, M1 Ultra.

I will structure the answer based on these four.

One nuance: "Local run". 2x4090 requires a motherboard supporting 2 slots and sufficient power. This adds complexity (Nuisance). I will note this.

Draft complete. Proceeding to formal output.
