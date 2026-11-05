【草稿】

**1. 问题拆解与核心逻辑**
我们要找的是“每完成一个任务的成本最低”。
成本 = (硬件折旧 + 能耗 + 运维) / (有效完成任务数)。
有效完成任务数 = 总吞吐量 × 一次通过率。
因为题目说“目前的通过率不是 100%”，且没有给具体数值，这是一个巨大的变量。但通常 LLM 在固定任务上的重试率是固定的。假设通过率为 $P$ ($0 < P \le 1$)，则每个*成功*任务消耗的资源是 $1/P$ 倍于单个尝试。由于所有候选方案面临相同的任务分布和模型（27B），$P$ 对所有硬件的影响因子是相同的倍数。因此，比较不同硬件的相对优劣时，可以暂时忽略 $P$ 的具体值，只比较单位时间的“原始尝试”成本效率，最后再统一除以 $P$。或者更严谨地说，我们需要计算的是 `Cost_per_successful_task`。

**关键场景特征：9:1:1 比例 & 64K 上下文**
- 缓存命中 90%：这意味着 Prefill 阶段极其重要。虽然只有 10% 是新 token，但这 10% 必须处理完才能开始 Decode。而且 64K 的长上下文意味着 KV Cache 占用显存极大。
- 输出 1 token：Decode 阶段非常短。
- **瓶颈判断**：
    - 如果并发低，Prefill 延迟主导。
    - 如果并发高，显存容量（KV Cache）和带宽（Decode 读取权重/KV）主导。
    - 题目约束：单流解码速度 $\ge 20$ token/s。这其实是一个很高的要求。对于 27B 模型，20 tok/s 的单流速度需要极高的内存带宽或极低的量化精度。
    - *修正思考*：等等，20 tok/s 是“每个并发流”。如果我有 8 个并发流，总吞吐是 160 tok/s？还是说系统总吞吐要支持 N 个用户，每个用户都要跑得快？
    - 通常“本地跑”意味着单机多卡。
    - 27B 模型，FP16 约 54GB。INT8 约 27GB。INT4 约 13.5GB。
    - 64K Context, 27B Model。
    - KV Cache 大小估算：
        - 假设层数 L=32 (典型 27B 架构如 Llama-3-70B 是 80 层，但 27B 可能是 Mixtral 或类似 MoE? 不，27B 通常是 Dense 或小型 MoE。假设 Dense 32 层，Head Dim 128, Heads 32/40?)
        - 让我们用通用公式：KV Size $\approx 2 \times L \times H_{kv} \times D_{head} \times T \times BytesPerParam$.
        - 粗略估计：64K tokens 的 KV Cache 对于 27B 模型，在 FP16 下可能高达 20-30GB。
        - 加上模型权重 54GB (FP16)。总共需要 > 80GB 显存才能跑 FP16 单实例。
        - 如果用 INT4 量化，权重 ~14GB，KV Cache 也可以量化到 INT8/FP8，体积减半。
        - 结论：**显存容量是第一硬约束**。RTX 4090 (24GB) 几乎不可能跑 64K context 的 27B 模型而不爆显存，除非极度激进的量化和分页交换（Swap），但 Swap 会杀死 20 tok/s 的速度。
        - A100 80GB: 刚好够 FP16 或宽松点的 INT8。
        - H100 80GB: 同上，但算力更强。
        - L40S 48GB: 可能不够 FP16 64K，需量化。
        - MI300X 192GB: 显存巨大，轻松装下。

**2. 候选硬件筛选**
基于“本地”、“27B”、“64K Context”、“20 tok/s/stream”：
- **RTX 3090/4090**: 显存太小 (24GB)。27B 模型 + 64K KV Cache 很难塞进 24GB 并保持高速。排除，除非接受极低并发或超慢速度。
- **A100 80GB PCIe/SXM**: 经典选择。显存够。带宽 1.5-2TB/s (HBM2e/3)。
- **H100 SXM/PCIe**: 性能最强，带宽 3.35TB/s (HBM3)。贵。
- **L40S 48GB**: 推理卡，但 48GB 对 64K 27B 很紧张。
- **AMD MI300X**: 192GB 显存，带宽 3.2TB/s。性价比高（相对于 H100）。
- **Mac Studio M2 Ultra**: 统一内存大，但速度慢，生态封闭，不适合高并发生产级“最低成本”计算，通常用于开发而非批量推理。

**主要竞争者**：A100, H100, MI300X, 也许还有 4x RTX 4090 (NVLink 不支持，只能跨机或慢速互联，排除)。
考虑到“本地”可能指服务器机房或高性能工作站。
重点对比：**A100 80GB**, **H100 80GB**, **MI300X 192GB**。
为了控制成本和复杂度，我们主要看单卡或多卡组合。
如果是单卡：
- A100: 能跑，但 64K context 下 KV cache 吃紧，可能需要量化。
- H100: 能跑，速度快。
- MI300X: 显存富余，可全精度运行，速度也不错。

**3. 成本模型构建**
- **购置价**：
    - A100 80GB: ~$10k-$15k (二手更低，新货溢价)。
    - H100 80GB: ~$25k-$30k。
    - MI300X: ~$20k-$25k (估计)。
- **折旧**：按 3-5 年直线折旧。
- **能耗**：
    - A100: 400W。
    - H100: 700W。
    - MI300X: 750W。
- **运维**：数据中心托管费或电费+散热。假设自建机房，电费占比大。
- **残值**：参考附录。A100 保值率高，H100 掉价快。

**4. 性能估算 (Roofline Model)**
- **Prefill (Compute Bound)**:
    - FLOPs $\approx 2 \times N \times T_{in}$.
    - 27B 模型，9% 新输入 = 5760 tokens.
    - 计算量不大，主要是矩阵乘。
    - A100 FP16: 312 TFLOPS.
    - H100 FP16: 989 TFLOPS.
    - MI300X FP16: 1300+ TFLOPS (理论峰值).
- **Decode (Memory Bandwidth Bound)**:
    - 每生成 1 token，需要读取整个模型权重 (如果是 Dense) 或激活部分专家 (MoE)。
    - 假设 Dense 27B: 读取 54GB (FP16) 或 27GB (INT8).
    - 带宽需求：$Bandwidth = WeightSize / TimePerToken$.
    - 目标：20 tok/s per stream.
    - 如果并发 K 路，总带宽需求 = $K \times WeightSize / (1/20) = K \times WeightSize \times 20$.
    - A100 BW: 2.0 TB/s.
    - H100 BW: 3.35 TB/s.
    - MI300X BW: 3.2 TB/s.
    
    *检查可行性*：
    若使用 INT8 (27GB weights):
    - A100: Max Total Throughput (tok/s) $\approx 2000 / 27 \approx 74$ tok/s (总).
      若要求每流 20 tok/s，最大并发 $74/20 \approx 3.7$ 路。即最多 3 路并发满足约束。
    - H100: $3350 / 27 \approx 124$ tok/s. 最大并发 $124/20 \approx 6.2$ 路。即 6 路。
    - MI300X: $3200 / 27 \approx 118$ tok/s. 最大并发 5 路。
    
    *注意*：这只是 Decode 阶段的极限。Prefill 还要算。
    Prefill 时间：
    - Input 5760 tokens.
    - A100: $2 \times 27e9 \times 5760 / 312e12 \approx 1$ second.
    - H100: $\approx 0.3$ seconds.
    
    总任务时间 = Prefill + Decode.
    Decode 1 token 很短，所以 Prefill 占大头？
    不，90% 是缓存命中。缓存命中的 token 不需要重新计算 Attention 的全部 QKV 投影吗？
    **关键点**：Prefix Caching (前缀缓存)。
    如果前缀被缓存，那么这 90% 的 token **不需要** 进行 Prefill 计算！只需要加载 KV Cache 到 GPU 显存（如果之前卸载了）或直接复用。
    所以，实际计算的 Input 只有 9% (5760 tokens) 的新 token？
    不对，Prompt 是 64K。
    “缓存命中的 token : 新输入 token : 输出 token ≈ 90 : 9 : 1”.
    这意味着 64K 中，57.6K 是旧的前缀（已缓存），5.76K 是新的 Prompt 部分（需要 Prefill），0.64K 是 Output（需要 Decode）。
    
    所以：
    1. **Prefill 计算量**：仅针对 5.76K 新 token。
    2. **KV Cache 显存占用**：针对全部 64K token (因为旧的也要保留以便后续轮次或本次完整 Attention)。
       - 显存压力依然很大。
       
    **重新评估并发能力**：
    显存限制：
    - 27B Model Weights (INT8): ~27 GB.
    - KV Cache for 64K context:
      - 假设 32 layers, 32 heads, head_dim 128.
      - KV size per token $\approx 2 \times 32 \times 32 \times 128 \times 1 byte (INT8) = 262 KB$.
      - 64K tokens $\times 262 KB \approx 16 GB$.
      - 总计：27 (Weights) + 16 (KV) + Activation/Buffer (~5-10GB) $\approx 50-60 GB$.
      - A100 80GB: 够用。
      - L40S 48GB: 太挤，可能 OOM 或严重 Swap。排除。
      - MI300X 192GB: 非常宽裕，甚至可以跑 FP16 或更高并发。
      
    **吞吐量计算 (Throughput)**：
    假设系统满负荷运行。
    瓶颈通常在 Decode 的带宽，因为 Prefill 计算量较小（5.76K tokens）。
    
    - **A100 80GB (INT8)**:
      - BW 2.0 TB/s.
      - Weights read per token: 27 GB.
      - Max Decode Rate (Total): $2000 GB/s / 27 GB/token \approx 74$ tokens/s.
      - Constraint: Each stream $\ge 20$ tok/s.
      - Max Concurrent Streams $N$: $N \times 20 \le 74 \Rightarrow N \le 3$.
      - So, max 3 streams.
      - Total Task Throughput: 3 tasks/sec? No, 3 streams running in parallel.
      - Time per task:
        - Prefill: $2 \times 27e9 \times 5760 / 312e12 \approx 1.0 s$.
        - Decode: 640 tokens / 20 tok/s = 32 s. (Wait, output is 1 part of 100 total? No, 9:1:1 ratio. Total 64K? Or just the dynamic part?)
        - "单次上下文总长 64K token". This usually means the window size.
        - If Output is 1 unit and Input is 9 units... does it mean 64K is the *total* sequence length at any point?
        - Let's assume the prompt grows. But for a single task instance:
          - New Input: 5760 tokens.
          - Output: 640 tokens.
          - Decode time: 640 tokens. At 20 tok/s, that's 32 seconds.
          - Prefill time: ~1 sec.
          - Total Latency per stream: ~33 seconds.
          - Since we have 3 concurrent streams, throughput = 3 streams / 33 sec $\approx 0.09$ tasks/sec.
          
    - **H100 80GB (INT8)**:
      - BW 3.35 TB/s.
      - Max Decode Rate: $3350 / 27 \approx 124$ tokens/s.
      - Max Concurrent Streams: $124 / 20 = 6.2 \Rightarrow 6$ streams.
      - Prefill: Faster compute. $2 \times 27e9 \times 5760 / 989e12 \approx 0.3 s$.
      - Decode time: Still limited by bandwidth per stream? No, aggregate bandwidth is shared.
      - With 6 streams, each gets $124/6 \approx 20.6$ tok/s. Meets constraint.
      - Total Latency per stream: 0.3 + 32 = 32.3 sec.
      - Throughput: 6 streams / 32.3 sec $\approx 0.185$ tasks/sec.
      
    - **MI300X 192GB (INT8 or even FP16?)**:
      - Let's stick to INT8 for comparison fairness, or use FP16 since memory allows?
      - If FP16 (54GB weights):
        - BW 3.2 TB/s.
        - Max Decode Rate: $3200 / 54 \approx 59$ tokens/s.
        - Max Concurrent Streams: $59 / 20 = 2.95 \Rightarrow 2$ streams.
        - This is worse than INT8 on A100!
      - So MI300X should also use INT8 to maximize concurrency.
      - MI300X (INT8):
        - BW 3.2 TB/s.
        - Max Decode Rate: $3200 / 27 \approx 118$ tokens/s.
        - Max Concurrent Streams: $118 / 20 = 5.9 \Rightarrow 5$ streams.
        - Prefill: Compute bound. MI300X has high FP16/INT8 compute. Assume similar to H100 or slightly better/worse depending on software maturity. Let's say comparable to H100, maybe 0.3-0.5s.
        - Throughput: 5 streams / 32.5 sec $\approx 0.15$ tasks/sec.
        
    **Comparison so far**:
    - A100: ~0.09 tasks/s. Cost: Low-Med.
    - H100: ~0.185 tasks/s. Cost: High.
    - MI300X: ~0.15 tasks/s. Cost: Med-High.
    
    **Is there a cheaper option?**
    What about **4x RTX 4090**?
    - Memory: 4x 24GB = 96GB. Enough for INT8 model + KV.
    - Interconnect: PCIe. Slow.
    - Parallelism: Tensor Parallelism across 4 GPUs.
    - Bandwidth per GPU: 1 TB/s (GDDR6X).
    - Total Bandwidth: 4 TB/s. BUT, communication overhead is huge.
    - Effective bandwidth might drop to 50-60%. ~2.4 TB/s effective.
    - Similar to A100 but with higher latency due to PCIe sync.
    - Compute: 4x 82 TFLOPS (FP16) = 328 TFLOPS. Comparable to A100.
    - Cost: 4x $1.6k = $6.4k. Much cheaper than A100 ($10k+).
    - Power: 4x 450W = 1800W vs A100 400W. Energy cost is 4.5x higher.
    - Reliability: Consumer cards, no ECC, higher failure rate.
    - Does it meet 20 tok/s?
      - With TP=4, each layer is split. Communication every layer adds latency.
      - Likely struggles to hit stable 20 tok/s per stream under load without significant tuning.
      - However, if it *can* run, the capex is lower.
      - But OpEx (power) is much higher.
      - Depreciation: 4090 drops fast.
      
    Let's refine the **A100 vs H100 vs MI300X** analysis as these are the reliable datacenter options.
    
    **Cost Calculation (Annualized)**:
    Assume 3-year depreciation, straight line to 50% residual for A100, 30% for H100, 40% for MI300X [E].
    Electricity: $0.10/kWh.
    
    1. **A100 80GB**:
       - Price: $12,000 [E].
       - Residual (3yr): $6,000.
       - Depreciation/yr: $(12000-6000)/3 = 2000$.
       - Power: 400W = 0.4 kW.
       - Energy/yr: $0.4 \times 24 \times 365 \times 0.10 = 352.8$ USD.
       - Ops/Maint: ~10% of Capex/yr? Or fixed $500/yr. Let's say $500.
       - Total Annual Cost: $2000 + 353 + 500 = 2853$ USD.
       - Throughput: 0.09 tasks/s = 324 tasks/hr = 2,851,200 tasks/year (if running 24/7).
       - Wait, 0.09 tasks/sec is very low.
         $0.09 \times 3600 \times 24 \times 365 = 2,851,200$?
         $0.09 \times 3600 = 324$ tasks/hour.
         $324 \times 24 = 7776$ tasks/day.
         $7776 \times 365 = 2,838,240$ tasks/year.
       - Cost per successful task:
         Need Pass Rate $P$. Let's assume $P=0.9$ [E] (typical for complex fixed tasks).
         Successful tasks/yr = $2,838,240 \times 0.9 = 2,554,416$.
         Cost/Task = $2853 / 2,554,416 \approx 0.0011$ USD.
         
    2. **H100 80GB**:
       - Price: $28,000 [E].
       - Residual (3yr): $8,400 (30%).
       - Depreciation/yr: $(28000-8400)/3 = 6533$.
       - Power: 700W = 0.7 kW.
       - Energy/yr: $0.7 \times 24 \times 365 \times 0.10 = 617.4$ USD.
       - Ops: $500.
       - Total Annual Cost: $6533 + 617 + 500 = 7650$ USD.
       - Throughput: 0.185 tasks/s.
         $0.185 \times 3600 \times 24 \times 365 = 5,740,000$ tasks/year (raw).
         Successful: $5,740,000 \times 0.9 = 5,166,000$.
         Cost/Task = $7650 / 5,166,000 \approx 0.00148$ USD.
         
    3. **MI300X**:
       - Price: $22,000 [E].
       - Residual (3yr): $8,800 (40%).
       - Depreciation/yr: $(22000-8800)/3 = 4400$.
       - Power: 750W = 0.75 kW.
       - Energy/yr: $0.75 \times 24 \times 365 \times 0.10 = 657$ USD.
       - Ops: $500.
       - Total Annual Cost: $4400 + 657 + 500 = 5557$ USD.
       - Throughput: 0.15 tasks/s.
         $0.15 \times 3600 \times 24 \times 365 = 4,730,000$ tasks/year (raw).
         Successful: $4,730,000 \times 0.9 = 4,257,000$.
         Cost/Task = $5557 / 4,257,000 \approx 0.0013$ USD.
         
    **Result**: A100 seems cheapest per task in this rough model.
    
    **However**, I must check the **4x 4090** scenario more carefully because Capex is so much lower.
    
    **Scenario: 4x RTX 4090**
    - Capex: $6,400.
    - Residual (3yr): $2,000 (30%).
    - Depreciation/yr: $1,466.
    - Power: 1800W (GPU) + 300W (CPU/RAM/PSU) = 2100W = 2.1 kW.
    - Energy/yr: $2.1 \times 24 \times 365 \times 0.10 = 1845$ USD.
    - Ops: Higher risk, let's say $1000/yr for replacement/spare parts.
    - Total Annual Cost: $1466 + 1845 + 1000 = 4311$ USD.
    
    - Performance:
      - Can it do 20 tok/s per stream?
      - With TP=4, latency increases.
      - Estimate throughput: Maybe 0.06 tasks/s (worse than A100 due to comm overhead and lower effective BW).
      - Raw Tasks/Yr: $0.06 \times 3.15M = 189,000$.
      - Successful: $170,000$.
      - Cost/Task: $4311 / 170,000 \approx 0.025$ USD.
      - This is **20x more expensive** per task than A100.
      - Why? Because A100 is a specialized inference card with high HBM bandwidth and low power efficiency penalty compared to GDDR6X consumer cards when doing heavy matrix ops and memory transfers. The "cheap" GPU loses on energy and throughput efficiency.
      
    **What about A100 40GB?**
    - Not enough memory for 64K context 27B model easily. Excluded.
    
    **What about L40S 48GB x 2?**
    - Memory: 96GB.
    - BW: 864 GB/s each. Total 1.7 TB/s.
    - Compute: 362 TFLOPS each. Total 724 TFLOPS.
    - Price: $2 \times 2500 = 5000$.
    - Power: $2 \times 350W = 700W$.
    - Perf: BW is lower than A100 (1.7 vs 2.0 TB/s). Compute is higher.
    - Throughput likely similar or slightly lower than A100.
    - Cost:
      - Dep: $(5000-1500)/3 = 1166$.
      - Energy: $700W \rightarrow 617$.
      - Ops: 500.
      - Total: 2283.
      - If Throughput is 0.08 tasks/s (slightly less than A100 due to BW).
      - Cost/Task: $2283 / (0.08 \times 3.15M \times 0.9) = 2283 / 226,800 \approx 0.01$ USD.
      - Still higher than A100 ($0.0011$).
      - Wait, my A100 throughput estimate was 0.09.
      - Let's re-evaluate A100 throughput.
        - 3 streams.
        - Latency 33s.
        - 3/33 = 0.09. Correct.
      - L40S 2-card:
        - BW 1.7 TB/s.
        - Max decode rate: $1700/27 = 63$ tok/s.
        - Max streams: $63/20 = 3$.
        - Prefill: Faster compute. 0.5s.
        - Latency: 0.5 + 32 = 32.5s.
        - Throughput: 3/32.5 = 0.092 tasks/s.
        - Basically same throughput as A100!
        - But Cost:
          - A100 Annual: 2853.
          - L40S Pair Annual: 2283.
          - **L40S Pair is cheaper!**
    
    **Re-evaluating L40S**:
    - Is 48GB enough?
      - Model INT8: 27GB.
      - KV Cache 64K: 16GB.
      - Total 43GB.
      - Buffer: 5GB.
      - Total 48GB. It fits *exactly*. Risky. Might OOM during peak activation.
      - If it OOMs, performance crashes.
      - Safer to assume it needs quantization to INT4 or smaller batch.
      - If INT4: Weights 13.5GB. KV 8GB. Total 21.5GB. Fits easily in one L40S?
      - Single L40S 48GB:
        - BW 864 GB/s.
        - Max Decode: $864/13.5 = 64$ tok/s.
        - Max Streams: $64/20 = 3$.
        - Throughput: 3 streams / (Prefill + 32s).
        - Prefill on single L40S: Slower. Maybe 1s.
        - Latency 33s.
        - Throughput: 0.09 tasks/s.
        - Cost:
          - Price $2500.
          - Dep: $(2500-750)/3 = 583$.
          - Energy: $350W \rightarrow 308$.
          - Ops: 500.
          - Total: 1391.
          - Cost/Task: $1391 / (0.09 \times 3.15M \times 0.9) = 1391 / 255,000 \approx 0.0054$ USD.
          - Still higher than A100 ($0.0011$).
          - Why? Because A100 does 3 streams too, but costs more annually ($2853 vs $1391$).
          - Wait. $2853 / 255,000 = 0.011$.
          - My previous A100 calculation:
            - Throughput 0.09.
            - Annual Cost 2853.
            - Tasks/Yr: 255,000.
            - Cost/Task: $2853 / 255,000 = 0.0112$.
          - L40S Single:
            - Throughput 0.09.
            - Annual Cost 1391.
            - Tasks/Yr: 255,000.
            - Cost/Task: $1391 / 255,000 = 0.0054$.
            
          **Conclusion**: Single L40S is cheaper than Single A100 if they have the same throughput.
          Do they have the same throughput?
          - A100: 3 streams.
          - L40S: 3 streams.
          - Both ~33s latency.
          - Yes, similar throughput.
          - But L40S is much cheaper to buy and run.
          
          **Is L40S reliable for 64K?**
          - Memory pressure is high.
          - If I use INT4 on L40S, quality might degrade?
          - The prompt says "27B scale language model". Usually implies standard precision unless specified.
          - If I must use INT8/FP16, L40S 48GB is tight for FP16 (54GB weights alone don't fit).
          - So L40S *must* use INT8 or INT4.
          - A100 can use INT8 comfortably.
          - If the task requires high accuracy, INT4 might fail the "Pass Rate" test significantly, increasing retries.
          - If Pass Rate drops from 0.9 to 0.5 due to quantization:
            - L40S Success Tasks: $255,000 \times 0.5 = 127,500$.
            - Cost/Task: $1391 / 127,500 = 0.0109$.
            - Now it's comparable to A100 ($0.0112$).
            
          **Decision Point**:
          The "Pass Rate" is sensitive to precision.
          A100 allows safer quantization (INT8) or even FP16 if optimized (though 64K FP16 is hard on 80GB, it's possible with paged attention).
          Actually, FP16 27B = 54GB. KV 64K = 32GB (FP16). Total 86GB. Doesn't fit in A100 80GB.
          So A100 also MUST use INT8 for 64K context.
          Therefore, both A100 and L40S operate in INT8 regime for 64K context.
          Accuracy should be similar.
          
          So, why is A100 often preferred?
          - Stability.
          - NVLink (if using multiple).
          - Ecosystem.
          
          But for **Single Card** lowest cost:
          **L40S 48GB** looks like the winner on pure numbers, provided it doesn't OOM.
          However, L40S is an "Inference Optimized" card.
          Is there a cheaper alternative?
          **RTX 4090 24GB**: Cannot fit 64K INT8 KV + Weights.
  
  **Let's check the "Local" constraint again.**
  "Local" might imply a workstation, not a server rack.
  L40S is a datacenter card, passive cooling, needs chassis.
  A100 is also datacenter.
  If "Local" means "On-premise", both are valid.
  
  **Final Candidates for Comparison Table:**
  1. **A100 80GB PCIe**: The baseline enterprise solution.
  2. **L40S 48GB**: The cost-effective inference solution.
  3. **H100 80GB**: The high-performance solution (likely overkill/costly).
  4. **4x RTX 4090**: The consumer cluster solution (likely inefficient/high opex).
  
  I will focus the detailed calculation on A100 vs L40S vs H100.
  
  **Refining the "Pass Rate" Factor**:
  The prompt says "Current pass rate is not 100%".
  If I switch hardware, does pass rate change?
  Only if precision changes.
  If all candidates use INT8 (required for 64K on most cards), pass rate is constant.
  So I can normalize by Pass Rate $P$.
  
  **One more check: MI300X**.
  MI300X 192GB.
  Can run FP16?
  Weights 54GB. KV 32GB. Total 86GB. Fits in 192GB easily.
  If running FP16, accuracy is higher than INT8.
  If Pass Rate improves from 0.9 (INT8) to 0.95 (FP16):
  - MI300X Throughput (FP16):
    - BW 3.2 TB/s. Weights 54GB.
    - Max Decode: $3200/54 = 59$ tok/s.
    - Max Streams: $59/20 = 2$.
    - Throughput: 2 streams / (Prefill + 32s).
    - Prefill FP16: Slower than INT8. Say 0.5s.
    - Latency 32.5s.
    - Throughput: $2/32.5 = 0.0615$ tasks/s.
    - Raw Tasks/Yr: $193,000$.
    - Success Tasks: $193,000 \times 0.95 = 183,350$.
    - Annual Cost: $5557$.
    - Cost/Task: $5557 / 183,350 = 0.030$ USD.
    - Much worse than INT8 solutions.
    - So MI300X should also use INT8 to maximize concurrency.
    - If MI300X uses INT8:
      - Throughput 0.15 tasks/s.
      - Success Tasks: $4,257,000$.
      - Cost/Task: $0.0013$.
      - Comparing A100 ($0.011$) and MI300X ($0.0013$).
      - Wait, my A100 calculation was $0.011$.
      - Let's re-read A100 calc.
        - Annual Cost 2853.
        - Tasks 255,000.
        - Cost 0.011.
      - MI300X Calc:
        - Annual Cost 5557.
        - Tasks 4,257,000.
        - Cost 0.0013.
      - MI300X is **8x cheaper** per task than A100??
      - Let's check the math.
      - A100 Throughput: 0.09 tasks/s.
      - MI300X Throughput: 0.15 tasks/s.
      - Ratio: 1.67x faster.
      - Cost Ratio: $5557 / 2853 = 1.95$x more expensive.
      - Cost per task ratio: $1.95 / 1.67 = 1.17$.
      - So MI300X should be *more* expensive per task, not cheaper.
      - Where did I get 4,257,000 for MI300X?
        - $0.15 \times 3.15M = 472,500$.
        - Ah, I multiplied by 10 somewhere?
        - $0.15 \times 3600 = 540$ tasks/hr.
        - $540 \times 24 = 12,960$ tasks/day.
        - $12,960 \times 365 = 4,730,000$.
        - Okay, so 4.7 Million tasks.
      - A100:
        - $0.09 \times 3600 = 324$ tasks/hr.
        - $324 \times 24 = 7776$ tasks/day.
        - $7776 \times 365 = 2,838,240$.
        - 2.8 Million tasks.
      - Ratio of Tasks: $4.73 / 2.83 = 1.67$. Correct.
      - Ratio of Costs: $5557 / 2853 = 1.95$. Correct.
      - Cost per Task:
        - A100: $2853 / 2.83M = 0.0010$.
        - MI300X: $5557 / 4.73M = 0.00117$.
      - So A100 is slightly cheaper per task than MI300X in this INT8 scenario.
      
      **Why did I think MI300X was 0.0013 earlier?**
      - $5557 / 4,257,000 = 0.0013$.
      - $4,257,000$ was $4,730,000 \times 0.9$.
      - A100 Success Tasks: $2,838,240 \times 0.9 = 2,554,416$.
      - A100 Cost/Task: $2853 / 2,554,416 = 0.00111$.
      - MI300X Cost/Task: $5557 / 4,257,000 = 0.00130$.
      
      **Winner: A100 is cheaper than MI300X.**
      
      **Now compare A100 vs L40S**.
      - A100: 0.00111 USD/task.
      - L40S (Single):
        - Throughput 0.09 tasks/s.
        - Same as A100.
        - Annual Cost: 1391 USD.
        - Success Tasks: 2,554,416.
        - Cost/Task: $1391 / 2,554,416 = 0.00054$.
      
      **L40S is 2x cheaper than A100.**
      
      **Is L40S viable?**
      - Risk: OOM.
      - Mitigation: Use INT4? If INT4 works, memory is safe.
      - If INT4 hurts accuracy, Pass Rate drops.
      - If Pass Rate drops to 0.5:
        - L40S Cost/Task: $1391 / (2,554,416 \times 0.5 / 0.9)$? No.
        - Base Success Tasks (at P=0.9) = 2,554,416.
        - If P becomes 0.5, Success Tasks = $Raw \times 0.5 = 1,419,120$.
        - Cost/Task: $1391 / 1,419,120 = 0.00097$.
        - Still cheaper than A100 ($0.00111$).
      
      **Conclusion**: L40S appears to be the lowest cost option, assuming it can stably handle 64K context without catastrophic slowdowns.
      
      **However**, there is a catch.
      "Common candidate hardware".
      L40S is becoming common.
      But what about **A100 40GB**? No, memory insufficient.
      
      **What about 2x L40S?**
      - More memory, more safety.
      - Cost doubles. Throughput roughly doubles (TP=2).
      - Cost per task remains similar.
      
      **Final Recommendation Logic**:
      1. Identify that 64K Context + 27B Model forces INT8 quantization on most cards except high-memory ones.
      2. Identify that Decode is bandwidth-bound.
      3. Calculate Max Concurrency based on Bandwidth / (Weight Size / Target Speed).
      4. Calculate Annualized Cost (Dep + Energy + Ops).
      5. Compare Cost per Successful Task.
      6. Result: L40S 48GB offers the best balance of Capex and Efficiency for this specific workload, beating A100 on cost while matching throughput. H100 is overkill. MI300X is pricier with marginal throughput gain. Consumer GPUs fail on efficiency/memory.

## 0. 研究问题定义

QUESTION: 在本地部署 27B 语言模型处理 64K 上下文、9:1:1 缓存比的任务时，哪种常见硬件方案的单任务完成成本最低？
SCOPE:   包含单卡及小规模多卡（≤4卡）方案；包含 INT8/INT4 量化场景；不包含云端 API 调用；不包含超过 4 卡的集群方案。
OUT:     每完成一个成功任务的平均成本（美元/任务）。
PASS:    找到成本最低的硬件方案，且该方案能满足单流解码 ≥ 20 token/s 的硬约束。
FAIL:    所有候选方案均无法满足 20 token/s 的约束，或无法在预算内稳定运行 64K 上下文。
HARD:    1. 单并发流解码速度 ≥ 20 token/s。
         2. 单次上下文长度 64K token。
         3. 模型规模 27B 参数。
         4. 硬件必须是可独立采购的商用产品。
OBJ:     最小化 `Cost_per_Successful_Task` = `(Annual_Depreciation + Annual_Energy + Annual_Ops) / (Total_Throughput * Pass_Rate)`
COST:    折旧: 2853 USD/年 [E][±10%] (以 A100 为基准估算)
         能耗: 353 USD/年 [F][±5%] (400W @ $0.1/kWh)
         运维: 500 USD/年 [E][±20%] (人工与备件摊销)
         残值: 6000 USD [E][±15%] (A100 3年后估值)

## 1. 研究路径

### R-1.1  显存容量可行性分析
PARENT:  R-0
NOW:     未知哪些硬件能装下 27B 模型 + 64K KV Cache
GOAL:    筛选出显存足够的候选硬件列表
GAP:     缺乏各硬件在 INT8/FP16 下的精确显存占用数据
IN:      model_params, context_len, kv_cache_bytes_per_token, gpu_memory_gb
OUT:     viable_gpu_list
KIND:    signal
RES:     数据
SHARE:   0.3
BOUND:   1.5x (从 2 个候选增加到 3 个候选的可能性)
TEST:    使用 vLLM 或 PyTorch 估算 27B INT8 权重大小及 64K KV Cache 大小，加总后与各 GPU 显存对比
STATUS:  leaf

### R-1.2  解码带宽瓶颈计算
PARENT:  R-0
NOW:     未知各硬件在 Decode 阶段的最大并发流数
GOAL:    计算每个可行硬件在满足 20 tok/s/流 约束下的最大并发数 N_max
GAP:     缺乏各硬件的有效内存带宽数据
IN:      viable_gpu_list, gpu_bandwidth_tb_s, weight_size_gb, target_speed_tok_s
OUT:     max_concurrent_streams
KIND:    signal
RES:     数据
SHARE:   0.3
BOUND:   2.0x (带宽差异导致的并发数差异)
TEST:    公式 `N_max = floor(GPU_Bandwidth / (Weight_Size * Target_Speed))`，代入各硬件参数计算
STATUS:  leaf

### R-1.3  年化运营成本建模
PARENT:  R-0
NOW:     未知各硬件的全生命周期年化成本
GOAL:    计算每个可行硬件的年度总持有成本 (TCO_annual)
GAP:     缺乏准确的二手残值和电价假设
IN:      viable_gpu_list, purchase_price, power_watts, electricity_cost_kwh, residual_value_pct
OUT:     annual_tco
KIND:    signal
RES:     钱
SHARE:   0.4
BOUND:   1.2x (价格波动对成本的影响)
TEST:    查当前市场价，按 3 年直线折旧计算年折旧；按 24/7 运行计算年电费；设定固定运维费
STATUS:  leaf

### R-1.4  任务吞吐量与最终成本合成
PARENT:  R-0
NOW:     未知单任务成本
GOAL:    计算并比较各硬件的 `Cost_per_Successful_Task`
GAP:     需要将并发数转化为系统吞吐量，并引入通过率因子
IN:      max_concurrent_streams, task_latency_s, annual_tco, pass_rate
OUT:     cost_per_task
KIND:    signal
RES:     算力
SHARE:   0.0 (综合节点，不计入父节点 SHARE 总和，但依赖前三者)
BOUND:   ?
TEST:    `Throughput = N_max / Task_Latency`; `Cost_Per_Task = Annual_TCO / (Throughput * 3600 * 24 * 365 * Pass_Rate)`
STATUS:  stopped:S4

*注：R-1.4 为综合计算步骤，其输入完全由 R-1.1, R-1.2, R-1.3 决定，无需额外调研资源，故标记为 stopped:S4 (已有可行解，且这一步不影响可行性，仅为数学汇总)*

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  显存容量与精度的权衡
NODES:   R-1.1 × R-1.2
TENSION:  高显存硬件允许使用高精度(FP16)从而可能提高通过率，但高精度导致权重变大，降低解码带宽效率，减少并发数。
MECHANISM:  FP16 权重是 INT8 的两倍。在带宽受限的 Decode 阶段，更大的权重意味着更少的并发流。虽然 FP16 可能提升 Pass_Rate，但如果并发数下降幅度大于 Pass_Rate 提升幅度，总成本反而上升。
STATUS:  resolved
VERIFICATION: 分别计算 INT8 和 FP16 模式下的 `Cost_per_Task`。假设 FP16 使 Pass_Rate 从 0.9 提升至 0.95，而并发数减半。比较两者结果。
IMPACT:   ±30% (取决于精度对通过率的敏感程度)

CON-2  消费级显卡的低资本支出与高运营成本的冲突
NODES:   R-1.3 × R-1.2
TENSION:  RTX 4090 等消费卡购买便宜，但功耗高、无 ECC、带宽相对较低，导致长期运营成本(OPEX)远高于数据中心卡(CAPEX)。
MECHANISM:  消费卡 GDDR6X 带宽虽高，但能效比(Performance/Watt)远低于 HBM 架构的数据中心卡。且 24GB 显存不足以单独承载 64K 27B 模型，需多卡并联，进一步增加通信开销和故障点。
STATUS:  resolved
VERIFICATION: 对比 4x RTX 4090 与 1x A100/L40S 的 3 年总拥有成本(TCO)。
IMPACT:   +50% (消费卡方案的单任务成本显著高于数据中心卡)

## 3. 计算与验证

CALCULATIONS:

CAL-1  显存占用估算 (INT8)
FORMULA: `Mem_Total = Model_Weights_INT8 + KV_Cache_64K_INT8 + Overhead`
INPUTS:  Model_Weights_INT8 ≈ 27 GB; KV_Cache_64K_INT8 ≈ 16 GB (Est); Overhead ≈ 5 GB
OUTPUT:  Mem_Total ≈ 48 GB
ERROR:   ±10%
SOURCE:  [E]

CAL-2  A100 80GB 最大并发数
FORMULA: `N_max = floor(Bandwidth / (Weight_Size * Target_Speed))`
INPUTS:  Bandwidth = 2.0 TB/s; Weight_Size = 27 GB; Target_Speed = 20 tok/s
OUTPUT:  N_max = floor(2000 / (27 * 20)) = floor(3.7) = 3
ERROR:   ±5%
SOURCE:  [F]

CAL-3  L40S 48GB 最大并发数
FORMULA: `N_max = floor(Bandwidth / (Weight_Size * Target_Speed))`
INPUTS:  Bandwidth = 0.864 TB/s; Weight_Size = 27 GB; Target_Speed = 20 tok/s
OUTPUT:  N_max = floor(864 / (27 * 20)) = floor(1.6) = 1
*修正*: 这里发现 L40S 单卡带宽 864GB/s 只能支撑 1 路 20tok/s?
$864 / 540 = 1.6$. 是的，只能跑 1 路。
如果只能跑 1 路，吞吐量大幅下降。
让我们重新检查 L40S 带宽。L40S 是 864 GB/s。
A100 是 2039 GB/s (HBM2e).
H100 是 3350 GB/s.
MI300X 是 3200 GB/s.

如果 L40S 只能跑 1 路：
Latency ≈ 33s.
Throughput = 1/33 = 0.03 tasks/s.
A100 (3路) = 0.09 tasks/s.
A100 吞吐量是 L40S 的 3 倍。
A100 年成本 2853. L40S 年成本 1391.
Cost/Task A100: $2853 / (0.09 * 3.15M * 0.9) = 0.0011$.
Cost/Task L40S: $1391 / (0.03 * 3.15M * 0.9) = 0.0033$.
**L40S 变得比 A100 贵了！**

之前的错误在于假设 L40S 能跑 3 路。实际上带宽不够。
**谁还能跑？**
**2x L40S**?
BW = 1.7 TB/s.
N_max = floor(1728 / 540) = 3.
Throughput = 3/33 = 0.09 tasks/s.
Same throughput as A100.
Cost 2x L40S:
Price: $5000.
Dep: $1166/yr.
Energy: $617/yr.
Ops: $500/yr.
Total: $2283/yr.
Cost/Task: $2283 / 255,000 = 0.0089$.
Still more expensive than A100 ($0.0011$).

**Who is the cheapest?**
**A100** seems robust.
**What about MI300X?**
BW 3.2 TB/s.
N_max = floor(3200 / 540) = 5.
Throughput = 5/33 = 0.15 tasks/s.
Cost/Year: $5557.
Cost/Task: $5557 / (0.15 * 3.15M * 0.9) = $5557 / 425,000 = 0.013$.
Wait.
$0.15 * 3600 * 24 * 365 = 4,730,000$ raw tasks.
Success = $4,257,000$.
Cost = $5557 / 4,257,000 = 0.0013$.

Comparing:
A100: 0.0011 USD/task.
MI300X: 0.0013 USD/task.
H100: 
BW 3.35 TB/s. N_max = 6.
Throughput = 6/33 = 0.18 tasks/s.
Cost/Year: $7650.
Tasks: $0.18 * 3.15M * 0.9 = 510,000$? No.
$0.18 * 3600 * 24 * 365 = 5,275,200$ raw.
Success: $4,747,680$.
Cost/Task: $7650 / 4,747,680 = 0.0016$.

**Ranking (Cheapest to Most Expensive):**
1. A100 80GB (0.0011)
2. MI300X (0.0013)
3. H100 (0.0016)
4. 2x L40S (0.0089)

**Is there anything cheaper than A100?**
Maybe **A100 40GB**? No, memory insufficient.
Maybe **used A100**?
If used A100 price is $6000.
Dep: $(6000-3000)/3 = 1000$.
Energy: 353.
Ops: 500.
Total: 1853.
Cost/Task: $1853 / 255,000 = 0.0007$.
**Used A100 is the cheapest.**

But the question asks for "common candidate hardware". Used is a market condition, not a hardware type. I will present A100 as the primary recommendation, noting that acquisition price variance significantly impacts the result.

VERIFICATIONS:

VER-1  验证 A100 并发能力
TARGET:  CAL-2 (N_max=3)
METHOD:  在实际 A100 上运行 vLLM，设置 max_num_seqs=3，监控 decode speed。
RESULT:  partial (需实测确认软件栈开销是否影响带宽利用率)
CONFIDENCE: 0.8

VER-2  验证 L40S 带宽瓶颈
TARGET:  L40S 单卡 N_max=1
METHOD:  在 L40S 上运行 benchmark，测量单流 decode speed。
RESULT:  pass (理论计算符合物理极限)
CONFIDENCE: 0.9

## 4. 结论与置信度

BOTTLENECK: R-1.2 (解码带宽)
REASON: SHARE 0.3 (from R-0) * BOUND 2.0x = 0.6. While not the absolute highest product, it is the primary differentiator between candidates that determines the numerator (Throughput) of the cost function. The memory constraint (R-1.1) filters out many candidates, but among survivors, bandwidth dictates concurrency, which directly scales cost efficiency.
STOPPED: R-1.4 (S4)
OPEN: none
CONFIDENCE: 0.85
RECOMMENDATION: 建议使用 NVIDIA A100 80GB (PCIe 或 SXM) 作为首选硬件。若在预算极度受限且能接受较高运维风险，可考虑二手 A100。H100 性能过剩导致单位成本最高。L40S 因带宽不足，需双卡才能匹配 A100 吞吐量，但总成本仍高于 A100。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 | 运维/年 | 残值/保值[E±%] | 全周期净成本 (3年) | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A100 80GB** | $12,000 | $2,000 | $353 | $500 | $6,000 (50%) | $8,559 | **$0.0011** |
| **MI300X** | $22,000 | $4,400 | $657 | $500 | $8,800 (40%) | $16,716 | $0.0013 |
| **H100 80GB** | $28,000 | $6,533 | $617 | $500 | $8,400 (30%) | $24,219 | $0.0016 |
| **2x L40S** | $5,000 | $1,166 | $617 | $500 | $1,500 (30%) | $4,925 | $0.0089 |
| **4x RTX 4090**| $6,400 | $1,466 | $1,845 | $1,000 | $2,000 (30%) | $7,356 | $0.0250 |

*(注：全周期净成本 = 折旧总额 + 能耗总额 + 运维总额。$/验收任务基于 3 年周期、24/7 运行、Pass Rate 0.9 估算)*