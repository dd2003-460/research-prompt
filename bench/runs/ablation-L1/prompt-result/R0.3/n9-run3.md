【草稿】

**1. 核心问题拆解与直觉检查**
*   **任务特征**：27B模型，64K上下文。90%前缀命中（缓存），9%新输入（Prefill），1%输出（Decode）。
*   **瓶颈判断**：
    *   Prefill阶段：处理 $0.09 \times 64K \approx 5760$ tokens。这是计算密集型（Compute-bound），受限于算力（FLOPS）。
    *   Decode阶段：生成 $0.01 \times 64K = 640$ tokens。这是内存带宽密集型（Memory-bandwidth-bound），受限于显存带宽（GB/s）和显存容量。
    *   约束：单流解码 $\ge 20$ token/s。
*   **成本构成**：题目要求“每完成一个任务的成本最低”。通常指全生命周期成本（TCO）或单次运行电费+折旧摊销。考虑到是“本地跑”，硬件购置是一次性投入，但为了比较公平，需折算为“每任务成本” = (硬件总价 / 预计寿命任务数) + (能耗/任务)。或者更直接地，如果假设硬件已购，只看边际成本（电费）？不，题目问“哪一种...成本最低”，隐含了选型决策，所以必须包含硬件折旧或购买价格分摊。
*   **候选硬件调研方向**：
    *   **消费级显卡**：RTX 4090 (24GB), RTX 3090 (24GB), RTX 4080 Super, etc. 优点：便宜，单卡性能尚可。缺点：显存小，27B模型FP16需要~54GB，INT8需要~27GB，INT4需要~13.5GB。单张4090只能跑INT4量化版本。多卡互联成本高且慢。
    *   **专业工作站显卡**：A6000 (48GB), A40 (48GB), L40S (48GB)。优点：显存大，可跑FP16或INT8，无需多卡。缺点：贵，功耗高。
    *   **服务器GPU**：H100/A100。太贵，不适合“本地”低成本场景，除非是二手市场。
    *   **CPU+NPU组合**：如Mac Studio (M2 Ultra/M3 Ultra)。统一内存架构，访存效率高，但算力较低。适合小并发，27B模型在Apple Silicon上速度可能较慢，需验证是否满足20 tok/s。
    *   **二手矿卡/老卡**：风险高，不计入主流推荐，除非作为极端低成本选项提及。

**2. 关键技术参数估算**
*   **模型大小**：27B params。
    *   FP16: ~54 GB
    *   INT8: ~27 GB
    *   INT4 (AWQ/GPTQ): ~13.5 - 15 GB
*   **KV Cache大小**：
    *   27B模型通常是GQA (Grouped Query Attention) 或 MHA。假设是Llama-3 70B结构的缩小版？不，27B可能是类似Qwen-2.5-32B或Llama-3-70B的裁剪？查一下常见27B模型。
    *   实际上，很多开源模型没有精确的27B，接近的是24B (Llama 3.1 8B? No), 32B (Qwen), 34B (Llama 3.1). 这里假设用户指的是一个标准的27B Dense Model。
    *   KV Cache per token: $2 \times \text{Layers} \times \text{Heads} \times \text{HeadDim} \times \text{BytesPerParam}$。
    *   对于64K上下文，KV Cache占用巨大。
        *   若FP16 KV Cache: 64K tokens * 2 bytes * ... 很大。
        *   通常KV Cache会进行量化（FP8/INT8）或使用PagedAttention。
        *   粗略估算：27B模型，64K上下文的KV Cache可能需要 10-20GB (取决于层数和头数)。
        *   总显存需求 = 权重 + KV Cache + 激活值。
        *   INT4权重 (~15GB) + 64K KV Cache (~15GB?) = 30GB+。单张4090 (24GB) 可能放不下64K上下文的INT4模型，除非KV Cache也极度压缩或分页到CPU。
        *   这意味着：**单卡消费级显卡很难流畅跑满64K上下文的27B模型**。可能需要双卡4090 (48GB) 或者一张A6000/L40S (48GB)。

**3. 性能约束分析 (20 token/s)**
*   Decode速度取决于带宽。
*   公式：$\text{Latency}_{token} \approx \frac{\text{Weight Size}}{\text{Bandwidth}}$。
*   目标：$\le 50ms/token$。
*   若INT4权重 15GB：所需带宽 $\ge 15GB / 0.05s = 300 GB/s$。
    *   RTX 4090: ~1 TB/s (理论)，实际有效带宽可能打折，但远大于300。
    *   Mac M2 Ultra: ~800 GB/s。
    *   CPU DDR5: ~100 GB/s (不够，除非多通道聚合很高，但延迟高)。
    *   所以，只要显存够装下权重，20 tok/s很容易达到。瓶颈主要在**能否装下**以及**Prefill速度**（影响整体吞吐，但不影响单流decode下限，除非队列阻塞）。
*   但是，题目说“每个并发流的解码速度不低于 20 token/s”。如果是单流，很容易。如果是多流，带宽被共享。题目问“每完成一个任务的成本”，暗示我们要看单位任务的总耗时或总能耗。
*   由于有90%前缀缓存，Prefill计算量很小（只算9%的新token）。这使得任务非常偏向于Decode和KV Cache管理。

**4. 候选方案筛选**
*   **方案A: 双卡 RTX 4090 (NVLink or PCIe)**
    *   显存: 48GB。可跑INT8 (27GB) + 较大KV Cache。
    *   价格: ~$2000 x 2 = $4000 (含主板等约$5000系统)。
    *   功耗: 2x 450W = 900W GPU + 系统 = ~1000W。
    *   效率: 极高。
*   **方案B: 单卡 NVIDIA A6000 / L40S (48GB)**
    *   显存: 48GB。
    *   价格: L40S ~$2000-$2500 (新品)，A6000二手 ~$1500。
    *   功耗: L40S 350W, A6000 300W。
    *   优势: 单卡部署简单，无互联开销，ECC显存稳定。
*   **方案C: Mac Studio M2 Ultra (128GB RAM)**
    *   显存: 统一内存 128GB (可用给GPU的部分受限，但通常可分配大部分)。
    *   价格: ~$4000-$5000。
    *   功耗: 极低 (<100W 满载)。
    *   性能: 带宽 800GB/s。跑INT4 27B模型，Decode速度可能只有 10-15 tok/s (因为软件栈优化不如CUDA，且CPU-GPU数据搬运)。*风险点：可能达不到20 tok/s硬约束*。需要查证最新LLM on Apple Silicon的性能报告。
*   **方案D: 二手 Tesla P40 / V100 (多卡)**
    *   显存: P40 24GB x 2 = 48GB。
    *   价格: ~$100 x 2 = $200。
    *   功耗: 2x 250W = 500W。
    *   性能: 无Tensor Core (P40) 或旧Gen (V100)，Prefill极慢，Decode带宽低 (P40 300GB/s, V100 900GB/s)。
    *   P40带宽300GB/s，刚好卡在边缘，且无FP16加速，速度慢。V100带宽900GB/s，性能好，但买两张V100也要$1000+，且散热噪音大。

**5. 矛盾与张力**
*   **显存 vs 成本**: 要跑64K上下文27B，至少需要40GB+显存。单卡4090(24GB)不够。必须双卡或大显存专业卡。
*   **功耗 vs 性能**: 消费级卡功耗高，但性能好。专业卡功耗适中，价格高。苹果功耗低，但性能存疑。
*   **精度选择**: 用INT4可以省显存，但精度损失可能导致“做对了没有”的判定失败率上升，从而增加重试次数，间接增加成本。题目提到“通过率不是100%”，这是一个关键变量。**高精度（FP16/INT8）能减少重试，降低有效成本**。因此，必须优先保证精度，其次才是速度。这推向了需要更大显存以支持更高精度的硬件。

**6. 细化计算路径**
*   需要确定27B模型的具体架构参数（层数、头数）来估算KV Cache。
*   假设标准Transformer结构：27B params, ~32 layers, GQA (8 KV heads).
*   KV Cache size per token = $2 \times L \times H_{kv} \times D \times B$.
    *   $L=32, H_{kv}=8, D=128, B=2 (FP16)$.
    *   $2 \times 32 \times 8 \times 128 \times 2 = 131,072$ bytes/token $\approx 128$ KB/token.
    *   64K tokens: $64000 \times 128 KB = 8 GB$.
    *   如果是INT8 KV Cache: 4 GB.
    *   总显存 (INT8 Weights 27GB + INT8 KV 4GB) = 31 GB.
    *   总显存 (FP16 Weights 54GB + FP16 KV 8GB) = 62 GB. -> 需要两张48GB卡或一张96GB卡。
    *   结论：**INT8精度是最佳平衡点**。它能在单张48GB卡（如L40S/A6000）或双卡4090上运行，且精度高于INT4，减少重试。

**7. 最终候选对比**
*   **Option 1: Dual RTX 4090 (PCIe/NVLink)**
    *   Cost: ~$5000 system.
    *   Power: ~1kW.
    *   Speed: Fastest Prefill, Fast Decode.
    *   Risk: NVLink adds cost/complexity. PCIe bandwidth bottleneck for tensor parallelism? Usually fine for inference.
*   **Option 2: Single NVIDIA L40S (48GB)**
    *   Cost: ~$2500 GPU + $500 server = $3000.
    *   Power: ~350W.
    *   Speed: Good. Bandwidth 864 GB/s.
    *   Advantage: Single card, ECC, lower power than dual 4090.
*   **Option 3: Mac Studio M2 Ultra (192GB config to be safe, or 128GB)**
    *   Cost: ~$4000+.
    *   Power: ~100W.
    *   Speed: Uncertain. If it hits <20 tok/s, it fails hard constraint. Recent benchmarks show Llama-70B on M2 Ultra is ~10-15 tok/s. 27B might be faster, maybe 25-30 tok/s? Needs verification. If it passes, its electricity cost is negligible.
*   **Option 4: Used Tesla V100 32GB x 2**
    *   Cost: ~$800 GPUs + $500 system = $1300.
    *   Power: ~500W.
    *   Speed: Bandwidth 900GB/s. Can run INT8 27B (27GB) easily on one card? No, 32GB card can hold 27GB weights + 4GB KV = 31GB. Tight but possible. Or split across two cards.
    *   This is the cheapest entry that supports INT8 precision with sufficient bandwidth.

**8. 决定研究方向**
重点调研：
1.  27B模型在INT8下的确切显存占用（含64K KV cache）。
2.  Mac Studio M2 Ultra 跑27B模型的实测Token/s（验证20 tok/s约束）。
3.  L40S vs Dual 4090 vs V100 的能效比（Joules per Token）。
4.  “重试率”对成本的敏感度分析（如果INT4错误率高10%，是否抵消了其硬件成本优势？）。

**9. 停止条件检查**
目前尚未开始正式调研，无法停止。需进入研究路径。

---

## 0. 研究问题定义

QUESTION: 在本地部署环境下，针对27B规模语言模型执行64K上下文（90:9:1比例）固定任务，哪种候选硬件方案的全生命周期每任务成本最低？
SCOPE:   包含：NVIDIA消费级/专业级显卡、Apple Silicon工作站、二手数据中心显卡；INT4/INT8/FP16量化精度对成本的影响（通过重试率关联）；不包含：云端API调用、集群超大规模并行、非推理训练任务。
OUT:     每完成任务的平均总成本（美元/任务），包含硬件折旧分摊、能源消耗、因精度不足导致的重试惩罚。
PASS:    找出一个硬件配置，其预测的每任务成本低于所有其他候选配置至少10%，且严格满足单流解码 $\ge 20$ token/s 的硬约束。
FAIL:    无法找到满足 $\ge 20$ token/s 且显存足够的硬件，或成本差异在误差范围内无法区分优劣。
HARD:    
1. 单并发流解码速度 $\ge 20$ token/s [M]
2. 单次上下文总长 64K token [S]
3. 任务包含长期不变前缀（90%缓存命中）[S]
4. 硬件必须可在本地物理部署（排除云实例）[S]
OBJ:     最小化 $Cost_{total} = \frac{Price_{hardware}}{Life\_Tasks} + Energy_{per\_task} \times Price_{kWh} + Retry\_Penalty$
COST:    
折旧: 按5年直线折旧，残值20%计算，单位 USD/task [E][±10%]
能耗: 基于满载功率与平均负载系数，单位 USD/task [M/S][±5%]
运维: 忽略人工，仅计故障停机损失，估计为折旧的5% [E][±50%]
残值: 5年后二手市场估价，取区间中值 [E][±20%]

## 1. 研究路径

### R-1.1 模型显存足迹估算
PARENT:  R-1
NOW:     未知 27B 模型在 64K 上下文下的确切显存占用 (GB)
GOAL:    得出 INT8 和 INT4 精度下，权重 + KV Cache 的最小显存需求 (GB)
GAP:     缺少特定 27B 模型的架构参数（层数、注意力头数、KV头数）
IN:      model_params, context_len, quantization_precision
OUT:     mem_weight_gb, mem_kv_cache_gb, total_mem_required_gb
KIND:    signal
RES:     数据
SHARE:   0.30
BOUND:   1.2x
TEST:    使用 vLLM 或 llama.cpp 的显存计算器，输入 27B 参数和 64K 序列长度，获取理论显存峰值
STATUS:  leaf

### R-1.2 候选硬件规格采集
PARENT:  R-1
NOW:     候选列表未定 (List)
GOAL:    获得 5 种典型本地硬件方案的显存容量、带宽、算力、功耗、价格 (Table)
GAP:     需筛选出显存 $\ge$ R-1.1 结果的候选项
IN:      total_mem_required_gb, market_price_data
OUT:     hw_spec_table
KIND:    signal
RES:     数据, 钱
SHARE:   0.20
BOUND:   ?
TEST:    查询 NVIDIA 官网、Amazon、闲鱼/ebay 二手价，记录 RTX 4090, L40S, A6000, V100, M2 Ultra 的关键指标
STATUS:  leaf

### R-1.3 解码速度可行性验证
PARENT:  R-1
NOW:     理论带宽已知，实测速度未知 (tok/s)
GOAL:    确认各候选硬件在 27B 模型上的单流解码速度是否 $\ge 20$ token/s
GAP:     理论带宽不等于实际吞吐，受软件栈、KV Cache 管理影响
IN:      hw_spec_table, total_mem_required_gb
OUT:     valid_hw_list, speed_metrics
KIND:    blackbox
RES:     算力, 人的时间
SHARE:   0.30
BOUND:   1.5x
TEST:    在目标硬件上运行 llm-bench 或 vllm benchmark，测量 27B 模型 INT8 版本的 decode latency，记录 p50/p99 速度
STATUS:  split

#### R-1.3.1 消费级显卡测试 (RTX 4090 x2)
PARENT:  R-1.3
NOW:     未测
GOAL:    测得 Dual 4090 的 decode speed
GAP:     需搭建环境
IN:      hw_spec_table[4090]
OUT:     speed_4090
KIND:    signal
RES:     算力, 人的时间
SHARE:   0.33
BOUND:   ?
TEST:    安装 CUDA 12.x, 加载 27B INT8 模型至双卡，运行 100 次生成请求，统计平均 token/s
STATUS:  leaf

#### R-1.3.2 专业卡测试 (L40S/A6000)
PARENT:  R-1.3
NOW:     未测
GOAL:    测得 L40S 的 decode speed
GAP:     需硬件
IN:      hw_spec_table[L40S]
OUT:     speed_l40s
KIND:    signal
RES:     算力, 人的时间
SHARE:   0.33
BOUND:   ?
TEST:    同 R-1.3.1，单卡 L40S 环境
STATUS:  leaf

#### R-1.3.3 Apple Silicon 测试 (M2 Ultra)
PARENT:  R-1.3
NOW:     未测
GOAL:    测得 M2 Ultra 的 decode speed
GAP:     需 Mac 设备
IN:      hw_spec_table[M2U]
OUT:     speed_m2u
KIND:    signal
RES:     算力, 人的时间
SHARE:   0.34
BOUND:   ?
TEST:    使用 MLX 框架加载 27B AWQ 模型，测量生成速度
STATUS:  leaf

### R-1.4 重试率与精度敏感性分析
PARENT:  R-1
NOW:     假设重试率为常数 (Constant)
GOAL:    建立精度(INT4/INT8)与任务通过率(Pass Rate)的关系模型
GAP:     缺乏该特定任务在不同精度下的实测通过率数据
IN:      task_accuracy_baseline, quantization_level
OUT:     retry_factor_per_precision
KIND:    blackbox
RES:     数据
SHARE:   0.20
BOUND:   2.0x
TEST:    选取 100 个典型测试用例，分别在 INT4 和 INT8 模型上运行，统计失败率，计算期望重试次数倍数
STATUS:  leaf

COUPLE: R-1.1 × R-1.2 | 共享资源: 数据 | 杀伤实验: 若 R-1.1 估算的显存小于 R-1.2 中所有低价卡的显存，则显存不再是约束，R-1.1 可降级为 nuisance；反之若显存需求超过单卡上限，则 R-1.2 必须剔除单卡方案。

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  显存容量与硬件价格的非线性关系
NODES:   R-1.1 × R-1.2
TENSION:  满足 64K 上下文 INT8 精度所需的 ~35GB 显存，使得便宜的单卡消费级显卡(24GB)失效，迫使选择昂贵的专业卡或多卡方案，导致初始硬件成本激增。
MECHANISM: Transformer 模型的 KV Cache 随上下文长度线性增长，而显存芯片密度提升速度慢于模型规模膨胀，造成“显存墙”。
STATUS:  unresolved
VERIFICATION: 检查是否有 INT4 KV Cache 方案能将总显存压至 24GB 以下，若有，则重新评估 RTX 4090 单卡方案。
IMPACT:   ±40% (硬件购置成本主导项)

CON-2  精度损失与重试成本的权衡
NODES:   R-1.4 × R-1.2
TENSION:  使用 INT4 量化可降低硬件门槛（可能允许单卡4090），但精度下降可能导致任务失败率上升，增加重试次数，从而抵消硬件节省的成本。
MECHANISM: 量化噪声累积影响复杂推理链条的准确性，尤其在长上下文中更容易出现幻觉或逻辑断裂。
STATUS:  unresolved
VERIFICATION: 执行 R-1.4 中的 TEST，对比 INT4 和 INT8 在固定测试集上的 Pass Rate 差异，若差异 <5%，则 CON-2 解除，优先选低成本硬件；若差异 >20%，则必须选高精度硬件。
IMPACT:   ±25% (运维/重试成本)

## 3. 计算与验证

CALCULATIONS:

CAL-1  显存需求估算 (INT8, 64K Context)
FORMULA: $Mem_{total} = Mem_{weights} + Mem_{kv}$. $Mem_{weights} \approx Params \times BytesPerParam$. $Mem_{kv} \approx 2 \times Layers \times Heads_{kv} \times HeadDim \times SeqLen \times BytesPerParam$.
INPUTS:  Params=27e9, BytesPerParam=1 (INT8), Layers≈32, Heads_kv≈8, HeadDim≈128, SeqLen=65536
OUTPUT: $Mem_{weights} = 27 GB$. $Mem_{kv} = 2 \times 32 \times 8 \times 128 \times 65536 \times 1 / 1024^3 \approx 8 GB$. Total $\approx 35 GB$.
ERROR:   ±10% (架构参数假设误差)
SOURCE:  [F]

CAL-2  解码速度理论下限 (Bandwidth Bound)
FORMULA: $Speed_{max} = \frac{Bandwidth}{Mem_{access}}$. 假设每次 token 生成需读取全部权重 (INT8: 27GB).
INPUTS:  Bandwidth_L40S=864 GB/s, Mem_access=27 GB
OUTPUT: $Speed_{L40S} \approx 864 / 27 \approx 32$ token/s. (Note: Actual overhead reduces this by ~20-30%, so ~22-25 tok/s).
ERROR:   ±15% (软件效率因子)
SOURCE:  [F]

CAL-3  每任务能耗成本 (L40S vs Dual 4090)
FORMULA: $Cost_{energy} = Power \times Time_{task} \times Price_{kWh}$.
INPUTS:  Power_L40S=350W, Power_Dual4090=900W. Time_task $\approx 640 tokens / 25 tok/s \approx 25.6 s$ (Decode only, Prefill negligible due to cache). Price_kWh=$0.15.
OUTPUT: 
L40S: $0.35 kW \times (25.6/3600) h \times 0.15 \approx \$0.0035$
Dual4090: $0.90 kW \times (25.6/3600) h \times 0.15 \approx \$0.0096$
ERROR:   ±20% (负载波动)
SOURCE:  [F]

CAL-4  硬件折旧成本 (5年, 10万任务假设)
FORMULA: $Cost_{dep} = \frac{Price \times (1 - Residual)}{Life\_Tasks}$.
INPUTS:  Life_Tasks=100,000 (Assumption for local dedicated use). Residual=0.2.
OUTPUT: 
L40S ($2500): $(2500 \times 0.8) / 100000 = \$0.020$
Dual4090 ($5000): $(5000 \times 0.8) / 100000 = \$0.040$
V100 32G x2 ($1000): $(1000 \times 0.8) / 100000 = \$0.008$
ERROR:   ±30% (任务量假设敏感)
SOURCE:  [E]

VERIFICATIONS:

VER-1  验证 L40S 实际解码速度
TARGET:  CAL-2 的理论速度是否成立
METHOD:  租用或借用 L40S 服务器，运行 vLLM 基准测试
RESULT:  pending (需外部执行)
CONFIDENCE: 0.8 (基于过往经验，L40S 跑 7B-13B 很快，27B 应也在 20+ tok/s)

VER-2  验证 V100 兼容性
TARGET:  V100 是否能稳定运行 27B INT8 模型
METHOD:  检查 PyTorch/CUDA 对 V100 的支持情况及显存占用监控
RESULT:  partial (V100 无 FP8 支持，INT8 可用，但驱动老旧可能有兼容性问题)
CONFIDENCE: 0.6

## 4. 结论与置信度

BOTTLENECK: R-1.4
REASON: SHARE(0.20) 连乘父节点... 实际上 R-1.4 是决定“有效成本”的关键黑盒。若重试率差异巨大，硬件选择的排序会完全翻转。其 BOUND 设为 2.0x，意味着它能让总成本翻倍或减半，是影响 OBJ 最大的不确定因素。
STOPPED: none
OPEN: CON-1 (显存墙), CON-2 (精度vs重试)
CONFIDENCE: 0.7
RECOMMENDATION: 建议优先采购 **NVIDIA L40S (48GB)** 或 **二手 Tesla V100 32GB (x2)** 进行实测。
理由：
1. L40S 单卡 48GB 足以容纳 INT8 权重 + 64K KV Cache，且带宽高，大概率满足 20 tok/s，功耗优于双 4090，维护简单。
2. V100 若预算极度敏感，是低成本高带宽选项，但需承担兼容性和无 ECC 的风险。
3. **切勿选择** 单卡 RTX 4090，除非能将 KV Cache 量化至 INT4 并证实精度损失可接受（需先执行 R-1.4 测试）。
4. Mac Studio M2 Ultra 虽省电，但 27B 模型解码速度存在跌破 20 tok/s 的风险，建议作为备选而非首选。

| 方案 | 购置价[E] | 折旧/年 | 能耗成本/年 (100k tasks) | 运维/年 | 残值/保值[E±%] | 全周期净成本 (5y) | $/验收任务(粗算) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L40S (48GB)** | $2,500 | $400 | $350 | $20 | $500 (20%) | $2,300 | **$0.023** |
| **Dual RTX 4090** | $5,000 | $800 | $960 | $40 | $1,000 (20%) | $4,500 | $0.045 |
| **Tesla V100 32GB x2** | $1,200 | $192 | $500 | $10 | $200 (15%) | $1,000 | **$0.010** |
| **Mac Studio M2 Ultra** | $4,500 | $720 | $150 | $10 | $900 (20%) | $3,500 | $0.035 |
| **Single RTX 4090 (INT4)** | $2,500 | $400 | $480 | $20 | $500 (20%) | $1,800 | $0.018 (+RetryRisk) |

*注：V100 方案成本最低，但前提是它能稳定运行且精度达标。若 V100 因驱动或精度问题导致重试率飙升，其实际成本将反超 L40S。鉴于 V100 年代久远，建议将其标记为“高风险低成本”，L40S 为“均衡型推荐”。*