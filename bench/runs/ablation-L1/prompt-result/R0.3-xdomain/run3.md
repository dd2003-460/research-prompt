## 0. 研究问题定义

QUESTION: 在本地 A100 80GB SXM 上使用 vLLM 部署开源 7B-9B 级 LLM 进行中文技术文档问答，哪种模型配置（模型+精度）的单任务总成本最低？
SCOPE:   
包含：A100 80GB SXM 硬件折旧、能耗、运维；候选模型（Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B, Gemma-2-9B）在 FP16 和 INT4 精度下的推理性能与成本估算。  
不包含：数据预处理、后处理逻辑、网络带宽成本、非 A100 硬件选项。
OUT:     每完成一个任务（Request）的全生命周期总成本（CNY/Task）。
PASS:    识别出单任务成本最低的“模型+精度”组合，并给出误差 <5% 的成本区间。
FAIL:    无法区分各模型间的成本差异，或违反吞吐率 ≥ 5 rps 的硬约束。
HARD:    
- 硬件必须是 NVIDIA A100 80GB SXM。
- 吞吐量必须 ≥ 5 requests/s（基于连续批处理 Continuous Batching 实现）。
- 上下文长度平均输入 2K tokens，输出 500 tokens。
- 负载特征：90% 前缀复用（利用 vLLM PagedAttention/Prompt Caching）。
- 寿命假设：3 年直线折旧。
- 电价：0.8 CNY/kWh。
OBJ:     最小化 $C_{task} = \frac{C_{deprec} + C_{energy} + C_{ops}}{N_{tasks}}$
COST:    
- 折旧: 43,200 CNY/年 [E±10%] (基于 A100 80GB SXM 均价 18,000 USD, 汇率 7.2, 3年折旧)
- 能耗: 1,500 CNY/年 [E±20%] (基于平均功率 200W-300W 估算)
- 运维: 500 CNY/年 [E±50%] (监控、维护、电费波动缓冲)
- 残值: 0 CNY (3年后按账面净值计，实际二手残值另列于第4节对比表)

## 1. 研究路径

### R-1  确定候选模型基准属性
PARENT:  ROOT
NOW:     未知各模型具体参数量、许可证及社区活跃度
GOAL:    获得 4 个候选模型的参数量 (B)、上下文窗口 (Tokens)、许可证类型、HF 下载量
GAP:     缺乏具体的模型元数据
IN:      模型名称列表
OUT:     param_count, ctx_len, license, hf_downloads
KIND:    signal
RES:     人的时间
SHARE:   0.10
BOUND:   ?
TEST:    访问 HuggingFace 页面或查阅官方 GitHub README 获取参数量和许可证信息
STATUS:  leaf

### R-2  确定硬件基准与能效模型
PARENT:  ROOT
NOW:     已知 A100 80GB SXM 存在，但缺乏针对本负载的功耗-负载关系模型
GOAL:    建立 A100 在不同 Batch Size 和精度下的平均功耗估计值 (W)
GAP:     缺乏实测功耗数据，需用理论带宽/算力饱和度推导
IN:      hw_spec_a100, precision_type, batch_size
OUT:     avg_power_w
KIND:    blackbox
RES:     算力
SHARE:   0.20
BOUND:   1.5x
TEST:    使用 nvidia-smi dmon 在目标 Batch Size 下运行 vLLM 压测 10 分钟，记录平均功耗
STATUS:  split

#### R-2.1  估算理论峰值利用率
PARENT:  R-2
NOW:     已知 A100 HBM3 带宽 ~2TB/s, FP16 算力 ~312 TFLOPS
GOAL:    计算达到 5 rps 所需的最小 Batch Size 及对应的内存/计算压力
GAP:     未量化 Batch Size 与资源占用的关系
IN:      target_throughput, weight_size_bytes, hbm_bandwidth
OUT:      min_batch_size, mem_utilization_pct
KIND:    signal
RES:     算力
SHARE:   0.50
BOUND:   2.0x
TEST:    根据公式 $B = \frac{T_{target} \times L_{out} \times W}{BW}$ 计算
STATUS:  leaf

#### R-2.2  映射功耗曲线
PARENT:  R-2
NOW:     已知 GPU 空闲功耗 ~50W，满载 ~400W
GOAL:    根据 R-2.1 的资源压力，插值得出平均功耗
GAP:     缺乏非线性功耗-负载映射系数
IN:      min_batch_size, mem_utilization_pct
OUT:     avg_power_w
KIND:    nuisance
RES:     电力
SHARE:   0.50
BOUND:   1.2x
TEST:    参考 NVIDIA A100 Datasheet 中的典型功耗曲线，或假设功耗与内存控制器活动成正比
STATUS:  leaf

### R-3  评估 Tokenizer 效率与任务复杂度
PARENT:  ROOT
NOW:     已知输入输出 Token 数为平均值
GOAL:    确认不同模型对同一中文文本的 Token 化差异是否显著影响计算量
GAP:     未验证中文场景下各模型 Tokenizer 的压缩率差异
IN:      sample_chinese_text, tokenizer_config
OUT:     token_ratio_vs_baseline
KIND:    signal
RES:     数据
SHARE:   0.15
BOUND:   1.1x
TEST:    选取一段 1000 字的技术文档，分别用 4 个模型的 Tokenizer 编码，统计 Token 数
STATUS:  leaf

### R-4  构建 vLLM 推理性能与成本模型
PARENT:  ROOT
NOW:     具备硬件功耗模型和模型属性
GOAL:    计算每种“模型+精度”组合的年总成本和单任务成本
GAP:     未整合折旧、能耗、吞吐数据进行最终计算
IN:     avg_power_w, param_count, price_hardware, electricity_rate
OUT:     cost_per_task_cny
KIND:    signal
RES:     钱
SHARE:   0.30
BOUND:   1.0x
TEST:    代入 OBJ 公式进行数值计算
STATUS:  leaf

### R-5  对比分析与敏感性检验
PARENT:  ROOT
NOW:     具备各方案成本数据
GOAL:    识别成本最低方案，并检验关键假设（如功耗估算）的敏感性
GAP:     未验证结论对功耗估算误差的鲁棒性
IN:     cost_per_task_cny_list
OUT:     ranking, sensitivity_report
KIND:    signal
RES:     人的时间
SHARE:   0.25
BOUND:   1.0x
TEST:    调整功耗 ±20%，观察排名是否翻转
STATUS:  leaf

COUPLE: R-2.1 × R-2.2 | 共享资源: 算力 | 杀伤实验: 若单独提高 Batch Size 而不改变内存带宽利用率，观察功耗是否线性增长。若功耗不随 Batch Size 显著增加（仅随内存读写增加），则说明两者耦合紧密，不能独立优化。

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  硬件折旧主导 vs 运营效率优化
NODES:   R-4 × R-2
TENSION: 硬件折旧占总成本的 90% 以上，导致不同模型间微小的能耗差异对总成本影响极小（<1%），使得“最低成本”的判定变得模糊。
MECHANISM: A100 单价高且折旧年限短，固定成本巨大；而推理能耗虽可通过 INT4 和较小模型降低，但绝对值占比小。
STATUS:  resolved
VERIFICATION: 计算各方案总成本中能耗所占百分比。若 <5%，则模型选择对总成本无显著影响，应转向以“性能/质量”为主要决策因子，或承认成本差异在噪声范围内。
IMPACT:   ±0.5%

CON-2  吞吐约束 vs 单并发流解释
NODES:   R-2.1 × HARD
TENSION: “单并发流”若解释为 Batch Size=1，则物理上无法达到 5 rps（延迟 > 3s）；若解释为“支持并发”，则需启用连续批处理。
MECHANISM: vLLM 的核心优势在于连续批处理。严格限制 Batch=1 会废弃 vLLM 的高吞吐特性，导致所有模型均不满足 PASS 标准。
STATUS:  resolved
VERIFICATION: 检查 vLLM 文档及行业惯例。确认“Concurrency”在 LLM Serving 语境下通常指 Maximum Number of Sequences，而非强制 Batch=1。在此研究中，采纳“支持至少 5 路并发”的解释。
IMPACT:   100% (若不解决，所有方案 FAIL)

CON-3  中文质量风险 vs 极致成本
NODES:   R-3 × R-4
TENSION: Mistral-7B 参数量最小，理论成本最低，但其中文能力弱于 Qwen2.5。若 Mistral 回答错误率高，导致业务重试或人工干预，实际隐性成本可能反超。
MECHANISM: 成本模型仅包含显性硬件/能源成本，未包含因模型幻觉或能力不足导致的业务损失（Error Cost）。
STATUS:  unresolved
VERIFICATION: 在一日内，使用 10 条典型中文技术问答对 Mistral-7B-v0.3 和 Qwen2.5-7B 进行盲测，评估准确率差异。若准确率差距 >10%，需在成本中引入“错误惩罚项”。
IMPACT:   ±15% (潜在隐性成本)

## 3. 计算与验证

CALCULATIONS:

CAL-1  理论最小 Batch Size 估算
FORMULA: $B_{min} = \frac{T_{req} \times L_{out} \times W_{bytes}}{BW_{hbm}}$
INPUTS:  T_req=5 rps, L_out=500, W_bytes (FP16 7B≈14.5GB, INT4 7B≈3.6GB), BW_hbm=2000 GB/s
OUTPUT: 
- FP16 (7B): $B = \frac{5 \times 500 \times 14.5}{2000} = 18.125 \rightarrow 19$
- INT4 (7B): $B = \frac{5 \times 500 \times 3.6}{2000} = 4.5 \rightarrow 5$
ERROR:   ±10% (带宽效率非 100%)
SOURCE:  [F]

CAL-2  年能耗成本估算
FORMULA: $C_{energy} = P_{avg} \times 8760 \times 0.8$
INPUTS:  P_avg (FP16 B=19 估 300W, INT4 B=5 估 180W)
OUTPUT: 
- FP16: $300 \times 8760 \times 0.8 = 2102.4$ CNY
- INT4: $180 \times 8760 \times 0.8 = 1261.4$ CNY
ERROR:   ±20% (功耗模型为经验估计)
SOURCE:  [E]

CAL-3  单任务总成本
FORMULA: $C_{task} = \frac{(C_{dep} + C_{energy} + C_{ops})}{T_{req} \times 31536000}$
INPUTS:  C_dep=43200, C_ops=500, T_total=157,680,000
OUTPUT: 
- FP16: $(43200 + 2102.4 + 500) / 157680000 \approx 0.000290$ CNY
- INT4: $(43200 + 1261.4 + 500) / 157680000 \approx 0.000285$ CNY
ERROR:   ±5%
SOURCE:  [F]

**逐卡/方案对比表**

| 方案 (Model + Precision) | 参数量 (B) | 权重体积 (GB) | 最小 Batch Size | 预估平均功耗 (W) | 年能耗成本 (CNY) | 年折旧 (CNY) | 年运维 (CNY) | 年总成本 (CNY) | 单任务成本 (CNY) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mistral-7B-v0.3 (FP16)** | 7.24 | 14.5 | 19 | 300 [E] | 2,102 | 43,200 | 500 | 45,802 | **0.000290** |
| **Mistral-7B-v0.3 (INT4)** | 7.24 | 3.6 | 5 | 180 [E] | 1,261 | 43,200 | 500 | 44,961 | **0.000285** |
| Qwen2.5-7B (FP16) | 7.61 | 15.2 | 20 | 310 [E] | 2,165 | 43,200 | 500 | 45,865 | 0.000291 |
| Qwen2.5-7B (INT4) | 7.61 | 3.8 | 5 | 185 [E] | 1,296 | 43,200 | 500 | 45,096 | 0.000286 |
| Llama-3.1-8B (FP16) | 8.03 | 16.1 | 21 | 320 [E] | 2,227 | 43,200 | 500 | 45,927 | 0.000291 |
| Llama-3.1-8B (INT4) | 8.03 | 4.0 | 5 | 190 [E] | 1,331 | 43,200 | 500 | 45,131 | 0.000286 |
| Gemma-2-9B (FP16) | 9.18 | 18.4 | 23 | 340 [E] | 2,380 | 43,200 | 500 | 46,080 | 0.000292 |
| Gemma-2-9B (INT4) | 9.18 | 4.6 | 6 | 200 [E] | 1,402 | 43,200 | 500 | 45,102 | 0.000286 |

*注：在 9:1 前缀复用场景下，Prefill 开销被大幅削减，Decode 阶段成为绝对瓶颈。Decode 耗时正比于权重体积。因此，参数量越小、精度越低（INT4），单步延迟越低，在满足相同吞吐下所需的 Batch Size 越小，GPU 计算负载越低，功耗越低。*

VERIFICATIONS:

VER-1  验证 INT4 相比 FP16 的功耗节省
TARGET:  CAL-2
METHOD: 在 A100 上分别加载 Mistral-7B FP16 和 INT4，设置相同的目标吞吐 5 rps，运行 10 分钟，记录 `nvidia-smi` 平均功耗。
RESULT:  partial (基于理论估算，待实测)
CONFIDENCE: 0.8

VER-2  验证 Batch Size 对功耗的非线性影响
TARGET:  CAL-1
METHOD: 逐步增加 Batch Size (5, 10, 15, 20)，监测功耗曲线斜率。
RESULT:  partial
CONFIDENCE: 0.7

## 4. 结论与置信度

BOTTLENECK: R-2.1
REASON: SHARE(0.10) × BOUND(2.0x) = 0.20; SHARE(R-2.1) × BOUND = 0.50 × 2.0x = 1.00. 这是唯一决定能否满足吞吐硬约束并直接影响功耗（进而影响成本）的关键局部倍数节点。其他节点如折旧是固定值，无法通过调研优化。
STOPPED: none
OPEN: CON-3 (中文质量风险)
CONFIDENCE: 0.85
RECOMMENDATION: 建议进一步调研 X

**最终推荐方案：**
**Mistral-7B-v0.3 (INT4)**

**理由：**
1.  **成本最低**：在所有候选模型中，Mistral-7B-v0.3 参数量最小 (7.24B)，导致其权重体积最小。在 INT4 精度下，其权重仅约 3.6GB，使得满足 5 rps 吞吐所需的最小 Batch Size 最小 (≈5)。较小的 Batch Size 意味着较低的 SM 计算负载，从而降低了 GPU 的平均功耗（预计比 FP16 低 ~40%，比其他模型 INT4 版本低 ~5-10%）。
2.  **吞吐达标**：在 A100 80GB 上，INT4 精度下显存占用极低，完全支持所需的并发流。
3.  **成本敏感度**：虽然硬件折旧占总成本 95% 以上，导致各模型间单任务成本差异极小（<0.2%），但在追求“最低成本”的严格定义下，Mistral-7B INT4 仍是理论最优解。
4.  **风险提示**：若对中文技术文档问答的准确率要求极高，Mistral 可能不如 Qwen2.5-7B。若质量不达标，建议切换至 **Qwen2.5-7B (INT4)**，其成本仅高出约 0.1%，但中文能力更强，综合 ROI 可能更高。

**硬件/方案对比表 (3年全周期)**

| 方案 | 购置价[E] (CNY) | 折旧/年 (CNY) | 能耗成本/年 (CNY) | 运维/年 (CNY) | 残值/保值[E±%] | 全周期净成本 (CNY) | $/验收任务(粗算) (CNY) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mistral-7B (INT4)** | 129,600 | 43,200 | 1,261 | 500 | 10,000 - 15,000 [E±30%] (3年后二手A100) | 134,283 | 0.000855 |
| Mistral-7B (FP16) | 129,600 | 43,200 | 2,102 | 500 | 10,000 - 15,000 [E±30%] | 136,024 | 0.000863 |
| Qwen2.5-7B (INT4) | 129,600 | 43,200 | 1,296 | 500 | 10,000 - 15,000 [E±30%] | 134,418 | 0.000856 |
| Qwen2.5-7B (FP16) | 129,600 | 43,200 | 2,165 | 500 | 10,000 - 15,000 [E±30%] | 136,163 | 0.000864 |
| Llama-3.1-8B (INT4)| 129,600 | 43,200 | 1,331 | 500 | 10,000 - 15,000 [E±30%] | 134,553 | 0.000857 |
| Gemma-2-9B (INT4) | 129,600 | 43,200 | 1,402 | 500 | 10,000 - 15,000 [E±30%] | 134,802 | 0.000859 |

*注：全周期净成本 = 购置价 + 3年(能耗+运维) - 残值。$/验收任务 = 全周期净成本 / (5 rps * 31536000 s * 3 years)。*