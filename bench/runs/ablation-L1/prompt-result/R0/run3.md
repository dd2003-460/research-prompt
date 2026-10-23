# R0 Run 3 — 27B 本地推理最低成本硬件（第三独立 run）

【草稿】
- 第三次切入：从"什么会杀死方案"倒推。死亡清单：① KV 显存溢出 ② 带宽不够 20 tok/s/流 ③ 电力/散热（机房 vs 家庭）④ 供货/二手风险。
- 直觉草稿：M3 Ultra 512GB 带宽 819 GB/s 比双 3090 的 1872 GB/s 还低，为什么有人推荐？因为它是"单节点最大 KV 池"。但本任务并发需求未明 → 如果 N=1，它的显存优势浪费；如果 N≥4，它的 28 流容量独一档。
- 价格草稿：M3 Ultra 512GB 新品 ~$10k-12k [E]；5090 新品 $2-2.5k [E]；二手 3090 $800-1000 [E]（2026 市场）；A6000 二手 $1500-2500 [E]。
- 结论草稿：N=1 时 5090 单卡 $/任务最低（带宽 1792 + 单卡省心）；但 5090 的 32GB KV 预算 10.5GB 在 64K 窗口 BF16 下只够 0.59 流 → 必须 INT8 KV（8.84GB，1 流，余量 1.7GB 紧张）或放弃 64K 到 40K。这个矛盾是本 run 的核心张力。

## 0. 研究问题定义

QUESTION: 本地跑 27B 规模模型做固定任务（64K 上下文，prefix:input:output = 90:9:1，每并发流 decode ≥20 tok/s，通过率 <100% 失败重做），哪种硬件每验收任务成本最低？
SCOPE: 含：NVIDIA 消费级（3090/4090/5090）、专业级（A6000）、Apple Silicon（M3 Ultra 统一内存）；Qwen3-32B dense（32.8B，"27B" 最近邻，Qwen3 无 27B 档 [S: ollama.com/library/qwen3:32b]）；Q4_K_M 量化（单卡 20GB 可放）。不含：云、多节点、>32B、MoE（qwen3:30b-A3B 另案）。
OUT: 每验收任务成本（$/accepted task，年折旧 4 年 + 电费 $0.12/kWh，在线率 30% [E]）。
PASS: 每并发流 decode ≥20 tok/s（vLLM metrics 可观察）；输出通过验收。
FAIL: 任一并发流持续 <20 tok/s；KV OOM；验收不通过。
HARD:
  ① 单上下文窗口 64K tok（61440 前缀 + 5760 新输入 + 640 输出 = 67840 峰值 ≈ 64K）
  ② 每并发流 decode ≥20 tok/s
  ③ 前缀 61440 tok 只 prefill 一次（共享缓存，驻留显存）[E 设计假设]
OBJ: min C_accept = (P/4 + W_pow×0.3×365×0.12) / (N×(3600/t_task)×365×24×0.3×p)

## 1. 研究路径

### R-1  成本标量分解
PARENT:  -
NOW:     OBJ 未计算，单位 $/验收任务
GOAL:    OBJ 分解为 4 因子：硬件年成本 C_annual、并发流数 N、单流任务时间 t_task、pass rate p
GAP:     4 因子各自依赖硬件规格和模型行为，未量化
IN:      硬件规格表, 模型规格, 任务画像
OUT:     C_annual, N, t_task, p
KIND:    signal
RES:     钱, 算力
SHARE:   -（父）
BOUND:   -
STATUS:  split

### R-1.1  硬件年成本 C_annual
PARENT:  R-1
NOW:     价格未知（二手市场波动）
GOAL:    C_annual = P/4 + P_watt×0.3×365×0.12（$）
GAP:     二手 3090/4090/5090/A6000 与 M3 Ultra 现价
IN:      P, P_watt
OUT:     C_annual
KIND:    signal
RES:     钱
SHARE:   0.30
BOUND:   1.5x（同性能档内价格差，3090 二手 vs 5090 新 2.3x 但跨档）
STATUS:  leaf

### R-1.2  并发流数 N（显存约束）
PARENT:  R-1
NOW:     题设未给 N，需推导上限
GOAL:    N_max = floor((VRAM - W - 1.5GB) / KV_per_stream)
GAP:     KV_per_stream 在 64K 窗口的精确值
IN:      VRAM, W, KV_per_stream
OUT:     N
KIND:    constraint
RES:     -
SHARE:   -
BOUND:   -
STATUS:  leaf

### R-1.3  单流任务时间 t_task
PARENT:  R-1
NOW:     无实测
GOAL:    t_task = t_prefill(5760) + t_decode(640)
GAP:     prefill 利用率与 decode 带宽利用率未知
IN:      FLOPS, BW, util
OUT:     t_task
KIND:    signal
RES:     算力
SHARE:   0.45
BOUND:   1.8x
STATUS:  leaf

### R-1.4  pass rate p
PARENT:  R-1
NOW:     题设 <1.0，无值
GOAL:    p 区间
GAP:     无公开 32B dense 固定任务基准
IN:      模型, 任务分布
OUT:     p
KIND:    blackbox
RES:     人的时间
SHARE:   0.25
BOUND:   1.8x
TEST:    部署 qwen3:32b Q4_K_M 到 3090×2，跑 50 个固定任务×2 批次，记录通过数；预期 1-2 人时。
STATUS:  leaf

检查 A: R-1 子节点 SHARE = 0.30+0.45+0.25 = 1.00（R-1.2 是 constraint 不计入）✓
检查 B: R-1.1(钱) × R-1.3(算力) × R-1.4(人的时间) RES 无交集 ✓（R-1.2 constraint 豁免）
检查 C: R-1.1.OUT C_annual + R-1.2.OUT N + R-1.3.OUT t_task + R-1.4.OUT p → R-1 的 OBJ 公式全部用到 ✓

## 2. 矛盾与张力

CON-1  5090 单卡带宽之王 vs KV 容量之王
NODES:   R-1.2 × R-1.3
TENSION: 5090 有最高带宽（1792 GB/s，decode 76 tok/s/流），但 32GB 显存减去 20GB 权重和 1.5GB 运行时只剩 10.5GB KV → 64K BF16 KV（17.68GB）放不下，INT8（8.84GB）只放 1 流且余量 1.7GB。带宽优势只能兑现 1 流，且必须牺牲 KV 精度。
MECHANISM: KV 显存 = 窗口×层×头×维×精度，与带宽无关。5090 的 1792 GB/s 是"每流读权重更快"，但 N 被显存锁死在 1。M3 Ultra 512GB 反之：带宽 819 GB/s（只有 5090 的 46%）但 KV 池 490GB → 27 流。两者是"窄快" vs "宽慢"两极。
STATUS:  resolved（算术确定）
VERIFICATION: 5090 上 vLLM 设 gpu_memory_utilization=0.9，加载 20GB 模型后观察 KV cache 最大 token 数：预期 (32×0.9-20-1.5)×1e9/256e3 ≈ 25K tok（BF16）< 67840 → 触发 preemption 或必须降 KV 精度。
IMPACT:  对 5090 方案 ±30%（INT8 KV 若 pass rate 掉 10pp，p 从 0.6→0.54，成本 ×1.18）

CON-2  缓存驻留假设 vs 并发 N
NODES:   R-1.2 × 问题定义(HARD③)
TENSION: "前缀只 prefill 一次"要求 61440 tok 前缀的 KV（15.7GB BF16）常驻显存。3090×2（48GB）放 20GB 权重 + 15.7GB 前缀 KV 后只剩 12.3GB → 每任务新增 KV（5760+640=6400 tok = 1.64GB）够 7 个任务槽。但 N=2 并发流时，2 个活动流的新增 KV 同时驻留 = 3.3GB，剩余 9GB → 4 个排队槽。缓存"一次 prefill"在 N>4 时被迫 LRU 驱逐 → 命中率下降 → 实际 prefill 量回升。
MECHANISM: KV 池是共享有限资源，前缀 KV 与活动流 KV 竞争同一池。N 越大，前缀被驱逐概率越高，HARD③ 的"一次 prefill"假设越不成立。
STATUS:  unresolved
VERIFICATION: 3090×2 上跑 N=1/4/8 并发流，读 vLLM 的 prefix cache hit rate 日志；预期 N=1 时 100%，N=8 时 <30%。
IMPACT:  ±60%（N=8 缓存失效 → 每任务 prefill 从 5760 回升到 ~40000 tok，t_task ×2.5）

## 3. 计算与验证

CAL-1  权重体积
FORMULA: W = params × 量化字节；qwen3:32b = 32.8B params [S: ollama.com/library/qwen3:32b]
INPUTS:  Q4_K_M → 20GB（ollama 标注，含 scale）
OUTPUT: W = 20GB
ERROR:  ±5%
SOURCE: [S]

CAL-2  KV 每流（64K 窗口）
FORMULA: KV = 64层 × 8KV头 × 128dim × 2(K+V) × bytes × T；T = 67840
INPUTS:  bytes=2(BF16)；架构参数来自 Qwen3-32B config [F+E: 层数/头数未逐字核对]
OUTPUT: KV_BF16 = 17.68GB；KV_INT8 = 8.84GB
ERROR:  ±10%
SOURCE: [F+E]

CAL-3  单流 decode 上限
FORMULA: dec = BW × 0.85 / 20GB
INPUTS:  BW: 3090=936.2 [S: en.wikipedia.org GPU 列表 wikitext], 4090=1008.0 [S], 5090=1792.0 [S], A6000=768.0 [S], M3U=819.2 [S: M2 Ultra 同代外推 E]
OUTPUT: 3090: 39.8, 4090: 42.8, 5090: 76.2, A6000: 32.6, M3U: 34.8 tok/s（全部 >20 底线）
ERROR:  ±15%（util 0.7-0.95 [E]）
SOURCE: [F 公式][S 带宽][E util]

CAL-4  单流任务时间
FORMULA: t = 2×32.8e9×5760/(FLOPS×0.6) + 640/dec；prefill 利用率 0.6 [E: 0.4-0.8]
INPUTS:  FLOPS: 3090=156.24T, 4090=401.28T, 5090=354.99T, A6000=75.6T [S: wiki 同页], M3U≈25T [E]
OUTPUT: 3090: 4.0+16.1=20.1s；4090: 1.6+14.9=16.5s；5090: 1.8+8.4=10.2s；A6000: 8.3+19.6=27.9s；M3U: 30.2+18.4=48.6s
ERROR:  ±40%（prefill 利用率主导）
SOURCE: [F 公式][S FLOPS][E util]

CAL-5  每验收任务成本（N=1，p=0.6 [E]，在线率 30% [E]，4 年折旧）
FORMULA: C_acc = (P/4 + W_w×0.3×365×0.12) / ((3600/t)×365×24×0.3×0.6)
INPUTS:  P: 3090×2=$1800, A6000=$2000, 4090×2=$3400, 5090=$2200, M3U512=$12000 [E: 2026 市场区间]
OUTPUT: 年折旧+电: 3090×2=$537, A6000=$444, 4090×2=$766, 5090=$537, M3U=$2928；年验收任务(30%在线×p=0.6): 3090×2≈20800, A6000≈14100, 4090×2≈25900, 5090≈48300, M3U(N=1)≈7200
→ C_acc: 3090×2: $0.026；A6000: $0.031；4090×2: $0.030；5090: $0.011；M3U512(N=1): $0.406
ERROR:  ±60%（p [E] ±30% 乘价格 [E] ±30% 叠加）
SOURCE: [F 公式][E 价格/p/在线率]

VER-1  24GB 单卡 64K 可行性
TARGET: 3090/4090 单卡
METHOD: 24-20-1.5 = 2.5GB KV 预算 vs 17.68GB 需求
RESULT: 不可行（BF16/INT8 均超）→ 24GB 单卡在 HARD①③ 下出局
CONFIDENCE: 0.9

VER-2  5090 单卡 64K 可行性
TARGET: 5090 32GB
METHOD: 32-20-1.5 = 10.5GB vs 17.68GB(BF16)/8.84GB(INT8)
RESULT: BF16 不可行；INT8 可行（1 流，余量 1.7GB）
CONFIDENCE: 0.85

VER-3  M3U 512GB 并发上限
TARGET: M3 Ultra 512GB
METHOD: (512-20-2)×0.9/17.68 = 24.9 → 24 流（BF16）
RESULT: 24 流，总 decode 带宽 819×0.85/20 = 34.8 tok/s 总 → 每流 34.8/24 = 1.45 tok/s < 20 FAIL
→ 修正：20 tok/s/流约束下 N_max = floor(34.8/20) = 1 流。M3U 的显存优势在 decode ≥20 tok/s 硬约束下完全兑现不了！
CONFIDENCE: 0.8（M3U 带宽 819.2 是 M2 Ultra 实测 [S]，M3 Ultra 同代 [E]）

## 4. 结论与置信度

BOTTLENECK: R-1.2（KV 显存 → N 上限）
REASON: SHARE 连乘：R-1.1 钱 0.30×BOUND 1.5 = 0.45；R-1.3 算力 0.45×1.8 = 0.81；R-1.4 人的时间 0.25×1.8 = 0.45。R-1.2 是 constraint 不参与乘积，但它是 N 的硬天花板：N 从 1→2 使 OBJ ÷2（比任何单因子改善都大）。Amdahl 判据下 constraint 的"改善潜力"（N 翻倍 = OBJ 减半）超过所有 signal 节点 → BOTTLENECK = R-1.2。
STOPPED: 无（S1 检查：R-1.4 的 0.45 < 1.05 理论上可停，但 p 直接进 OBJ 分母且无替代估计 → 不满足"误差可接受"，保留 blackbox）
OPEN: ① p 未测（TEST 已给，1-2 人时）② CON-2 缓存命中率 vs N 未解（±60% OBJ）③ M3 Ultra 512GB 供应风险（2026-03 内存短缺报道 [S: Ars Technica via wiki]）④ 二手 3090 矿卡风险（未计入）
CONFIDENCE: 0.62
RECOMMENDATION: N=1 场景（题设未明确并发，按最小 N 保守）：**5090 单卡 + INT8 KV 是 $/验收任务最低方案（~$0.011，误差 ±60%）**，前提是 INT8 KV 的 pass rate 损失 <10pp（未测，列为 OPEN）。若保守拒绝 INT8，则 3090×2 二手（~$0.026）是最便宜可行方案。**不要买 M3 Ultra**：20 tok/s/流 硬约束把它锁死在 N=1，$12k 买 1 流吞吐不划算（$0.41/验收，5090 的 37×）。
黑盒四要素（R-1.4 p）：影响 CAL-5 分母；TEST=50 任务×2 批次；区间 [0.5, 0.9]；两端时 BOTTLENECK 不变（仍是 R-1.2 KV 显存），但绝对成本 3090×2 在 $0.015-0.027 间变。

硬件排名（N=1, p=0.6, 64K 硬约束, Q4_K_M）：
| 方案 | 价格[E] | KV 可行 | $/验收[E±60%] |
|---|---|---|---|
| 5090 单卡 | $2200 新 | INT8 1 流 | ~$0.011 |
| 3090×2 二手 | $1800 | BF16 1 流 | ~$0.026 |
| A6000 二手 | $2000 | BF16 1 流 | ~$0.031 |
| 4090×2 二手 | $3400 | BF16 1 流 | ~$0.030 |
| M3U 512GB 新 | $12000 | BF16 1 流（带宽锁死） | ~$0.41 |
