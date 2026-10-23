【草稿】
- 换个切入点：不先想硬件，先想"每验收任务到底包含哪些物理动作"。
  动作分解：① 新输入 5760 tok 的 prefill（算力受限）② 640 tok 的 decode（带宽受限）③ 失败重做（乘 1/p）。
- 草稿账（推翻一次）：直觉以为 decode 占大头 → 算一下 5760 tok prefill 在 3090 上要 5760×2×32.8B/156TFLOPS≈2.4s（理论），实际 compute 利用率低可能 5-15s；decode 640 tok @ 40 tok/s = 16s。所以 decode 仍占 ~60-70%，带宽还是主角。没推翻。
- 第二次草稿账：如果 prefix 每任务都要重新 prefill（缓存失效场景）→ 61440 tok prefill 变成 100s+，prefill 反超 decode。题设说"长期不变的前缀可以被缓存"→ 缓存命中是设计假设，不是每任务现实。这个假设要标 [E] 并做敏感性。
- 价格草稿：二手 3090 市场价 2026 年已跌到 ~$800-1000 [E]；5090 新品 $2000-2500 但供应紧张 [E]。二手 4090 ~$1500-1800。
- 结论草稿：24GB 单卡在 64K 窗口全 BF16 KV 下必死（20GB 权重+17.4GB KV>24GB）→ 单卡方案必须 INT8/FP8 KV 或放弃 64K。INT8 KV 对 32B 的 pass rate 影响未知 → blackbox。

## 0. 研究问题定义

QUESTION: 本地跑 27B 规模模型做固定任务（64K 上下文，缓存:新输入:输出 = 90:9:1，每并发流 decode ≥20 tok/s，通过率 <100% 失败重做），哪种硬件每验收任务成本最低？
SCOPE: 含：消费级/专业级 NVIDIA GPU、Apple Silicon 统一内存；不含：云 API、多节点集群、>32B 模型。模型取 Qwen3-32B（32.8B dense，27B 规模最接近的开源 dense，Qwen3 无 27B 档 [S: ollama.com/library/qwen3:32b]）。
OUT: $/验收任务（美元每通过验收的任务，含折旧+电费，不含人力）。
PASS: 每并发流 decode ≥20 tok/s [可观察：vLLM metrics 的 token/s per stream]；输出通过任务判定。
FAIL: decode <20 tok/s 持续 >1 任务周期；KV OOM；判定不通过。
HARD:
  - 64K token 单上下文窗口（61440 缓存 + 5760 新输入 + 640 输出 = 67840 ≈ 64K 峰值）
  - 每并发流 decode ≥20 tok/s
  - 前缀缓存命中为前提（每任务只 prefill 新输入 5760 tok）[E 设计假设，敏感性见 CAL-4]
OBJ: min 每验收任务成本 = (年折旧 + 年电费) / 年验收任务数

## 1. 研究路径

### R-1  成本函数分解
PARENT:  -
NOW:     目标函数 OBJ 是黑箱，单位 $/验收任务
GOAL:    OBJ 拆成 3 个可测因子：任务墙钟时间 t_task、年运行小时 H、硬件年成本 C，OBJ = C×(1/p)/（3600/t_task×H）
GAP:     不知道 t_task 由 prefill 还是 decode 主导
IN:      任务画像, 硬件规格, pass rate p
OUT:     t_task, H, C, p
KIND:    signal
RES:     算力
SHARE:   0.35
BOUND:   1.0x
STATUS:  split

### R-1.1  prefill 时间（新输入 5760 tok）
PARENT:  R-1
NOW:     无实测值
GOAL:    t_prefill = 5760 tok / prefill 吞吐(tok/s)
GAP:     prefill 吞吐 = f(FLOPS, 权重体积, 利用率)
IN:      FLOPS, W_bytes, util_c
OUT:     t_prefill
KIND:    signal
RES:     算力
SHARE:   0.45
BOUND:   2.5x（prefill 在任务时间里占比 30%-80%，取决于缓存命中假设）
STATUS:  leaf

### R-1.2  decode 时间（640 tok）
PARENT:  R-1
NOW:     无实测值
GOAL:    t_decode = 640 / decode 吞吐
GAP:     decode 吞吐 = BW×util / W_bytes（带宽受限）
IN:      BW, W_bytes, util_d
OUT:     t_decode
KIND:    signal
RES:     算力
SHARE:   0.40
BOUND:   2.0x（decode 占任务时间 20%-80%）
STATUS:  leaf

COUPLE: R-1.1 × R-1.2 | 共享资源: 算力 | 杀伤实验: 同一硬件上分别跑纯 prefill 5760 tok 和纯 decode 640 tok 两个 microbenchmark，若两者实测速度之比 ≠ (5760/640)×(各自 roofline 预测比)，说明共享资源假设破裂——实验应看到两阶段速度各自独立于对方。

### R-1.3  pass rate p
PARENT:  R-1
NOW:     题设 <100%，无具体值
GOAL:    p 的区间估计
GAP:     无公开"32B dense 固定任务"pass rate 基准
IN:      模型规模, 任务类型
OUT:     p
KIND:    blackbox
RES:     人的时间
SHARE:   0.15
BOUND:   1.8x
TEST:    部署 qwen3:32b Q4_K_M，跑 50 个同分布固定任务，统计通过数/50；重复 2 个批次取均值。预计 2 人时。
STATUS:  leaf

检查 A: R-1 子节点 SHARE = 0.45+0.40+0.15 = 1.00 ✓
检查 B: RES：R-1.1(算力) × R-1.2(算力) 有交集 → 已写 COUPLE 行；R-1.3(人的时间) 与两者无交集 ✓
检查 C: R-1.1.OUT t_prefill + R-1.2.OUT t_decode → R-1 汇总成 t_task（父 IN 用到）✓；R-1.3.OUT p → R-1 的 OBJ 公式用到 ✓

## 2. 矛盾与张力

CONTRADICTIONS:

CON-1  缓存命中假设 vs 每任务成本
NODES:   R-1.1 × 问题定义(HARD)
TENSION: 题设"前缀可缓存"意味着每任务只 prefill 5760 tok；但实际部署中缓存命中依赖调度（前缀驻留显存），并发流多时 LRU 驱逐导致命中率下降 → prefill 时间可能 ×2-3。
MECHANISM: 64K 窗口的 KV 每流占 17.4GB [F: CAL-2]，48GB 显存放 2 流后 KV 预算耗尽 → 第 3 流的前缀必须重新 prefill。缓存命中是"单流独占"的假设，并发 >2 时失效。
STATUS:  unresolved
VERIFICATION: 在 48GB 硬件上跑 3 并发流，用 vLLM 的 prefix cache 命中率指标（gpu_cache_usage 与 prefix 命中日志）实测命中率；预期 <1 流/17.4GB 时命中率 <50%。
IMPACT:  ±50%（若命中率掉到 50%，t_prefill 从 5760 变 33000 tok 平均，任务时间 ×1.8）

CON-2  20 tok/s 硬约束 vs 并发数
NODES:   R-1.2 × 问题定义(HARD)
TENSION: 20 tok/s/流 × N 流 = 20N tok/s 总 decode 吞吐需求；总吞吐上限 = BW×util/W。3090 单卡上限 39 tok/s → 只支持 1 流（39/20=1.95 流，向下取整 1 流还有余量，2 流则 19.5 tok/s <20 FAIL）。
MECHANISM: decode 是共享带宽资源，N 流均分 → 每流 = 总/N。N=2 时每流 19.5 tok/s，差 2.5% 不过线。这是刀刃约束：3090 单卡恰好卡在 1-2 流边界。
STATUS:  resolved（计算确定，见 CAL-3）
VERIFICATION: 3090 上 vLLM 跑 2 并发流，读 per-stream token/s，预期 19-20 tok/s（±5% 噪声，边界情形可能 flaky pass）。
IMPACT:  决定 N 的上限 → 直接决定年任务数（×N）→ 对 OBJ ±100%（N 从 1 到 2 翻倍）。

## 3. 计算与验证

CALCULATIONS:

CAL-1  权重体积
FORMULA: W_bytes = params × bits/8；qwen3:32b params=32.8e9 [S: ollama]
INPUTS:  Q4_K_M → 32.8e9×0.5625/1 ≈ 18.4e9... ollama 标注 20GB（含 scale 因子）取 20GB [S]；Q8_0 = 35GB [S]
OUTPUT: W(Q4)=20GB, W(Q8)=35GB
ERROR:   ±5%
SOURCE:  [S] ollama.com/library/qwen3:32b

CAL-2  KV cache 每流
FORMULA: KV = L×H_kv×D×2×2B×T；L=64, H_kv=8, D=128, T=67840（61440+5760+640）
INPUTS:  Qwen3-32B 架构参数 [F: 从 HuggingFace config 推断，未逐字核对 → 标 F+E]
OUTPUT: 64×8×128×4×67840 = 17.68GB (BF16)；INT8 = 8.84GB
ERROR:  ±10%（层数/KV头数若 ±1 则 ±5%）
SOURCE:  [F+E]

CAL-3  单流/双流 decode 上限
FORMULA: tok/s_per_stream = BW×util/W；util=0.85 [E: 0.7-0.95]
INPUTS:  3090 BW=936.2 [S: en.wikipedia.org NVIDIA 桌面 GPU 列表 wikitext], W=20GB
OUTPUT: 3090 单流 39.8 tok/s；2 流均分 19.9 tok/s（<20 FAIL 边界）；4090 单流 42.8；5090 单流 76.2；A6000 单流 32.6；M3U(819.2 GB/s) 单流 34.8
ERROR:  ±15%
SOURCE:  [F 公式] + [S 带宽] + [E util]

CAL-4  每任务墙钟时间（3090×1 单流，缓存命中场景）
FORMULA: t = 5760/(BW×0.6×W⁻¹×1000) + 640/(39.8)；prefill 取算力受限系数 0.6 [E: 0.4-0.8]
INPUTS:  R-1.1, R-1.2 的因子
OUTPUT: t_prefill = 5760×20/(936.2×0.6) = 204.9s... 重算: prefill 每 tok 读权重 20GB? 错——prefill 是 compute-bound，FLOPs=2×32.8e9×5760=3.78e14，3090 FP16 156 TFLOPS×0.6 利用率=94e12 → t=4.0s [F 修正]；decode=640/39.8=16.1s → t_task=20.1s
ERROR:  ±40%（prefill 利用率 0.3-0.8 区间 → t 在 16-30s）
SOURCE:  [F 公式]，利用率 [E]

VERIFICATIONS:

VER-1  24GB 单卡可行性
TARGET: 3090/4090 单卡能否跑 64K 窗口
METHOD:  算术：24GB - 20GB(Q4 权重) - 1.5GB(运行时) = 2.5GB KV 预算 vs 需求 17.68GB(BF16)/8.84GB(INT8)
RESULT:  fail（两种 KV 精度都不够）→ 24GB 单卡在 64K 硬约束下不可行，必须双卡 48GB 或 INT4 KV（质量风险未测）
CONFIDENCE: 0.9

VER-2  20 tok/s 边界
TARGET: 3090 双卡（带宽 1872.4 GB/s）2 流 decode
METHOD: CAL-3 外推：1872.4×0.85/20/2 = 39.8 tok/s/流
RESULT:  pass（39.8 ≥ 20，余量 99%）
CONFIDENCE: 0.75（util [E] 拖累）

VER-3  48GB 方案并发上限
TARGET: 3090×2 / A6000 的 64K 并发流数
METHOD: (48-20-1.5)/17.68 = 1.54 → 1 流（BF16 KV）；INT8: (26.5)/8.84 = 3 流
RESULT:  BF16 1 流 / INT8 3 流
CONFIDENCE: 0.8

## 4. 结论与置信度

BOTTLENECK: R-1.2（decode 时间 → 吞吐 → 成本）
REASON: SHARE 连乘 R-1(0.35)→R-1.2(0.40) = 0.14；×BOUND(2.0x) = 0.28。对比 R-1.1: 0.35×0.45×2.5 = 0.39 更高？→ 修正：R-1.1 的 BOUND 2.5x 是缓存失效敏感性，正常场景 prefill 仅占任务 20%（4s/20s），BOUND 应降为 1.3x → 0.35×0.45×1.3 = 0.20 < 0.28。BOTTLENECK = R-1.2 成立（Amdahl: decode 改善 2× 省 28% 成本，prefill 改善 2× 省 20%）。
STOPPED: 无节点停止（S1 未命中：R-1.3 的 SHARE×BOUND = 0.35×0.15×1.8 = 0.094 < 1.05 理论上够 S1，但 p 直接乘 OBJ，改 p 从 0.5→0.9 改 OBJ 1.8×，>1.05 阈值，S1 不成立 → 继续保留为 blackbox）
OPEN: CON-1（缓存命中率 vs 并发）未解；R-1.3 p 未测（TEST 已给出，2 人时）
CONFIDENCE: 0.6
RECOMMENDATION: 结论方向可信（48GB 级双卡 > 24GB 单卡，带宽 ≥900 GB/s 足够），但绝对成本数（$0.01-0.03/验收）置信度低，因 p [E] 和 prefill 利用率 [E] 各贡献 ±40-80%。建议：① 先花 2 人时实测 p（TEST 已写）② 实测一台 3090 的 prefill 5760 tok 耗时，把 CAL-4 的 ±40% 压到 ±10% 后再做采购决策。

硬件排名（64K 硬约束 + 单流 ≥20 tok/s + Q4_K_M）：
| 方案 | 价格[E] | 并发流 | 单流 decode | $/验收任务(粗算) |
|---|---|---|---|---|
| 3090×2 二手 | $1600-2000 | 1(BF16)/3(INT8) | 39.8/39.8 | ~$0.02 |
| A6000×1 二手 | $1500-2500 | 1/3 | 32.6 | ~$0.025 |
| 4090×2 二手 | $3000-3600 | 1/3 | 42.8/流 | ~$0.03 |
| 5090×1 新 | $2000-2500 | 0-1(KV 10.5GB) | 76.2 | ~$0.03（KV 紧张需 INT8） |
| M3 Ultra 256GB | $9000-11000 新 | 13(BF16) | 34.8 总共享 | ~$0.06 |

注：5090 单卡 32GB 的 KV 预算 = 32-20-1.5 = 10.5GB < 17.68GB → 64K BF16 放不下，INT8 8.84GB 可放 1 流（余量小）。5090 的带宽优势（1792 GB/s）在只能跑 1 流时不转化为并发优势，仅缩短 decode 16s→8.4s。
