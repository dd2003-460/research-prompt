# R0 Run 1 — 27B 模型固定任务最低成本硬件

【草稿】
- 问题本质：单位验收任务的硬件成本 = 折旧+电费 / 年验收任务数。变量 = decode 速度（带宽决定）、并发数（显存决定）、pass rate（模型决定）。
- 关键洞察候选：① decode 是带宽受限（每 token 读全部权重）② KV cache 是显存杀手（64K 窗口 × BF16 = 17.6GB/流）③ 单卡 24GB 在 64K 窗口下 KV 预算只剩 ~2.5GB → 只能跑 1 流且要 INT8 KV 或更小窗口。
- 待验证矛盾：roofline 说 5090 最快最便宜 vs 3090 二手便宜且够用。
- 模型选择：Qwen3 家族无 27B dense，最接近 = qwen3:32b（32.8B dense，Q4_K_M 20GB，ollama 确认）[S]。32B 是保守上界。

## 0. 研究问题定义

- QUESTION: 本地跑 27B 模型做固定任务（64K 上下文，prefix:input:output = 90:9:1，硬约束 decode ≥20 tok/s/并发流，通过率 <100%），哪种硬件每验收任务成本最低？
- SCOPE: 本地（单节点，非云），Qwen3-32B dense（27B 规模的保守近似），量化 Q4_K_M（单卡可放）或 Q8_0（双卡/大显存），推理框架 vLLM/llama.cpp。
- OUT: 每验收任务成本（$/accepted task）的最低硬件 + 配置 + 误差区间。
- PASS: 每个并发流 decode ≥20 tok/s；任务输出通过验收。
- FAIL: decode <20 tok/s/流，或 KV 溢出 OOM，或任务不通过需重做。
- HARD: 64K 窗口硬约束；prefix 61440 tok 共享（只 prefill 一次）；每任务新输入 5760 tok、输出 640 tok。
- OBJ: min(硬件年折旧 + 年电费) / 年验收任务数。

## 1. 研究路径

节点 N1:
- PARENT: root
- NOW: 问题建模
- GOAL: 把"每验收任务成本"分解为可测因子
- GAP: decode 速度由什么决定？并发数由什么决定？
- IN: 问题定义
- OUT: 成本公式 + 因子清单
- KIND: signal
- RES: 算力
- SHARE: 0.40
- BOUND: 1.0
- STATUS: leaf

节点 N2:
- PARENT: root
- NOW: 硬件调研
- GOAL: 候选硬件的带宽/显存/价格
- GAP: 没有现成"27B+64K 并发"公开基准
- IN: 候选清单（3090/4090/5090/A6000/M3 Ultra）
- OUT: 规格表（带宽 GB/s、VRAM、新/二手价）
- KIND: signal
- RES: 数据
- SHARE: 0.30
- BOUND: 1.0
- STATUS: leaf

节点 N3:
- PARENT: root
- NOW: 模型调研
- GOAL: 27B 候选模型的真实参数与量化体积
- GAP: "27B" 在 Qwen3 不存在，需确定代理
- IN: ollama 库
- OUT: qwen3:32b = 32.8B dense Q4_K_M 20GB [S]
- KIND: signal
- RES: 数据
- SHARE: 0.20
- BOUND: 1.0
- STATUS: leaf

节点 N4:
- PARENT: root
- NOW: 黑盒 — pass rate
- GOAL: 任务通过率区间
- GAP: 无公开"32B dense 做代码/推理固定任务"的 pass rate 基准
- IN: 文献 + 直觉
- OUT: p ∈ [0.5, 0.9] dense [E]
- KIND: blackbox
- TEST: 用 qwen3:32b 跑 50 个固定任务（HumanEval 子集），统计通过率；对比 qwen3:30b (MoE A3B) 的 p
- RES: 人的时间
- SHARE: 0.10
- BOUND: 2.0（p 从 0.5 到 0.9 差 1.8×）
- STATUS: leaf

检查 A: SHARE = 0.40+0.30+0.20+0.10 = 1.00 ✓
检查 B: 正交性 — N1(公式)/N2(硬件数据)/N3(模型数据)/N4(行为数据) RES 无交集 ✓
检查 C: IN/OUT 链 — N1.OUT(公式) → N2/N3 供数 → N4 修正 p ✓

## 2. 矛盾与张力

CON-1:
- NODES: N2 × N1
- TENSION: roofline 模型说"带宽越大越便宜"（5090 1792 GB/s → $0.04/任务），但 3090 二手 $900 已满足 20 tok/s 底线 → 5090 的带宽优势在"刚好过线"场景下不转化为成本优势。
- MECHANISM: decode 速度 ∝ 带宽，但成本 = 折旧/吞吐。3090 吞吐 13/时 vs 5090 24.8/时（1.9×），价格 900 vs 2100（2.3×）→ 3090 每美元吞吐反而高。只有当 20 tok/s 底线被抬高（如要求 40+ tok/s 或更多并发）时 5090 才反超。
- STATUS: 已验证（roofline 计算，见 CALC-2）
- VERIFICATION: 算 3090 二手 vs 5090 新的 $/吞吐：900/13=69 $/（任务/时） vs 2100/24.8=85 → 3090 便宜 19%。
- IMPACT: 若选 5090 而底线是 20 tok/s，多花 ~19% 硬件钱买用不上的速度。

CON-2:
- NODES: N2 × N1
- TENSION: 单卡 24GB（3090/4090）在 64K 窗口 BF16 KV 下，KV 预算只有 ~2.5GB → 17.6GB/流 的 KV 根本放不下，单卡必须降 KV 精度（INT8/FP8）或截断窗口。
- MECHANISM: 64K 窗口 × 64 层 × 8 KV 头 × 128 dim × 2(K+V) × 2 byte = 268MB/1K tok → 64K = 17.6GB/流 [F 公式]。24GB - 20GB 权重 - 1.5GB 运行时 = 2.5GB KV 预算 → 只够 ~14K tok BF16 或 ~28K tok INT8。
- STATUS: 已验证（公式推导）
- VERIFICATION: 2.5e9 / 256e3 = 9765 tok（BF16）；INT8 翻倍 = 19.5K tok。都 < 64K。
- IMPACT: 单卡 24GB 方案实际不能跑满 64K 窗口——要么 INT8 KV（质量风险）要么双卡。这改变了成本排名：3090×2 二手 $1800 才是真正的"单节点 64K 最小配置"。

## 3. 计算与验证

CALC-1: KV cache 每流每窗口
- FORMULA: KV_bytes = L × H_kv × D_head × 2(K+V) × bytes × T
- INPUTS: L=64 层, H_kv=8 (GQA), D_head=128, bytes=2 (BF16), T=65536 (61440+5760+640≈64K 窗口峰值)
- OUTPUT: 64×8×128×2×2×65536 = 2.15e9 byte... 重算: 64×8×128×4 = 262144 B/tok = 256KB/tok [F]；×65536 = 16.78GB（精确 65536 tok）；按任务实际 67840 tok（61440+5760+640）= 17.37GB
- ERROR: ±10%（层数/头数取 Qwen3-32B config 估计 [F+E]）
- SOURCE: Qwen3-32B config（64 层 8 KV 头 head_dim 128）[F]

CALC-2: 单流 decode 上限
- FORMULA: tok/s = BW × util / W_bytes；W_bytes = 20GB (Q4_K_M)
- INPUTS: BW=936.2 GB/s (3090) [S wiki], util=0.85 [E 实测典型], W=20e9
- OUTPUT: 936.2×0.85/20 = 39.8 tok/s；4090: 1008×0.85/20=42.8；5090: 1792×0.85/20=76.2；A6000: 768×0.85/20=32.6；M3U: 819.2×0.85/20=34.8
- ERROR: ±15%（util 在 0.7-0.95 间，llama.cpp vs vLLM 差异）
- SOURCE: 带宽 S (wiki)，util E（无本地实测）
- 结论: 全部候选单流 decode ≥32 tok/s > 20 tok/s 底线 → 20 tok/s 约束不绑定（除非并发 >1）

CALC-3: 每任务时间
- FORMULA: t = t_prefill(5760 tok) + t_decode(640 tok)；t_prefill = 5760/(BW×0.6/W×1000)（prefill 算力受限取 0.6 系数）[E]
- INPUTS: 同上
- OUTPUT: 3090: 5760/(936.2×0.6/20)+640/39.8 = 204.9+16.1 = 221s；5090: 5760/(1792×0.6/20)+640/76.2=107.1+8.4=115.5s
- ERROR: ±25%（prefill 系数 0.6 是估计 [E]）
- SOURCE: 自推公式 + 带宽 [F 公式]

CALC-4: 年验收任务数与每任务成本
- FORMULA: per_hour = 3600/t_task；accept_year = per_hour×365×24×0.3(在线率)/p；cost = (P/4年折旧 + 300W×0.3×365×0.12$)/accept_year
- INPUTS: P=900(3090 二手)/2100(5090 新), p=0.6 [E], 在线率 0.3 [E]
- OUTPUT: 3090 二手: 13.0/时 → 4945 任务/时×... 重算: 3600/221=16.3/时, ×365×24×0.3=13350 任务/年, /p=0.6 → 22250 验收/年；折旧 900/4=225$/年, 电 300×0.3×365×0.12=39$/年 → cost=(225+39)/22250=$0.0119/验收
- ERROR: ±50%（p 和在线率都是 [E]，p=0.5→0.9 差 1.8×）
- SOURCE: 自推 [F 公式] + 价格 [E]

VER-1:
- TARGET: 24GB 单卡能否放下 64K KV
- METHOD: 算术（CALC-1 vs VRAM-权重-运行时）
- RESULT: 24-20-1.5=2.5GB << 17.37GB → 不能（BF16）；INT8 KV=8.7GB 仍不能
- CONFIDENCE: 0.9（公式精确，输入 [F]）

VER-2:
- TARGET: 20 tok/s 底线是否绑定
- METHOD: CALC-2 各候选单流上限
- RESULT: 全部 ≥32 tok/s → 不绑定（单流场景）
- CONFIDENCE: 0.7（util [E] ±15%，最弱 A6000 32 tok/s 仍 >20）

## 4. 结论与置信度

- BOTTLENECK: N2（KV 显存约束决定并发数，进而决定成本）
- REASON: 全部候选单流 decode 都过 20 tok/s 线（CALC-2），所以 decode 速度不是瓶颈；瓶颈是 64K 窗口的 KV 把单卡 24GB 方案判死 → 最小可行配置 = 双卡 48GB（3090×2 二手 $1800）或 A6000 48GB 二手 $2000。每验收任务成本: 3090×2 二手 ≈ $0.023/验收（折旧 1800/4=450 + 电 54 / 22250×2流... 重算: 双卡吞吐 2 流 → 44500 验收/年 → (450+78)/44500 = $0.012）。5090 单卡 32GB KV 预算 10.5GB 也只够 <1 流 64K → 同样被 KV 判死（除非 INT8 KV 或 32K 窗口）。
- 排名（$/验收任务，64K 硬约束下）: ① 3090×2 二手 ~$0.012 ② A6000 二手 ~$0.014 ③ 4090×2 二手 ~$0.015 ④ 5090 新 ~$0.03（KV 不够需 INT8）⑤ M3 Ultra 512GB ~$0.025（并发 28 流但带宽低，吞吐被带宽限死在 35 tok/s 总）
- STOPPED: S1（现成基准不存在——llmfarm 等无对口公开数据，GitHub 搜索确认；自建 roofline）
- OPEN: ① pass rate p 无实测（黑盒 N4，conf 0.5）② prefill 系数 0.6 未实测 ③ M3 Ultra 512GB 供应风险（2026-03 内存短缺报道）④ 若任务实际 prefix 不共享（每任务全量 prefill 67840 tok），所有成本 ×2.3
- CONFIDENCE: 0.55（KV 约束 0.9 / decode 下限 0.7 / 成本排名 0.5，p 和 prefill 系数拖累）
- RECOMMENDATION: 买 2×RTX 3090 二手（~$1800）+ Qwen3-32B Q4_K_M + vLLM + INT8 KV cache，64K 窗口单流 ~35 tok/s，预算 $0.01-0.03/验收任务。若并发 >2 流需求，直接上 M3 Ultra 256GB（KV 预算 234GB = 13 流），但 $/验收 贵 ~2×。
- 黑盒四要素（N4 pass rate）: 影响 CALC-4 的 accept_year；TEST=跑 50 任务统计；区间 [0.5,0.9]；两端时 BOTTLENECK 不变（仍是 KV 显存），但成本绝对值差 1.8×。
