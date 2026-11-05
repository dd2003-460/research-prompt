# T33: research-prompt + S7V claim verification

## 背景

- R0.3 (research-prompt/bench/prompts/variants/R0.3/) = 18/20, σ=0, 当前最高分 prompt
- 失分点（稳定）：C4 保值率知识缺口(1/2) + C2 逐卡数值精度(1/2)
- pfb T30-T32 实验链结论：naive 来源注入 → M1 崩；keyword filter → 部分恢复；**two-pass claim verification → 完全恢复且 Pareto 最优**
- 本任务把 S7V 机制移植到 research-prompt 项目，测试能否突破 18 分天花板

## 分解树

```
A. 实现 R0.3V arm（S7V-style two-pass）
   A1. 为 GPU 选型问题写 FILTER_KEYWORDS（复用 pfb/research_hook.py 的 arXiv+web 检索逻辑）
   A2. Pass 1: R0.3 prompt + 注入来源（同 pfb S7F 模式）
   A3. Pass 2: 逐条核验 [F][M][S][E] 标记的数字断言 vs 来源全文，删除无支持断言
B. 执行 n=3 runs（qwen3.8-27b @ :8787，temperature 同 R0.3 baseline）
C. 评分：SCORING-RUBRIC-v2.md 20 分制，lead-agent 打分 + arena 交叉核验（hetero-judge ≥ 被评模型）
D. 对比分析：R0.3 (18/20) vs R0.3V (?) — 重点看 C2/C4 是否改善、其他维度是否退化
E. 论文/报告更新：写入 projects/research-prompt/bench/runs/ablation-L1/prompt-result/R0.3V/
```

正交性自检：A(实现)/B(执行)/C(评分)/D(分析)/E(落盘) 各管一段，无重叠。完备性：从代码到结论全链路覆盖。

## 缺口清单

- [待核验] R0.3 baseline 用的 temperature/max_tokens 具体值（run-r03.py 里是 temp=0.7, max_tokens=32000 — 需确认 3 次 run 是否同参）
- [待核验] research_hook.py 的 arXiv API 对 GPU 硬件价格/保值类查询的召回质量（pfb 实验里 Q1 PCIe/SXM 就缺数据，GPU 保值率可能同样缺）
- [待核验] arena hetero-judge 通道当前可用性（上次 ralph Playwright profile 登录失败，ego-browser 通道可用）
- [待核验] R0.3V 的 Pass 2 是否会破坏 R0.3 的五节硬性输出格式（verification prompt 要求"保持原有结构"，但五节标题一字不差是 rubric 硬约束）
- [待核验] 保值率数据来源：二手 GPU 市场（闲鱼/Taobao/eBay）数据 arXiv 不会有，需要 web 来源 — research_hook 的 ego-browser 直抓站点列表是否覆盖

## 风险

- 如果来源根本不含保值率数据（大概率），Pass 2 会删掉相关断言 → C4 可能从 1 降到 0（比 baseline 更差）。缓解：Pass 2 指令改为"无来源支持 → 保留但标注[E]+置信度降级"而非删除（针对 [E] 估计类断言）。
- 评分者偏差：lead-agent 打自己项目的分有偏。必须走 arena hetero-judge。