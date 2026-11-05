# R0.3V - Two-Pass Claim Verification (S7V-style)

# origin: ai

## 动机

R0.3 = 18/20, σ=0，稳定失分：C4(保值率)=1, C5(矛盾挖掘)=2 但无余量。
T30-T32 (pfb stage-ablation) 结论：two-pass "draft then verify" loop 是 Pareto 最优臂。
本变体测试：将 S7V 机制移植到 research-prompt 项目，能否突破 18 分天花板？

## 设计

- Pass 1: R0.3 prompt → qwen3.8-27b temp=0.7 max_tokens=32000（与 baseline 完全相同）
- Pass 2: verification prompt (注入 4 个硬件规格来源 + 逐条核验指令) → temp=0.1
- 来源: NVIDIA H100/A100/L40S 官方 spec 页 + Apple M2 Ultra Wikipedia (direct fetch)
- 核验指令核心: "数字断言 vs 来源比对；无支持→标[E]+区间；矛盾→修正"

## 结果 (lead-agent score, qwen3.8-27b)

| Run | Total | C2 | C4 | C5 |
|-----|-------|----|----|-----|
| run1 | 16    | 2  | 0  | 1   |
| run2 | **19**| 2  | 1  | 2   |
| run3 | 18    | 2  | 1  | 2   |
| **mean** | **17.67** | 2.00 | 0.67 | 1.67 |

R0.3 baseline: mean=18.00, σ=0.00
Δ = **-0.33** (σ 从 0 爆到 1.25)

## 判读

**假设被否定。** Two-pass verification 没有突破天花板，反而引入了方差：

1. **C2 无变化** (2.00 → 2.00)：Pass 1 已经做了逐卡估算，verification 没增加新信息
2. **C4 退化** (1.00 → 0.67)：run1 的 verification pass 把保值率讨论删了（"来源中没有→不要编造"指令过强），其余两 run 保留
3. **C5 微降** (2.00 → 1.67)：run1 的矛盾点标注不够完整
4. **方差引入**：σ 从 0 → 1.25，verification pass 的 temperature=0.1 仍有随机性

## 根因分析

S7V 在 pfb LoCoMo 上有效是因为：M1 (事实准确性) 是纯数字匹配，verification 能直接修。
research-prompt 的 rubric 不同：
- C2/C4/C5 考的是**分析深度和覆盖广度**，不是数字对错
- Verification pass 只能"修错"不能"补漏"——它不会主动添加来源中没有的分析维度
- 更糟：anti-hallucination 指令 ("无来源支持→删除") 会误伤模型已有的正确知识（如保值率常识）

## 结论

**S7V mechanism 不 transferable 到 analysis-depth rubrics。** 
Tool access 的收益取决于评分维度的性质：
- 数字准确性 rubric (pfb M1) → verification 有效
- 分析深度 rubric (research-prompt C2/C4/C5) → verification 无效甚至有害

下一步方向（若继续）：
- 不是加 verification pass，而是改 Pass 1 prompt 本身（如显式要求 C4 型号对比表）
- 或：多来源注入时包含二手市场数据源（闲鱼/eBay 价格页），让 Pass 1 有材料写 C4

## 文件

- run{1,2,3}.md — 最终输出（Pass 2 后）
- score-v2-lead.json — lead-agent 评分
- score-v2-hetero.json — arena hetero-judge 评分（待完成）
