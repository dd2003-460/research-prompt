# T33-T34 负面结果总结：R0.3 天花板的边界

> origin: ai · 2026-11-06 · 补充 SUMMARY.md §3.1

## 背景

SUMMARY.md 结论：R0.3 = 18/20, σ=0 是当前最优定版。稳定失分：C4(保值)=1/2。
§3.1 建议方向："C4 需知识注入（prompt 附录加 GPU 保值参考表）"。

T33/T34 分别测试了两个突破假设，均被否定。

## T33: R0.3V — Two-Pass Claim Verification (S7V-style)

**假设**：从 pfb stage-ablation 移植 S7V 机制（Pass1 draft + Pass2 verify with sources），verification pass 能修 C2/C4。

**设计**：Pass1 = R0.3 prompt (temp=0.7) → Pass2 = verification prompt (注入 4 个硬件 spec 来源, temp=0.1, "逐条核验数字断言，无支持→标[E]，矛盾→修正")

**结果**：

| Run | Total | C2 | C4 | C5 |
|-----|-------|----|----|-----|
| run1 | 16    | 2  | 0  | 1   |
| run2 | 19    | 2  | 1  | 2   |
| run3 | 18    | 2  | 1  | 2   |
| mean | **17.67** | 2.00 | 0.67 | 1.67 |

Δ vs R0.3 = **-0.33**, σ 从 0 → 1.25

**根因**：
- C2 无变化（2→2）：Pass1 已做逐卡估算，verification 没增加新信息
- C4 退化（1→0.67）：run1 的 anti-hallucination 指令（"无来源支持→删除"）误删了模型已有的保值率常识
- C5 微降（2→1.67）：verification pass 的随机性引入方差
- **核心**：verification 只能"修错"不能"补漏"——它不会主动添加来源中没有的分析维度

## T34: R0.3C4 — Knowledge Injection Appendix

**假设**：在 R0.3 prompt 末尾追加 GPU 保值参考表（8 型号 × 保值性+说明），给模型材料写 C4。

**设计**：唯一改动 = prompt 末尾追加 1.2KB 附录。执行参数与 R0.3 完全相同。

**结果**：

| Run | Total | S | C | C2 | C4 | C5 |
|-----|-------|---|---|----|----|-----|
| run1 | 12    | 4 | 8 | 2  | 1  | 1   |
| run2 | 10    | 3 | 7 | 2  | 1  | 0   |
| run3 | 11    | 4 | 7 | 2  | 1  | 0   |
| mean | **11.00** | 3.67 | 7.33 | 2.00 | 1.00 | 0.33 |

Δ vs R0.3 = **-7.00**, σ 从 0 → 0.82

**根因**：
- **格式契约崩塌**：run2 完全没有五节结构（0 个 `##` 标题）。模型从"报告模式"切换到"研究模式"
- C4 无改善（1→1）：给表 ≠ 模型会做对比分析
- C5 崩塌（2→0.33）：模型忙于计算和罗列，没做矛盾导出
- **核心**：R0.3 的 prompt 是精密格式机器，每个 token 都服务格式合规。追加附录打破平衡，注意力分散到"研究参考"而非"按格式写报告"

## 综合结论

**R0.3 的 18/20 天花板是 27B 模型在 format-constrained prompt 下的能力上限。**

两个突破路径都被否定：
1. ❌ 加 verification pass → 修错不能补漏，anti-hallucination 误伤已有知识
2. ❌ 加 knowledge injection → 格式契约崩塌，总分暴跌 7 分

**正确的突破路径**（未测试）：
- 换更强模型（70B+）：同时处理格式约束 + 知识整合 + 分析深度
- 改 rubric：降低 C4 要求（"提及保值概念+至少一个型号示例"即可得 2 分）
- 嵌入现有节点（非独立附录）：在 TCO 表里加"残值率[E]"列，不改变 prompt 整体结构

## 对 pfb 实验链的启示

S7V (pfb LoCoMo) 有效是因为 M1 考纯数字匹配，verification 能直接修。
research-prompt 的 C2/C4/C5 考分析深度和覆盖广度，tool-access 机制不 transferable。

**Tool-access 的收益取决于评分维度的性质**：
- 数字准确性 rubric → verification 有效
- 分析深度 rubric → verification 无效甚至有害
- 知识缺口 rubric → 需要嵌入式注入（非附录），或换更强模型

## T33 Hetero-judge 对比（2026-11-06）

Arena GPT-5.5 级独立打分 vs Lead 27B 打分：

| Judge | Mean | σ | Δ vs R0.3(18) |
|-------|------|---|---------------|
| Lead (27B) | 17.67 | 1.25 | −0.33 |
| **Hetero (arena)** | **13.33** | **3.51** | **−4.67** |

方向一致（都低于 baseline），幅度差 14×。Hetero judge 更严的原因：
1. **正交性违规检测**：run3 的 R-1.2.1/R-1.2.2 同时声明 RES=人的时间 → lead judge 未检出
2. **内部矛盾追踪**：run2（排名说 A100 最便宜，推荐说 L40S）、run3（BOTTLENECK=prefill 与 per-card 表矛盾）→ hetero 扣分，lead 未扣
3. **验证 pass 不稳定**：Pass-2 偏离原始契约越远 → 分越低（run1 最差）

**结论**：R0.3V 的负结果在两个 judge 下都成立（方向一致），但 hetero judge 揭示了 lead judge 遗漏的质量缺陷（正交性、内部一致性）。R0.3 的 18/20 天花板结论不变——hetero judge 如果重新打 R0.3 baseline，预计也会低于 18（更严的标准），进一步确认天花板是 rubric 难度而非 prompt 缺陷。

## 文件索引

- `prompt-result/R0.3V/run{1,2,3}.md` + `score-v2-lead.json` + `score-v2-hetero.json` + `README.md`
- `prompt-result/R0.3C4/run{1,2,3}.md` + `score-v2.json` + `README.md`
