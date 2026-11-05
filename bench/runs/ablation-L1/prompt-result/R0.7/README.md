# R0.7 - Minimal C2 Column Enforcement

# origin: ai · 2026-11-06

## 动机

SUMMARY §5.3: "C2 可改善：在 R0.3 基础上加一句'对比表必须含 prefill_ms 和 decode_tok_s 两列，逐卡填数'"。
R0.4 测试了 SELF-CHECK（+4行）→ 崩塌。T34 测试了知识注入附录（+1.2KB）→ 崩塌。
R0.7 假设：**最小改动**（仅强化 C2 列规格，+107B）不会触发崩塌，且能维持/提升 C2。

## 设计

- **唯一改动**: R0.3 的 C2 表格要求从"列至少含 prefill 时间(秒)和 decode 单 token 延迟(毫秒)"
  → "列必须含 Prefill(s)、Decode_tok_s、TPS、$/任务 四列 + 禁止定性描述"
- 无 SELF-CHECK 契约（R0.4 教训：+4行指令 → run1 崩塌到 6 分）
- 无附录（T34 教训：+1.2KB → 格式契约崩塌，mean 11.00）
- 执行参数与 R0.3 完全相同：temp=0.7, max_tokens=32000, qwen3.8-27b

## Delta vs R0.3

| 项 | R0.3 | R0.7 | Δ |
|----|------|------|---|
| Prompt size | 10326 B | 10433 B | +107 B |
| C2 列要求 | ≥2 列 (prefill s, decode ms) | =4 列 (P/D/TPS/$) + 禁定性 | 强化 |
| SELF-CHECK | 无 | 无 | — |
| 附录 | 无 | 无 | — |

## 预期

- C2: 2→2（维持）或 2→2（四列全满足仍得 2，机械判据不加分）
- 其他维度: 不受影响（改动仅 107B，注意力分散可忽略）
- 崩塌风险: ~0%（远小于 R0.4 的 +4行 / T34 的 +1.2KB）
- 如果 C2 维持 2 且总分维持 18 → 确认 R0.3 天花板是模型能力上限，非 prompt 缺陷
- 如果 C2 降到 1 → 四列要求反而让模型困惑（负面结果，有价值）

## 状态

- [x] prompt.txt 创建
- [x] run-r07.py 创建
- [x] score-v2.py 创建
- [ ] 3 runs 完成
- [ ] 评分
- [ ] 结论写入 SUMMARY
