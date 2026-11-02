# R0.2 - 对称契约版（C3/C4 保住 + C5 保底 + COST 紧凑化）

# origin: ai

## 动机

R0.1 结果：mean 16.33（同口径修正后 vs R0 基线 16.00，Δ+0.33）。
- 修复达成：C3 1→2，C4 0→1
- 真实代价：C2 2→1（逐卡 prefill/decode 对比被 COST 展开说明挤占）、C5 2→1.67（run2 只挖 1 个矛盾）

R0.2 两个最小改动：
1. COST 字段紧凑化：一行一因子「因子名: 数值+单位 [来源][±%]」，禁止展开说明 → 省 token 还给 C2
2. 第 2 节加对称契约：CONTRADICTIONS ≥2 个 CON-N，缺项矛盾维度不得分 → 保 C5

## 验收标准

- C3 ≥ 2 且 C4 ≥ 1（不回退）
- C5 ≥ 2（三个 run 都 ≥2 个矛盾）
- C2 回 2（逐卡对比恢复）
- mean ≥ 17（超过 R0 基线 16.00 和 R0.1 的 16.33）

## 状态

- [x] prompt 变体创建（bench/prompts/variants/R0.2/prompt.txt）
- [x] R0.2 × 3 run 实跑（完成，17/17/10）
- [x] 评分（SCORING-RUBRIC-v2.md 20 分制，判据档位与 R0/R0.1 对齐）
- [x] 三版对比 R0 vs R0.1 vs R0.2 → COMPARISON.md
