# R0.6 - 防崩基线 + S3/C2 双契约版

# origin: ai

## 动机

R0.5 全面崩塌（7.33）证明：在 R0.3 基础上只加一条契约时，模型可能整体失守。
但 R0.4 证明 S3 SELF-CHECK 方向正确（遵循时 S3 1→2），R0.3 的逐卡表让 C2 到 1.67。
R0.6 = R0.3 三防崩 + S3 SELF-CHECK + C2 Prefill/Decode 列强制（比 R0.3 更严格的 C2）。
如果 R0.6 也崩，结论就是：qwen3.8-27b 对长 prompt 多契约叠加的遵循能力有硬上限。

## 验收标准

- S3 ≥ 2（SELF-CHECK 行覆盖叶子节点）
- C2 ≥ 2（Prefill(s)/Decode(s) 两列都有数值）
- mean ≥ 18, σ < 1.0
- 如果再次崩塌 → 确认模型能力边界，消融实验终止于 R0.3

## 状态

- [x] prompt 变体创建
- [x] ×3 run 实跑（完成，6/18/16）
- [x] 评分 + 七版对比 → COMPARISON.md
