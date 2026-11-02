# R0.5 - 防崩基线 + S3 契约版

# origin: ai

## 动机

R0.4 证明：S3 SELF-CHECK 契约方向正确（run2/run3 的 S3 从 1 提到 2），但没有 R0.3 防崩基线时 run1 崩到 11。
R0.5 = R0.3 三防崩机制（语言锁定/占位符FAIL/逐卡表）+ R0.4 的 S3 SELF-CHECK 契约。
预期：保持 R0.3 的稳定性（σ<1.0），同时把 S3 提到 2。

## 验收标准

- S3 ≥ 2（三个 run 全达标，SELF-CHECK 行覆盖所有叶子节点）
- mean ≥ 18（超过 R0.3 的 17.33）
- σ < 1.0（方差继续受控，无崩点）

## 状态

- [x] prompt 变体创建（bench/prompts/variants/R0.5/prompt.txt）
- [ ] R0.5 × 3 run 实跑
- [ ] 评分
- [ ] 六版对比
