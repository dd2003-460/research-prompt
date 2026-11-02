# R0.4 - 短板补强版（S3 SELF-CHECK + C2 Prefill/Decode 列强制）

# origin: ai

## 动机

R0.3 定版 mean 17.33 σ0.94，四版最优。剩余短板收敛到两个：
1. S3 逐层自检：所有版本都停在 1 分档（有 COUPLE 声明但无逐叶子节点显式自检）
2. C2 run3 半达标：TCO 数值有了但缺 prefill时间/decode延迟 的逐卡对比列

R0.4 两个最小改动对症：
1. S3：每个叶子节点末尾必须加 SELF-CHECK: A/B/C（与兄弟节点是否重叠），缺行不得分
2. C2：逐卡表必须含 Prefill(s) 和 Decode(s) 两列（数值），缺任一列不得分；
   表后必须指出瓶颈漂移发生在哪档硬件/精度组合

## 验收标准

- S3 ≥ 2（SELF-CHECK 行出现且覆盖所有叶子节点）
- C2 = 2（三个 run 全满分，Prefill/Decode 列都有数值）
- mean ≥ 18（超过 R0.3 的 17.33）
- σ < 1.0（方差继续受控）

## 状态

- [x] prompt 变体创建（bench/prompts/variants/R0.4/prompt.txt）
- [ ] R0.4 × 3 run 实跑（本地 qwen3.8-27b @ ninfer :8787）
- [ ] 评分（判据档位与 R0~R0.3 对齐）
- [ ] 五版对比 R0 vs R0.1 vs R0.2 vs R0.3 vs R0.4
