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
- [x] R0.4 × 3 run 实跑（完成，6/18/18，run1 崩塌）
- [x] 评分 — score-v2.json（run1 FAIL 作废不计入 mean，mean=14.00 σ5.66）
- [x] 五版对比 R0 vs R0.1 vs R0.2 vs R0.3 vs R0.4 → COMPARISON.md

## 中期发现（r257, run1/run2 落盘时）

1. **S3 短板修复确认**：run2 五个叶子节点全部带 `SELF-CHECK: A` 行——「逐叶子自检」这条新契约在模型侧真的生效了，这是 R0~R0.3 四版都没做到的（都停在 1 分档）。
2. **C2 短板未修复**：run2 的 CALCULATIONS 里有 TPS 数值和 TCO 对比表，但逐卡表仍没有独立的 Prefill(s)/Decode(s) 两列，也没有显式瓶颈漂移结论。prompt 里「缺任一列 C2 不得分」的强制语没压住——推测因为 C2 判据写在第 3 节格式说明里、离实际填表位置太远，而 TCO 表（第 4 节附近）先满足了「有数值对比表」的心理预期。
3. **run1 暴露 max_tokens 风险**：草稿区写到一半就停（23K chars），五节完全没输出。R0.4 的草稿+五节双段结构比 R0.3 更长，16000 token 上限对长草稿不够用——若 run3 也崩，需把 run-r04.py 的 max_tokens 提到 24000 或加续写逻辑。
