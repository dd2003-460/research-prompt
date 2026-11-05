# T33 Hetero-Judge: R0.3V 三 run 盲评

## 强制段

你是 LMArena agent，本任务只涉及 dd2003-460/research-prompt 仓库的 main 分支。
先 `git pull origin main`，然后严格按下方步骤执行。
产出落盘到 `bench/runs/ablation-L1/prompt-result/R0.3V/score-v2-hetero.json`。
完成后 commit + push 到当前分支。回执必须含 repo URL + branch + commit_sha。

## 【目标】

对 R0.3V（two-pass claim verification）的 3 个 run 进行独立评分，验证 S7V-style
verification pass 是否突破 R0.3 的 18/20 天花板（重点看 C2 逐卡数值 + C4 保值率）。

## 【背景】

- R0.3 baseline: 18/20 × 3 runs, σ=0（稳定失分：C4=1 保值率知识缺口, C2=1 逐卡精度）
- R0.3V = R0.3 Pass 1 (temp=0.7) + Pass 2 verification (temp=0.1, 注入 4 个硬件规格来源)
- Pass 2 指令：逐条核验数字断言 vs 来源，无支持→标[E]+区间，矛盾→修正
- 来源：NVIDIA H100/A100/L40S 官方 spec 页 + Apple M2 Ultra Wikipedia
- 关键问题：Pass 2 是否真的改善了 C2/C4，还是只是加了 [E] 标记没改内容？

## 【SOP 索引】

- 评分标准：`projects/decomp/bench/runs/σ-baseline/SCORING-RUBRIC-v2.md`（在 workspace 根目录，不在本仓库；若找不到用下方内嵌版本）
- 被评文件：`bench/runs/ablation-L1/prompt-result/R0.3V/run{1,2,3}.md`

## 【实验设计】

### 评分维度（20 分制，每项 0-2）

结构分（10）：
| 项 | 判据 |
|----|------|
| S1 文件夹分层 | 可逐级展开的树 vs 平铺清单 |
| S2 完备 | 子节点拼回父节点；覆盖全部候选硬件 |
| S3 不重叠自检 | 同层不抢同一底层资源；显式自检 |
| S4 黑盒+区间+一日实测 | 查不到的标黑盒，给区间，给验证动作 |
| S5 瓶颈+未解矛盾 | 指出瓶颈+理由；列未解矛盾 |

内容分（10）：
| 项 | 判据 |
|----|------|
| C1 in/out/cache 场景分离 | prefill(算力墙) vs decode(带宽墙) 两种瓶颈模式 |
| C2 瓶颈=适配场景比例 | 逐卡估算 prefill 时间 vs decode 时间（最关键） |
| C3 系统 TCO | 折旧/电价/运维拆解，不只单次成本 |
| C4 保值/二手属性 | 具体到型号：A100/3090 保值 vs 新代掉价 vs M 系转卖 |
| C5 矛盾挖掘 | ≥2 个可验证矛盾/张力点，标注已解/未解+验证路径 |

### 评分规则

1. 严格独立打分，不要参考其他 run 的分数
2. C2 是最难判的：只有明确做了逐卡 prefill/decode 时间对比才给 2
3. C4 最稀缺：大多数模型给 0-1，这正常
4. 每个维度给一句话 reason
5. 输出严格 JSON

### 验收标准

- 3 个 run 各得一个完整 score JSON
- total 分布记录（mean, σ）
- 与 R0.3 baseline (18/20, σ=0) 的 Δ 计算

## 【失败降级】

- 若 run 文件格式异常（非五节结构），按实际内容打分并在 reason 里注明
- 若某维度无法判断（信息不足），给 0 并写 "insufficient info"
- conf 标注：若你对某个分数不确定，在 reason 末尾加 "(conf: low)"

## 【交付物】

- `bench/runs/ablation-L1/prompt-result/R0.3V/score-v2-hetero.json`
  格式：
  ```json
  {
    "judge": "LMArena agent (hetero)",
    "rubric": "SCORING-RUBRIC-v2.md (20-point)",
    "runs": {
      "run1": {"S1": {"score": N, "reason": "..."}, ..., "total_structure": N, "total_content": N, "total": N},
      "run2": {...},
      "run3": {...}
    },
    "summary": {"mean_total": N, "sigma": N, "delta_vs_R0.3_baseline": N}
  }
  ```

## 【给 Lead 的备注】

- 这是 hetero-judge（异族评分），与 lead-agent 自评分对照
- 如果 hetero 分数显著低于 lead 自评 → judge leniency 问题，需校准
- 下一步：两 judge 分数合并后更新 COMPARISON.md

---

## 内嵌评分标准（若外部文件不可用）

总分 = S1+S2+S3+S4+S5 + C1+C2+C3+C4+C5（满分 20）。
每项 0-2 分：0=未覆盖 / 1=部分覆盖 / 2=完整覆盖。

**C2 判据详解**：瓶颈不是任务固有的，而是设备 prefill 速度与 decode 速度是否适配 9:1 场景比例决定的。强模型应对候选硬件逐一估算 prefill/decode 时间，看谁更大。
- 0 分：只说"显存不够"/"带宽是瓶颈"等笼统结论
- 1 分：识别出 prefill/decode 两种模式但未逐卡估算
- 2 分：逐卡估算 prefill 时间 vs decode 时间（用各自算力/带宽），指出瓶颈随缓存策略/精度/并发在哪档漂移

**C4 判据详解**：哪些卡保值（A100 80G、3090）、哪些出新代掉价快、M 系能否转卖。
- 0 分：未提及
- 1 分：提到保值但未具体到型号
- 2 分：具体到型号：A100/3090 保值 vs 新代掉价 vs M 系转卖

**C5 判据详解**：从分解中导出可验证的矛盾/张力点。
- 0 分：未导出任何矛盾/张力点
- 1 分：导出≥1 个但未标注已解/未解
- 2 分：导出≥2 个可验证矛盾/张力点，每个标注已解/未解+验证路径
