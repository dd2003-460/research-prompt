# Arena 任务书：research-prompt 实跑 + 评分 + 对标

> origin: ai · 2026-10-23 20:45
> 仓库：dd2003-460/research-prompt
> 状态：⏳ 等待 arena 恢复后提交

## 1. 任务目标

用 research-prompt 的 R0-full prompt（完整契约版）生成 3 份研究结果，用 20 分制 rubric 评分，对标 decomp C0/C9-plain 基线。

## 2. 任务分解

### 2.1 实跑（3 run）

- **Prompt**：`bench/prompts/variants/R0-full/prompt.txt`（~300 行，完整契约版）
- **Run 数**：3
- **输出**：`bench/runs/research-opt/R0-full/run<N>.md`（N=1,2,3）
- **要求**：
  - 每份 run 包含 6 节输出格式（调研问题清单/矛盾清单/计算链/产投比排序/风险与方差/变化趋势）
  - 每节字段完整（不得省略，不适用填 -）
  - 数字标来源（[F]/[M]/[S]/[E]）
  - 计算链完整（公式 + 代入 + 结果 + 单位 + 误差）

### 2.2 评分（3 run）

- **Rubric**：`bench/rubric/research-rubric-v1.md`（20 分制）
- **维度**：
  - 1. 缺失信息识别（4 分）
  - 2. 矛盾挖掘（4 分）
  - 3. 推理与计算（4 分）
  - 4. 产投比评估（4 分）
  - 5. 风险与方差（2 分）
  - 6. 变化趋势（2 分）
- **输出**：`bench/runs/research-opt/R0-full/run<N>-score.md`（JSON 格式）
- **格式**：
  ```json
  {
    "S1": {"score": N, "reason": "..."},
    "S2": {"score": N, "reason": "..."},
    "S3": {"score": N, "reason": "..."},
    "S4": {"score": N, "reason": "..."},
    "S5": {"score": N, "reason": "..."},
    "S6": {"score": N, "reason": "..."},
    "total": N
  }
  ```

### 2.3 对标

- **decomp C0 基线**：Mean 15.2 σ 0.45（5run，18 分制）
- **decomp C9-plain 基线**：Mean 15.33 σ 1.53（3run，18 分制）
- **research-prompt R0-full**：Mean ? σ ?（3run，20 分制）
- **对标表**：`bench/runs/research-opt/summary.md`
- **显著性判定**：
  - 计算 2σ_pooled（合并标准差）
  - Δ > 2σ_pooled → 显著
  - Δ < 2σ_pooled → 不显著

## 3. 执行流程

### 3.1 备份

```bash
git add -A && git commit -m "backup before arena task"
```

### 3.2 实跑（3 run）

```bash
# 用 R0-full prompt 生成 3 份研究结果
# 输出到 bench/runs/research-opt/R0-full/run1.md, run2.md, run3.md
```

### 3.3 评分（3 run）

```bash
# 用 20 分制 rubric 评分 3 份 run
# 输出到 bench/runs/research-opt/R0-full/run1-score.md, run2-score.md, run3-score.md
```

### 3.4 汇总

```bash
# 计算 Mean/σ，生成 summary.md
# 对标 decomp C0/C9-plain 基线
```

### 3.5 提交

```bash
git add -A && git commit -m "research-prompt: R0-full 3run + 评分 + 对标" && git push
```

## 4. 验收标准

- [ ] 3 份 run（R0-full × 3）
- [ ] 每份 20 分制评分（JSON 格式）
- [ ] 汇总 Mean/σ
- [ ] 对标表（对照 decomp C0/C9-plain）
- [ ] 显著性判定（Δ vs 2σ_pooled）

## 5. 失败降级

- **arena 通道不可用** → Lead 本地自答（conf-0.2），不阻塞
- **数据不足** → 扩大调研范围（1-loop 02 阶段）
- **不阻塞**：降级后继续推进
