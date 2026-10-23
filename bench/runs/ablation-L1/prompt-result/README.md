# prompt-result 运行结果目录（research-prompt）

# origin: ai

本目录按 **prompt 变体** 区分运行结果，每个变体一个子文件夹，包含：
- `prompt.md`：prompt 原文（从 `projects/research-prompt/bench/prompts/variants/<variant>/prompt.txt` 复制）
- `run1.md` / `run2.md` / `run3.md`：运行结果（arena-direct/battle 实跑）
- `score-v2.md`：20 分制评分（如果有）
- `README.md`：说明该变体的内容和运行结果位置

## 目录结构

```
prompt-result/
├── R0/                  # R0 = 研究 prompt 完整契约版（21 零件）
│   ├── prompt.md        # R0 prompt 原文
│   ├── run1.md          # 运行结果 1（arena-direct/battle）
│   ├── run1-score-v2.md # 20 分制评分
│   ├── run2.md
│   ├── run2-score-v2.md
│   ├── run3.md
│   ├── run3-score-v2.md
│   └── README.md        # R0 说明
├── R1-small/            # R1-small = 研究 prompt 简化版（9 零件）
├── R2-expos/            # R2-expos = R0+草稿节
├── R3-exposneg/         # R3-exposneg = R0+草稿节+负面示例
└── R4-arch/             # R4-arch = 研究 prompt 架构版
```

## 命名规则

- 文件夹名 = prompt 变体名（与 `projects/research-prompt/bench/prompts/variants/` 一致）
- `prompt.md` = prompt 原文（从 variants/ 复制）
- `runN.md` = 第 N 次运行结果
- `runN-score-v2.md` = 第 N 次运行的 20 分制评分
- `README.md` = 该变体的说明（内容/零件数/是否指定候选硬件/运行结果位置）

## 下一步

- 创建 R0/ 子文件夹（研究 prompt 完整契约版）
- 实跑 R0 × 3 run（arena-direct/battle）
- 创建 R1-R4 子文件夹（研究 prompt 简化版/架构版）
- 实跑 R1-R4 × 3 run（arena-direct/battle）
- 20 分制 rubric 评分（结构 10 分 + 内容 10 分）
- 汇总对标表（research-prompt vs 开源 vs decomp）
