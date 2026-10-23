# Arena 调研结果（2026-10-23 · bench round 5）

# origin: ai

## 调研任务

- 任务 001：bench round 5
- 内容：V2×Q1/Q2/Q3 扩题 + V3×Q4 交叉验证

## 执行结果

Arena 完成了 bench round 5，落盘 11 个文件、4 卷 + 4 评分：

| 卷 | 分 | 路径 |
|-----|-----|------|
| V2×Q1（Shannon 1948） | 89 success | `bench/runs/round5-v2-expand/arena-session/q1-run1.md` |
| V2×Q2（Bell/Shannon 分工） | 88 success | `bench/runs/round5-v2-expand/arena-session/q2-run1.md` |
| V2×Q3（量子纠错阈值定理） | 90 success | `bench/runs/round5-v2-expand/arena-session/q3-run1.md` |
| V3×Q4（LMArena 执行层差异） | 90 success | `bench/runs/round5-v3-q4/arena-session/run1.md` |

## 关键判定（如实标注，不谎报）

1. **arena 复检**：只读抓取已恢复（官网/博客今日实时打开成功），但交互式 Direct 执行仍不可用（无登录会话 + reCAPTCHA）→ claude/gemini 两档继续 N/A，未代跑冒充。
2. **通道纪律**：本轮执行者就是本 Arena 会话（型号未披露、带联网工具），全部标 degraded:true 并单列 arena-session/ 目录，不挂 qwen 名义档。
3. **分数边际不可计算**：因无同模型同通道对照卷，90−80 之类差值未读作 V3−V2 边际；本轮可读增益在证据等级——实时 URL 核对、未核 claim（C7 站点沿革）被主动剔除。
4. **附带发现 3 处锚点口径事项**（已留痕、未单方改冻结锚点）：
   - Q2 晶体管日期应为「12-16 制成 / 12-23 演示」
   - Q3 阈值证明主流归属为 AB 1997 / KLZ 1998 / Kitaev 1997 而非旧锚的 Gottesman 1996/97
   - Q1 连续容量公式 1948 原文 Part II 已给出、与 round4 卷「归属 1949」口径分歧
5. **协议收紧**：本轮 V2 不含 02.5（round4 卷曾混入）、V3 改「官方×独立第三方」双族交叉（原 DeepSeek+GLM 计划在本沙箱不可得），差异均已声明；四卷自评带窄（88–90）本身列为自利偏差嫌疑，待独立盲评复核。

## round6 第一优先

在同会话通道补 V0×Q1–Q4 + V1b×Q4 对照卷，让 V2−V0、V3−V2 首次可计算。

## 结论

- Arena 调研完成，但交互式 Direct 执行仍不可用
- 研究 prompt × 3 run 实跑需要降级（arena-direct 不可用，改用 battle 或本地）
- 开源最强模型 × 3 run 对照同样需要降级
