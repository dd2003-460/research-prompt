# R0.5 retest-32k — max_tokens 上限对照实验

## 动机

R0.5 ×3 run（max_tokens=16000）全部停在草稿阶段（8/7/7），六版对比初判为「模型指令遵循随机性」。
但对照数据存疑：R0.3 的 run 输出 30-36K chars 却完整收尾，说明 16000 这个参数值与「字符数」的关系不透明。
本实验固定 R0.5 prompt + temperature 0.7，只改 max_tokens 到 32000（服务器上限 32768，40000 实测 HTTP 400 `max_output_tokens_exceeded`），跑 3 次对照。

## 结果（2026-11-03 02:07-02:16, qwen3.8-27b @ :8787）

| run | chars | finish_reason | completion_tokens | 五节结构 | SELF-CHECK |
|-----|-------|---------------|-------------------|----------|------------|
| runA (retest-32k) | 25500 | stop | 11349 | **5/5** | 4（覆盖全部 4 叶子节点） |
| runB (b1) | 21819 | stop | ? | 0/5（停草稿，结尾 "I will formulate the response"） | 0 |
| runC (b2) | 15017 | stop | ? | 0/5（停草稿，结尾 "Ready to write."） | 0 |

## 发现

1. **max_tokens=16000 不是截断根因**：runA 在 32000 下正常收尾且五节齐全；而 runB/runC 同样 32000 仍停在草稿——同一参数下三种结局并存。
2. **指令遵循是概率性的**：3/3 全崩（16000）→ 1/3 成功（32000）。样本量太小不能定概率，但「prompt 无法完全控制」的判断方向不变，程度比初判轻。
3. **runA 的结构退化**：研究路径只有 4 个平铺一级节点（R-1~R-4），无三级树（R0.3 有 R-1.1.1 级深度）——S3 满分靠的是 SELF-CHECK 全覆盖，不是拆解深度。若 lead 复核 S1/S2 大概率降档。
4. **英文漂移仍在**：runB/runC 草稿后半段切英文，语言锁定机制在长草稿流里失效（与 R0.5 主跑一致）。

## 对 COMPARISON.md 结论的修正

原结论「qwen3.8-27b 指令遵循有 ~20% 概率全面崩塌，prompt 无法完全控制」需补充限定：
- 崩塌率与 max_tokens 配置交互（16000 下 3/3，32000 下 2/3），非纯模型随机；
- 后续消融轮次的 run 脚本应统一 max_tokens=32000（或按 prompt 变体预估长度设足），否则跨版本比较混入截断噪声。

## 文件

- runA.md / runB.md / runC.md — 原始产出
- 调用方式：urllib 直连 http://localhost:8787/v1/chat/completions（绕系统 proxy），prompt = bench/prompts/variants/R0.5/prompt.txt