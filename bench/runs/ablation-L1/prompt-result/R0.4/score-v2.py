#!/usr/bin/env python3
"""R0.4 评分器：读 run{N}.md，按 rubric v2 判据机械检查 + 输出 score-v2.json。

用法: python3 score-v2.py
判据档位与 R0~R0.3 对齐（lead-agent qwen3.8-27b 同族判官，分数作相对比较）：
S1-S5 结构维度（各 0/1/2），C1-C5 内容维度（各 0/1/2），满分 20。
S3/C2 自动定档；其余 8 维输出信号供 lead 复核。
run 作废判定：五节标题 0/5 → FAIL（草稿区截断或格式崩溃）。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

def mechanical(text):
    secs = {h: (h in text) for h in ["## 0. 研究问题定义","## 1. 研究路径","## 2. 矛盾与张力","## 3. 计算与验证","## 4. 结论与置信度"]}
    nodes = re.findall(r"### R-\d+(?:\.\d+)*\b", text)
    fields = {f: len(re.findall(rf"^{re.escape(f)}:\s", text, re.M)) for f in ["PARENT","NOW","GOAL","GAP","IN","OUT","KIND","RES","SHARE","BOUND","TEST"]}
    selfcheck = len(re.findall(r"SELF-CHECK:", text))
    cost_factors = [f for f in ["折旧","能耗","运维","残值"] if re.search(rf"{f}:", text)]
    placeholders = len(re.findall(r"\$X\b|\$Y\b|待定", text))
    has_prefill_col = bool(re.search(r"[Pp]refill\s*\(?s\)?", text))
    has_decode_col = bool(re.search(r"[Dd]ecode\s*\(?s\)?", text))
    time_nums = len(re.findall(r"\d+\.?\d*\s*(?:s|ms)\b", text))
    drift = bool(re.search(r"(compute-bound.{0,120}bandwidth-bound|bandwidth-bound.{0,120}compute-bound|瓶颈.{0,40}(漂移|迁移))", text))
    cards = set(re.findall(r"\b(A6000|A100|H100|MI300X|L40S|B200|4090|5090|M2 Ultra|M3 Ultra)\b", text))
    # S3：R0.4 契约 —— SELF-CHECK 行覆盖节点；无行时回退 COUPLE/重叠声明档
    if selfcheck >= max(len(nodes), 3) and len(nodes) >= 3:
        s3, s3r = 2, f"SELF-CHECK {selfcheck} 行覆盖 {len(nodes)} 节点（全覆盖）"
    elif selfcheck >= 2:
        s3, s3r = 1, f"SELF-CHECK {selfcheck} 行但未全覆盖 {len(nodes)} 节点"
    elif ("COUPLE" in text) or ("重叠" in text):
        s3, s3r = 1, "有重叠声明但无逐节点 SELF-CHECK 行"
    else:
        s3, s3r = 0, "无 COUPLE/SELF-CHECK"
    # C2：逐卡表 Prefill(s)/Decode(s) 双列 + 数值 + 漂移结论
    if has_prefill_col and has_decode_col and time_nums >= 4 and drift and len(cards) >= 3:
        c2, c2r = 2, f"Prefill/Decode 双列 + {time_nums} 时间数值 + 漂移结论 + {len(cards)} 卡"
    elif (has_prefill_col or has_decode_col) and time_nums >= 2 and len(cards) >= 2:
        c2, c2r = 1, f"部分达标（prefill列={has_prefill_col}, decode列={has_decode_col}, 数值={time_nums}, 漂移={drift}）"
    elif len(cards) >= 2 and time_nums >= 2:
        c2, c2r = 1, f"有逐卡数值但缺 Prefill(s)/Decode(s) 表列（只有 TCO 对比表），按 R0.4 契约不得满分"
    else:
        c2, c2r = 0, f"缺逐卡时间列（prefill列={has_prefill_col}, decode列={has_decode_col}, 卡={sorted(cards)}）"
    out = {
        "chars": len(text), "five_sections": sum(secs.values()), "section_detail": secs,
        "node_blocks": len(nodes), "field_coverage": fields, "selfcheck_lines": selfcheck,
        "cost_factors": cost_factors, "placeholders": placeholders,
        "S3": {"score": s3, "reason": s3r},
        "C2": {"score": c2, "reason": c2r,
               "signals": {"prefill_col": has_prefill_col, "decode_col": has_decode_col,
                           "time_nums": time_nums, "drift": drift, "cards": sorted(cards)}},
        "lead_review_needed": ["S1","S2","S4","S5","C1","C3","C4","C5"],
    }
    if out["five_sections"] == 0:
        out["verdict"] = "FAIL — 草稿区直接结束，未输出五节结构（max_tokens 截断或格式崩溃）"
    return out

def main():
    runs = {}
    for n in (1, 2, 3):
        p = os.path.join(HERE, f"run{n}.md")
        if not os.path.exists(p):
            runs[f"run{n}"] = {"missing": True}
        else:
            runs[f"run{n}"] = mechanical(open(p, encoding="utf-8").read())
    out = {
        "rubric": "SCORING-RUBRIC-v2.md (20 分制) — 机械预检版：S3/C2 自动定档，其余 8 维待 lead 复核",
        "scorer": "score-v2.py 机械检查 + lead-agent (qwen3.8-27b) 复核；判据档位与 R0~R0.3 对齐",
        "runs": runs,
        "mean": None, "sigma": None,
        "notes": "mean/σ 由 lead 在 8 维复核后补全；作废 run 不计入 mean。",
    }
    path = os.path.join(HERE, "score-v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {}
    for k, v in runs.items():
        if v.get("missing"):
            summary[k] = {"missing": True}
        else:
            summary[k] = {kk: vv.get("score") for kk, vv in v.items() if isinstance(vv, dict) and "score" in vv}
            summary[k]["five"] = v.get("five_sections")
            if "verdict" in v: summary[k]["verdict"] = v["verdict"]
    print(json.dumps({"written": path, **summary}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
