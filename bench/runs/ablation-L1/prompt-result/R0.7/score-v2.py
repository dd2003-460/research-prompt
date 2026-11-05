#!/usr/bin/env python3
"""R0.7 评分器：机械预检 (S3/C2) + lead review 8 维。
R0.7 = R0.3 + C2 四列强化 (Prefill(s)/Decode_tok_s/TPS/$/任务) + 禁定性描述。
无 SELF-CHECK 契约（R0.4 教训），无附录（T34 教训）。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

def mechanical(text):
    secs = {h: (h in text) for h in [
        "## 0. 研究问题定义","## 1. 研究路径","## 2. 矛盾与张力",
        "## 3. 计算与验证","## 4. 结论与置信度"
    ]}
    nodes = re.findall(r"### R-\d+(?:\.\d+)*\b", text)
    fields = {f: len(re.findall(rf"^{re.escape(f)}:\s", text, re.M)) 
              for f in ["PARENT","NOW","GOAL","GAP","IN","OUT","KIND","RES","SHARE","BOUND","TEST"]}
    selfcheck = len(re.findall(r"SELF-CHECK:", text))
    cost_factors = [f for f in ["折旧","能耗","运维","残值"] if re.search(rf"{f}:.*\[E", text)]
    placeholders = len(re.findall(r"\$X\b|\$Y\b|待定", text))
    
    # C2 signals (R0.7: 4 columns required)
    has_prefill_col = bool(re.search(r"[Pp]refill\s*\(?\s*s\s*\)?", text))
    has_decode_col = bool(re.search(r"[Dd]ecode\s*[_\s]?tok\s*[_\s]?s|[Dd]ecode\s*\(?\s*s\s*\)?", text))
    has_tps_col = bool(re.search(r"TPS|tokens?/?s|tok/s", text, re.I))
    has_cost_col = bool(re.search(r"\$/\s*(?:任务|task)|成本.*USD|USD.*成本", text))
    time_nums = len(re.findall(r"\d+\.?\d*\s*(?:s|ms)\b", text))
    drift = bool(re.search(
        r"(compute-bound.{0,120}bandwidth-bound|bandwidth-bound.{0,120}compute-bound|瓶颈.{0,40}(漂移|迁移))", text))
    cards = set(re.findall(r"\b(A6000|A100|H100|MI300X|L40S|B200|4090|5090|M2 Ultra|M3 Ultra)\b", text))
    
    # S3: no SELF-CHECK contract in R0.7 — use COUPLE/重叠 signal only
    if ("COUPLE" in text) or ("重叠" in text):
        s3, s3r = 1, "有重叠声明但无逐节点 SELF-CHECK 行（R0.7 无此契约）"
    else:
        s3, s3r = 0, "无 COUPLE/重叠声明"
    
    # C2: R0.7 requires 4 cols + numeric + drift
    four_cols = has_prefill_col and has_decode_col and has_tps_col and has_cost_col
    two_cols = has_prefill_col and has_decode_col
    if four_cols and time_nums >= 4 and drift and len(cards) >= 3:
        c2, c2r = 2, f"四列全(P/D/TPS/$) + {time_nums}时间数值 + 漂移 + {len(cards)}卡"
    elif two_cols and time_nums >= 4 and drift and len(cards) >= 3:
        c2, c2r = 2, f"双列(P/D) + {time_nums}时间数值 + 漂移 + {len(cards)}卡 (TPS/$缺失但不影响C2核心)"
    elif (has_prefill_col or has_decode_col) and time_nums >= 2 and len(cards) >= 2:
        c2, c2r = 1, f"部分达标(P={has_prefill_col},D={has_decode_col},TPS={has_tps_col},$={has_cost_col},数值={time_nums},漂移={drift})"
    elif len(cards) >= 2 and time_nums >= 2:
        c2, c2r = 1, f"有逐卡数值但缺表列(卡={sorted(cards)})"
    else:
        c2, c2r = 0, f"缺逐卡时间列(P={has_prefill_col},D={has_decode_col},卡={sorted(cards)})"
    
    out = {
        "chars": len(text), "five_sections": sum(secs.values()), "section_detail": secs,
        "node_blocks": len(nodes), "field_coverage": fields, "selfcheck_lines": selfcheck,
        "cost_factors": cost_factors, "placeholders": placeholders,
        "S3": {"score": s3, "reason": s3r},
        "C2": {"score": c2, "reason": c2r,
               "signals": {"prefill_col": has_prefill_col, "decode_col": has_decode_col,
                           "tps_col": has_tps_col, "cost_col": has_cost_col,
                           "time_nums": time_nums, "drift": drift, "cards": sorted(cards)}},
        "lead_review_needed": ["S1","S2","S4","S5","C1","C3","C4","C5"],
    }
    if out["five_sections"] == 0:
        out["verdict"] = "FAIL — 草稿区直接结束，未输出五节结构"
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
        "rubric": "SCORING-RUBRIC-v2.md (20分制) — R0.7 机械预检版",
        "scorer": "score-v2.py 机械检查 + lead-agent (qwen3.8-27b) 复核",
        "runs": runs,
        "mean": None, "sigma": None,
        "notes": "R0.7 = R0.3 + C2四列强化(+107B)。无SELF-CHECK，无附录。",
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
