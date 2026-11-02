#!/usr/bin/env python3
"""R0.1 × 3 run 实跑：调本地 qwen3.8-27b @ ninfer :8787 (OpenAI 兼容 API)。

用法: python3 run-r01.py [run1|run2|run3|all]
输出: run{N}.md（同目录）
注意: localhost 流量必须绕系统 proxy（7890），否则 urllib 挂死。
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "..", "..", "..", "..", "prompts", "variants", "R0.1", "prompt.txt"), encoding="utf-8").read()
API = "http://localhost:8787/v1/chat/completions"
MODEL = "qwen3.8-27b"

def call(prompt: str, max_tokens: int = 16000, temperature: float = 0.7) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        # 关键：绕系统 proxy，直连 localhost
        "Proxy-Connection": "none",
    })
    # 显式禁用 proxy handler
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    with opener.open(req, timeout=600) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    print(f"  [{dt:.0f}s] {len(data['choices'][0]['message']['content'])} chars")
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["run1", "run2", "run3"]
    for name in targets:
        out = os.path.join(HERE, f"{name}.md")
        if os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"{name}: 已存在，跳过")
            continue
        print(f"{name}: 调用中...")
        text = call(PROMPT)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# R0.1 {name}\n\n{text}\n")
        print(f"{name}: 落盘 {out}")
