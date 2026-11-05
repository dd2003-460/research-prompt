#!/usr/bin/env python3
"""R0.7: Minimal C2 column enforcement (no SELF-CHECK, no appendix)
Delta vs R0.3: +107 bytes (C2 table spec tightened from 2 cols to 4 cols + anti-qualitative)
Hypothesis: C2 stays at 2 or improves; no collapse risk (minimal change).
"""
import json, os, time, urllib.request

PROMPT = open('bench/prompts/variants/R0.7/prompt.txt', encoding='utf-8').read()
API = 'http://localhost:8787/v1/chat/completions'
MODEL = 'qwen3.8-27b'
OUTDIR = 'bench/runs/ablation-L1/prompt-result/R0.7'
os.makedirs(OUTDIR, exist_ok=True)

results = []
for i in range(1, 4):
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': PROMPT}],
        'max_tokens': 32000,
        'temperature': 0.7,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    
    t0 = time.time()
    try:
        with opener.open(req, timeout=900) as r:
            out = json.loads(r.read())
        text = out['choices'][0]['message']['content']
        elapsed = time.time() - t0
        print(f"Run {i}: {len(text)} chars, {elapsed:.0f}s")
        
        with open(f'{OUTDIR}/run{i}.md', 'w') as f:
            f.write(text)
        
        results.append({'run': i, 'chars': len(text), 'time_s': round(elapsed, 1)})
    except Exception as e:
        print(f"Run {i} FAILED: {e}")
        results.append({'run': i, 'error': str(e)})
    
    if i < 3:
        time.sleep(5)

with open(f'{OUTDIR}/run.log', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nAll done. Results: {json.dumps(results)}")
