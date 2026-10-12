"""
SymboLM Local Hardware Benchmark (AMD Ryzen 7 5700G)
===================================================
Benchmarks 4-Bit (Q4_0) vs 8-Bit (Q8_0) GGUF models running
locally on CPU AVX2 via Ollama runner.

Principal Systems Architect: Sanish Gyawali (Rupandehi, Nepal)
"""

import json
import time
import urllib.request
import sys

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROMPTS = [
    {
        "domain": "Δ0 Factual Bypass",
        "prompt": "What is the chemical symbol for potassium?"
    },
    {
        "domain": "Δ1 Polynomial Invariant",
        "prompt": "If x + 1/x = 6, find the value of x^2 + 1/x^2."
    },
    {
        "domain": "Δ2 Legendre Valuations",
        "prompt": "Find the highest power of 3 that divides 200!."
    },
    {
        "domain": "Δ3 Chinese Remainder Theorem",
        "prompt": "Find the smallest positive integer x such that x = 4 (mod 7) and x = 9 (mod 11)."
    },
    {
        "domain": "Social Pragmatics",
        "prompt": "How do you respond politely to a demanding client?"
    }
]

def query_model(model_name: str, prompt: str):
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_predict": 64
        }
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        res = urllib.request.urlopen(req)
        raw = res.read().decode("utf-8")
        resp = json.loads(raw)
        wall_time = time.time() - t0
        return resp, wall_time
    except Exception as e:
        print(f"Error querying {model_name}: {e}")
        return None, 0

def benchmark_model(model_name: str):
    print("\n" + "=" * 75)
    print(f"BENCHMARKING MODEL: {model_name} on AMD Ryzen 7 5700G")
    print("=" * 75)

    # Warmup
    print("Running warmup query...")
    query_model(model_name, "Hello")

    total_tokens = 0
    total_eval_time = 0
    total_prompt_tokens = 0
    total_prompt_time = 0
    results = []

    for item in PROMPTS:
        domain = item["domain"]
        prompt = item["prompt"]
        print(f"\n[Test] {domain}")
        print(f"Prompt   : {prompt}")

        resp, wall_time = query_model(model_name, prompt)
        if not resp:
            continue

        response_text = resp.get("response", "").strip()
        eval_count = resp.get("eval_count", 0)
        eval_dur_s = resp.get("eval_duration", 1) / 1e9
        prompt_count = resp.get("prompt_eval_count", 0)
        prompt_dur_s = resp.get("prompt_eval_duration", 1) / 1e9
        gen_speed = eval_count / eval_dur_s if eval_dur_s > 0 else 0
        prompt_speed = prompt_count / prompt_dur_s if prompt_dur_s > 0 else 0

        total_tokens += eval_count
        total_eval_time += eval_dur_s
        total_prompt_tokens += prompt_count
        total_prompt_time += prompt_dur_s

        print(f"Response : {response_text[:120]}...")
        print(f"Tokens   : {eval_count} generated | {prompt_count} prompt tokens")
        print(f"Speed    : {gen_speed:.2f} tok/s (Eval duration: {eval_dur_s:.3f}s)")
        print(f"Prompt Tx: {prompt_speed:.2f} tok/s (Prompt eval: {prompt_dur_s*1000:.1f}ms)")

        results.append({
            "domain": domain,
            "response": response_text,
            "tokens": eval_count,
            "speed": gen_speed,
            "latency": eval_dur_s
        })

    avg_speed = total_tokens / total_eval_time if total_eval_time > 0 else 0
    avg_prompt_speed = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0

    print("-" * 75)
    print(f"SUMMARY FOR {model_name}:")
    print(f"  • Total Generated Tokens : {total_tokens}")
    print(f"  • Average Generation Speed: {avg_speed:.2f} tokens/second")
    print(f"  • Average Prompt Speed    : {avg_prompt_speed:.2f} tokens/second")
    print(f"  • Total Wallclock Time    : {total_eval_time:.2f} seconds")
    print("-" * 75)

    return {
        "model": model_name,
        "avg_speed": avg_speed,
        "avg_prompt_speed": avg_prompt_speed,
        "results": results
    }

def main():
    print("=" * 75)
    print("SymboLM Local PC Performance Benchmark Suite")
    print("Hardware: AMD Ryzen 7 5700G (8 Cores, 16 Threads, Zen 3 AVX2)")
    print("Host RAM: 14 GB DDR4")
    print("Runtime : Ollama v0.34.2 (Native C++ llama.cpp Engine)")
    print("=" * 75)

    # 1. Benchmark 4-Bit Model
    q4_res = benchmark_model("symbolm:4bit")

    # 2. Benchmark 8-Bit Model for comparison
    q8_res = benchmark_model("symbolm:q8")

    print("\n" + "=" * 75)
    print("FINAL COMPARISON: 4-BIT vs 8-BIT SPEEDUP ON YOUR PC")
    print("=" * 75)
    print(f"4-Bit (Q4_0) Size: 1.0 GB | Speed: {q4_res['avg_speed']:.2f} tok/s")
    print(f"8-Bit (Q8_0) Size: 1.9 GB | Speed: {q8_res['avg_speed']:.2f} tok/s")
    speedup = (q4_res['avg_speed'] / q8_res['avg_speed'] - 1.0) * 100 if q8_res['avg_speed'] > 0 else 0
    print(f"Speed Gain with 4-Bit: {speedup:+.1f}% faster!")
    print("=" * 75)

if __name__ == "__main__":
    main()
