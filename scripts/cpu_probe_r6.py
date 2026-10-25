"""CPU-only probe for the r6 SFT adapter.

This is a diagnostic, not a benchmark. It loads the verified r6 adapter on CPU,
builds the exact v3 register prompt, and prints one raw response per prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.strict_scorer import score_declared_answer
from symbolic.prompts import REGISTER_SYSTEM_PROMPT, build_chat_prompt

BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
ADAPTER = ROOT / "checkpoints" / "sft_v3_r6" / "adapter"
RECEIPT = ROOT / "results" / "r6_closeout" / "CPU_PROBE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

PROMPTS = [
    {
        "id": "MY2021_002",
        "kind": "reasoning",
        "prompt": (
            "A delivery van travels for 2 hours at 50 km/h and then 3 hours at "
            "65 km/h. What is the total distance traveled?"
        ),
        "ground_truth": "295",
        "max_new_tokens": 256,
    },
    {
        "id": "MY2021_164",
        "kind": "atomic",
        "prompt": "What is the atomic number of Iron?",
        "ground_truth": "26",
        "max_new_tokens": 96,
    },
]


def main() -> None:
    torch.set_num_threads(8)
    adapter_sha256 = sha256_file(ADAPTER / "adapter_model.safetensors")
    print(f"[cpu] torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)
    print(f"[cpu] adapter_sha256={adapter_sha256}", flush=True)
    print(f"[cpu] loading tokenizer from {ADAPTER}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER), trust_remote_code=True, local_files_only=True
    )

    print(f"[cpu] loading base {BASE_ID} @ {BASE_REVISION} in bfloat16", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))

    print("[cpu] loading r6 PEFT adapter", flush=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()

    results = []
    for item in PROMPTS:
        prompt = build_chat_prompt(
            tokenizer, REGISTER_SYSTEM_PROMPT, item["prompt"], open_think=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        print("\n" + "=" * 72, flush=True)
        print(f"[probe] {item['id']} ({item['kind']})", flush=True)
        print(f"[probe] prompt_tokens={prompt_tokens}", flush=True)
        print(f"[probe] user={item['prompt']}", flush=True)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=item["max_new_tokens"],
                do_sample=False,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        generated_ids = out[0][prompt_tokens:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos_emitted = bool(
            len(generated_ids) > 0
            and int(generated_ids[-1].item()) == int(tokenizer.eos_token_id)
        )
        strict = score_declared_answer(
            generated,
            arm="C",
            ground_truth=item["ground_truth"],
            tokens_emitted=len(generated_ids),
            budget=item["max_new_tokens"],
            eos_emitted=eos_emitted,
            expected_kind="bypass" if item["kind"] == "atomic" else "reasoning",
        )

        print(f"[probe] generated_tokens={len(generated_ids)} elapsed={elapsed:.1f}s", flush=True)
        print(f"[probe] eos={eos_emitted} correct={strict['is_correct']} "
              f"format={strict['format_valid']} ({strict['format_reason']})", flush=True)
        print(f"[probe] extracted={strict['answer']!r} gt={item['ground_truth']!r}", flush=True)
        print("--- raw generation ---", flush=True)
        print(generated, flush=True)
        print("--- end ---", flush=True)

        results.append(
            {
                "problem_id": item["id"],
                "kind": item["kind"],
                "prompt": item["prompt"],
                "ground_truth": item["ground_truth"],
                "max_new_tokens": item["max_new_tokens"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated_ids),
                "elapsed_sec": round(elapsed, 3),
                "eos_emitted": eos_emitted,
                "generated_text": generated,
                "strict": strict,
            }
        )

    receipt = {
        "schema": "symboLM.cpu_probe.v1",
        "status": "diagnostic_only_not_a_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "base_model": {"id": BASE_ID, "revision": BASE_REVISION},
        "adapter": {
            "path": str(ADAPTER),
            "adapter_model_sha256": adapter_sha256,
        },
        "generation": {
            "do_sample": False,
            "repetition_penalty": 1.15,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "results": results,
        "note": (
            "Two CPU probes only. This verifies that the r6 adapter loads and "
            "emits valid v3 terminal traces on CPU; it is not an accuracy, "
            "token-efficiency, latency, or mobile result."
        ),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n[cpu] receipt written -> {RECEIPT}", flush=True)


if __name__ == "__main__":
    main()
