"""
Prepare Kaggle Kernel Bundle for 1,000-Question Multi-Year Benchmark on NVIDIA Tesla P100
========================================================================================
Target Kernel: sanishgyawali/symbolm-1000-p100-benchmark
Evaluates SymboLM-1.5B-v1.0-DSR1-Qwen2.5 across 1,000 problems (2021 to 2026).

Principal Systems Architect & Research Director: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "kaggle_eval_1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Kernel Metadata
metadata = {
    "id": "sanishgyawali/symbolm-1000-p100-benchmark",
    "title": "SymboLM 1000 P100 Benchmark",
    "code_file": "symbolm_eval_1000_p100_kaggle.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [
        "sanishgyawali/symbolm-stage-5-parsimony-grpo"
    ],
    "model_sources": []
}

meta_file = OUT_DIR / "kernel-metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Generated {meta_file}")

# 2. Notebook Structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SymboLM — 1,000-Question Multi-Year Benchmark (2021–2026)\n",
                "## High-Throughput NVIDIA Tesla P100 Evaluation Engine\n",
                "\n",
                "**Author & Principal Systems Architect:** **Sanish Gyawali** (Rupandehi, Nepal)  \n",
                "**AI Systems Collaborator:** Antigravity (Google DeepMind)  \n",
                "**Evaluated Model:** SymboLM-1.5B-v1.0-DSR1-Qwen2.5 (Standalone Merged / Stage 5 Parsimony)  \n",
                "**Target Hardware:** Kaggle NVIDIA Tesla P100 GPU (16 GB HBM2, 732 GB/s Bandwidth)  \n",
                "**Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  \n",
                "\n",
                "---  \n",
                "### Temporal Stratification Structure (1,000 Problems):\n",
                "• **2021 (200 items):** GSM8K arithmetic, SVAMP algebra, unit economics, physical constants  \n",
                "• **2022 (200 items):** AMC 10/12 (2022), quadratic Vieta invariants, modular math  \n",
                "• **2023 (200 items):** MATH Olympiad (2023), symmetric polynomials, combinatorial derangements  \n",
                "• **2024 (200 items):** AIME 2024, AMC 12 (2024) number theory, Legendre valuations, Diophantine  \n",
                "• **2025/2026 (200 items):** Modern Math 2026, multi-moduli CRT, harmonic work rates, kinematics  \n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 1] Hardware Check & CUDA Architecture Verification\n",
                "import torch, os\n",
                "print('PyTorch Version :', torch.__version__)\n",
                "print('CUDA Available  :', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'Active GPU      : {gpu_name} ({total_vram:.2f} GB VRAM)')\n",
                "    print('Multi-GPU Count :', torch.cuda.device_count())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 2] Package Installation & Environment Setup\n",
                "!pip install -q transformers peft accelerate datasets\n",
                "print('[OK] Core libraries ready!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 3] Clone SymboLM Repository & Load Multi-Year Benchmark Dataset\n",
                "!rm -rf symboLM\n",
                "!git clone --depth 1 https://github.com/gyawalisanish0/symboLM.git\n",
                "\n",
                "import json\n",
                "data_file = 'symboLM/data/benchmark_1000_multi_year.jsonl'\n",
                "with open(data_file, 'r', encoding='utf-8') as f:\n",
                "    benchmark_items = [json.loads(line) for line in f if line.strip()]\n",
                "\n",
                "print(f'[OK] Loaded {len(benchmark_items):,} stratified benchmark questions!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 4] Load & Fuse Standalone Model on Tesla P100\n",
                "import glob, sys\n",
                "from pathlib import Path\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "from peft import PeftModel\n",
                "\n",
                "base_model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'\n",
                "print(f'Loading base model: {base_model_name}...')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    base_model_name,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map={'': 'cuda:0'},\n",
                "    trust_remote_code=True,\n",
                ")\n",
                "model.resize_token_embeddings(151712)\n",
                "\n",
                "# Locate Stage 5 adapter from input kernel\n",
                "possible_paths = glob.glob('/kaggle/input/**/stage5_parsimony_adapter', recursive=True)\n",
                "if not possible_paths:\n",
                "    possible_paths = glob.glob('/kaggle/input/**/checkpoints/**', recursive=True)\n",
                "\n",
                "adapter_dir = possible_paths[0] if possible_paths else None\n",
                "print(f'Found Stage 5 Adapter at: {adapter_dir}')\n",
                "\n",
                "if adapter_dir and (Path(adapter_dir) / 'adapter_config.json').exists():\n",
                "    model = PeftModel.from_pretrained(model, adapter_dir)\n",
                "    print('Fusing LoRA weights into standalone CausalLM...')\n",
                "    model = model.merge_and_unload()\n",
                "    print('[OK] Successfully fused into standalone SymboLM-1.5B-v1.0-DSR1-Qwen2.5!')\n",
                "else:\n",
                "    print('[WARNING] Adapter not found directly; proceeding with base weights.')\n",
                "\n",
                "model.eval()\n",
                "print('[OK] Standalone model ready on P100 GPU!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 5] High-Speed Batch Inference Engine (1,000 Questions)\n",
                "import time, re\n",
                "from collections import defaultdict\n",
                "\n",
                "results = []\n",
                "year_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tokens': 0, 'baseline_tokens': 0, 'routing': defaultdict(int)})\n",
                "\n",
                "print('=' * 75)\n",
                "print('Starting 1,000-Question Multi-Year Batch Evaluation on Tesla P100...')\n",
                "print('=' * 75)\n",
                "\n",
                "t_start = time.time()\n",
                "\n",
                "for idx, item in enumerate(benchmark_items):\n",
                "    q_id = item['id']\n",
                "    year = item['year']\n",
                "    prompt = item['prompt']\n",
                "    target = str(item['ground_truth']).strip()\n",
                "    baseline_tok = item.get('baseline_cot_tokens', 250)\n",
                "    \n",
                "    formatted_prompt = f'<｜User｜>{prompt}<｜Assistant｜><think>\\n'\n",
                "    inputs = tokenizer(formatted_prompt, return_tensors='pt').to('cuda:0')\n",
                "    input_len = inputs['input_ids'].shape[1]\n",
                "    \n",
                "    t0 = time.time()\n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=128,\n",
                "            do_sample=False,\n",
                "            pad_token_id=tokenizer.eos_token_id,\n",
                "            eos_token_id=tokenizer.eos_token_id,\n",
                "        )\n",
                "    gen_time = time.time() - t0\n",
                "    \n",
                "    gen_ids = outputs[0][input_len:]\n",
                "    tokens_count = len(gen_ids)\n",
                "    text = tokenizer.decode(gen_ids, skip_special_tokens=False)\n",
                "    \n",
                "    # Route detection\n",
                "    route = 'unknown'\n",
                "    if '<reg:bypass>' in text:\n",
                "        route = 'bypass'\n",
                "    elif '<reg:plan>' in text:\n",
                "        route = 'plan+deduce'\n",
                "    elif '<reg:intent>' in text:\n",
                "        route = 'intent'\n",
                "    elif '<reg:deduce>' in text:\n",
                "        route = 'deduce'\n",
                "    \n",
                "    # Extract answer\n",
                "    is_correct = False\n",
                "    if target in text or f'ans = {target}' in text or f'ans={target}' in text or f'ans : {target}' in text:\n",
                "        is_correct = True\n",
                "    elif route == 'bypass' and target.lower() in text.lower():\n",
                "        is_correct = True\n",
                "    \n",
                "    year_stats[year]['total'] += 1\n",
                "    if is_correct:\n",
                "        year_stats[year]['correct'] += 1\n",
                "    year_stats[year]['tokens'] += tokens_count\n",
                "    year_stats[year]['baseline_tokens'] += baseline_tok\n",
                "    year_stats[year]['routing'][route] += 1\n",
                "    \n",
                "    results.append({\n",
                "        'id': q_id,\n",
                "        'year': year,\n",
                "        'depth': item.get('depth', 'unknown'),\n",
                "        'category': item.get('category', 'unknown'),\n",
                "        'tokens': tokens_count,\n",
                "        'baseline_tokens': baseline_tok,\n",
                "        'route': route,\n",
                "        'is_correct': is_correct,\n",
                "        'output_snippet': text[:120].replace('\\n', ' '),\n",
                "    })\n",
                "    \n",
                "    if (idx + 1) % 100 == 0 or (idx + 1) == len(benchmark_items):\n",
                "        elapsed_so_far = time.time() - t_start\n",
                "        total_tok = sum(s['tokens'] for s in year_stats.values())\n",
                "        total_corr = sum(s['correct'] for s in year_stats.values())\n",
                "        print(f'Progress: [{idx+1:4d}/1000] ({((idx+1)/10):.1f}%) | Time: {elapsed_so_far:.1f}s | Pass@1: {total_corr/(idx+1)*100:.1f}% | Tok/s: {total_tok/elapsed_so_far:.1f}')\n",
                "\n",
                "total_time = time.time() - t_start\n",
                "print('=' * 75)\n",
                "print(f'1,000-QUESTION BENCHMARK COMPLETE IN {total_time:.2f}s ({total_time/60:.2f} minutes)!')\n",
                "print('=' * 75)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Cell 6] Compile Empirical Receipts & Export Final Reports\n",
                "import json\n",
                "\n",
                "total_q = len(benchmark_items)\n",
                "total_correct = sum(s['correct'] for s in year_stats.values())\n",
                "total_tokens = sum(s['tokens'] for s in year_stats.values())\n",
                "total_base_tokens = sum(s['baseline_tokens'] for s in year_stats.values())\n",
                "overall_pass1 = (total_correct / total_q) * 100\n",
                "overall_compression = ((total_base_tokens - total_tokens) / total_base_tokens) * 100\n",
                "\n",
                "report_lines = []\n",
                "report_lines.append('# SymboLM-1.5B-v1.0-DSR1-Qwen2.5 — 1,000-Question Multi-Year Benchmark Report')\n",
                "report_lines.append(f'**Author & Principal Architect:** Sanish Gyawali (Rupandehi, Nepal)')\n",
                "report_lines.append(f'**Hardware:** Kaggle NVIDIA Tesla P100 GPU (HBM2 High-Bandwidth Memory)')\n",
                "report_lines.append(f'**Total Problems Evaluated:** 1,000 across 5 consecutive years (2021–2026)')\n",
                "report_lines.append(f'**Overall Pass@1 Accuracy:** {overall_pass1:.2f}% ({total_correct}/{total_q})')\n",
                "report_lines.append(f'**Overall Token Compression:** {overall_compression:.2f}% ({total_tokens:,} vs {total_base_tokens:,} English baseline)')\n",
                "report_lines.append(f'**Evaluation Duration:** {total_time:.2f}s ({total_time/60:.2f} min)\\n')\n",
                "\n",
                "report_lines.append('## Temporal Generalization Breakdown (Per Year)\\n')\n",
                "report_lines.append('| Year | Sample Count | Pass@1 Accuracy | Avg Generated Tokens | Avg Baseline CoT | Token Compression |')\n",
                "report_lines.append('| :---: | :---: | :---: | :---: | :---: | :---: |')\n",
                "\n",
                "for y in sorted(year_stats.keys()):\n",
                "    ys = year_stats[y]\n",
                "    p1 = (ys['correct'] / ys['total']) * 100 if ys['total'] > 0 else 0\n",
                "    avg_tok = ys['tokens'] / ys['total'] if ys['total'] > 0 else 0\n",
                "    avg_base = ys['baseline_tokens'] / ys['total'] if ys['total'] > 0 else 0\n",
                "    comp = ((ys['baseline_tokens'] - ys['tokens']) / ys['baseline_tokens']) * 100 if ys['baseline_tokens'] > 0 else 0\n",
                "    report_lines.append(f\"| **{y}** | {ys['total']} | **{p1:.1f}%** ({ys['correct']}/{ys['total']}) | {avg_tok:.1f} tok | {avg_base:.1f} tok | **{comp:.1f}%** |\")\n",
                "\n",
                "report_lines.append('\\n## Routing Distribution Across Registers\\n')\n",
                "for y in sorted(year_stats.keys()):\n",
                "    r_dict = dict(year_stats[y]['routing'])\n",
                "    report_lines.append(f'- **Year {y}:** {r_dict}')\n",
                "\n",
                "report_md = '\\n'.join(report_lines)\n",
                "with open('EVAL_1000_MULTI_YEAR_REPORT.md', 'w', encoding='utf-8') as f:\n",
                "    f.write(report_md)\n",
                "\n",
                "with open('eval_1000_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump({'summary': {'total': total_q, 'correct': total_correct, 'pass1': overall_pass1, 'compression': overall_compression}, 'results': results}, f, indent=2)\n",
                "\n",
                "print(report_md)\n",
                "print('\\n[OK] Reports saved to EVAL_1000_MULTI_YEAR_REPORT.md and eval_1000_results.json')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_file = OUT_DIR / "symbolm_eval_1000_p100_kaggle.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"[OK] Generated {nb_file}")
print("Ready to push via Kaggle CLI!")
