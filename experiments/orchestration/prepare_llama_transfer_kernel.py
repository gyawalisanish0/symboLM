"""
Prepare Kaggle Kernel Bundle for Cross-Architecture Transfer: SymboLM on LLaMA (SmolLM2-1.7B)
=============================================================================================
Creates:
  - experiments/kaggle/cross_family_llama/kernel-metadata.json
  - experiments/kaggle/cross_family_llama/symbolm_llama_3_2_transfer.ipynb

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT_DIR / "experiments" / "kaggle" / "cross_family_llama"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Kernel Metadata
metadata = {
    "id": "sanishgyawali/symbolm-cross-family-transfer-meta-llama-3-2-1b",
    "title": "SymboLM Cross-Family Transfer: LLaMA 1.7B Architecture",
    "code_file": "symbolm_llama_3_2_transfer.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
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
                "# SymboLM — Cross-Family Architecture Transfer: LLaMA Architecture (1.7B)\n",
                "\n",
                "**Author & Principal Architect:** Sanish Gyawali (Rupandehi, Nepal)  \n",
                "**AI Systems Exoskeleton:** Antigravity (Google DeepMind)  \n",
                "**Foundation Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Pure LLaMA Architecture, 1.71B Parameters)  \n",
                "**Hardware Target:** Kaggle NVIDIA Tesla T4 GPU (16 GB VRAM)  \n",
                "**Repository:** [https://github.com/gyawalisanish0/symboLM](https://github.com/gyawalisanish0/symboLM)  \n",
                "\n",
                "---  \n",
                "### Core Experiment Objectives:\n",
                "1. **Cross-Architecture Universality:** Prove that SymboLM's Register v2.0 cognitive architecture is foundation-agnostic by transferring from Alibaba Qwen-2.5 (151k vocab) to the LLaMA architecture (49k BPE vocab).\n",
                "2. **Tokenizer Prior Ablation:** Verify that LLaMA cleanly tokenizes and enforces `<reg:plan>`, `<reg:deduce>`, `<reg:verify>`, `<reg:bypass>`, and `<reg:intent>` delimiters with 0% drift.\n",
                "3. **Empirical Deductive Compression:** Verify whether the LLaMA architecture achieves ultra-dense derivations (< 35 tokens) on Olympiad invariants without circular overthinking."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Environment & GPU Verification\n",
                "import os\n",
                "import sys\n",
                "import torch\n",
                "\n",
                "print('Python version:', sys.version)\n",
                "print('Torch version :', torch.__version__)\n",
                "if torch.cuda.is_available():\n",
                "    print(f'[OK] NVIDIA CUDA detected! Found {torch.cuda.device_count()} GPU(s):')\n",
                "    for i in range(torch.cuda.device_count()):\n",
                "        mem = torch.cuda.get_device_properties(i).total_memory / 1e9\n",
                "        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB VRAM)')\n",
                "else:\n",
                "    print('[Warning] No CUDA GPU found!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Install Required Libraries & Fix Kaggle Torchao\n",
                "!pip uninstall -y -q torchao\n",
                "!pip install -q --upgrade pip\n",
                "!pip install -q transformers peft trl accelerate sympy datasets huggingface_hub bitsandbytes\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Clone SymboLM Repository\n",
                "!rm -rf /kaggle/working/symboLM\n",
                "!git clone https://github.com/gyawalisanish0/symboLM.git /kaggle/working/symboLM\n",
                "print('[OK] Successfully cloned latest SymboLM repository!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Load LLaMA Architecture Model (SmolLM2-1.7B-Instruct)\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
                "from peft import LoraConfig, get_peft_model, TaskType\n",
                "\n",
                "MODEL_ID = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'\n",
                "print(f'Loading LLaMA-architecture backbone: {MODEL_ID}...')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    MODEL_ID,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map='auto',\n",
                "    trust_remote_code=True\n",
                ")\n",
                "model.config.pad_token_id = tokenizer.pad_token_id\n",
                "\n",
                "lora_config = LoraConfig(\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    lora_dropout=0.05,\n",
                "    bias='none',\n",
                "    task_type=TaskType.CAUSAL_LM,\n",
                "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']\n",
                ")\n",
                "model = get_peft_model(model, lora_config)\n",
                "model.print_trainable_parameters()\n",
                "print('[OK] LLaMA-1.7B initialized with LoRA!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Prepare Dataset with ChatML Formatting\n",
                "import json\n",
                "from datasets import Dataset\n",
                "\n",
                "dataset_file = '/kaggle/working/symboLM/data/concept_math_curriculum.jsonl'\n",
                "samples = []\n",
                "with open(dataset_file, 'r', encoding='utf-8') as f:\n",
                "    for line in f:\n",
                "        item = json.loads(line)\n",
                "        prompt = item.get('prompt', '').strip()\n",
                "        target = item.get('full_target', '').strip()\n",
                "        if prompt and target:\n",
                "            text = f'<|im_start|>system\\nYou are SymboLM, a biomimetic multi-register reasoning engine.<|im_end|>\\n' \\\n",
                "                   f'<|im_start|>user\\n{prompt}<|im_end|>\\n' \\\n",
                "                   f'<|im_start|>assistant\\n{target}<|im_end|>'\n",
                "            samples.append({'text': text})\n",
                "\n",
                "print(f'[OK] Prepared {len(samples)} training samples from concept_math_curriculum.jsonl')\n",
                "train_dataset = Dataset.from_list(samples)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Train LoRA SFT on LLaMA Architecture\n",
                "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n",
                "\n",
                "def tokenize_fn(examples):\n",
                "    return tokenizer(examples['text'], truncation=True, max_length=512)\n",
                "\n",
                "tokenized_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])\n",
                "\n",
                "training_args = TrainingArguments(\n",
                "    output_dir='/kaggle/working/output_llama',\n",
                "    num_train_epochs=1,\n",
                "    per_device_train_batch_size=4,\n",
                "    gradient_accumulation_steps=4,\n",
                "    learning_rate=2e-4,\n",
                "    lr_scheduler_type='cosine',\n",
                "    warmup_steps=10,\n",
                "    logging_steps=10,\n",
                "    save_strategy='no',\n",
                "    fp16=True,\n",
                "    report_to='none'\n",
                ")\n",
                "\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=tokenized_dataset,\n",
                "    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)\n",
                ")\n",
                "\n",
                "print('Starting Cross-Family SFT Training on LLaMA 1.7B...')\n",
                "trainer.train()\n",
                "print('[OK] Training Complete!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Save Trained LLaMA Adapter & Create Archive\n",
                "import os\n",
                "\n",
                "adapter_dir = '/kaggle/working/symbolm_llama_1_7b_adapter'\n",
                "model.save_pretrained(adapter_dir)\n",
                "tokenizer.save_pretrained(adapter_dir)\n",
                "print(f'[OK] Adapter saved to {adapter_dir}')\n",
                "\n",
                "os.system(f'tar -czf /kaggle/working/symbolm_llama_1_7b_adapter.tar.gz -C {adapter_dir} .')\n",
                "print('[OK] Created archive: /kaggle/working/symbolm_llama_1_7b_adapter.tar.gz')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Comprehensive Cross-Family Verification Suite\n",
                "model.eval()\n",
                "\n",
                "TEST_PROMPTS = [\n",
                "    {\n",
                "        'domain': 'Δ0 Factual Bypass',\n",
                "        'prompt': 'What is the chemical symbol for potassium?',\n",
                "        'expected': '<reg:bypass>K</reg:bypass>'\n",
                "    },\n",
                "    {\n",
                "        'domain': 'Δ1 Polynomial Invariant',\n",
                "        'prompt': 'If x + 1/x = 6, find the value of x^2 + 1/x^2.',\n",
                "        'expected': '34'\n",
                "    },\n",
                "    {\n",
                "        'domain': 'Δ2 Legendre Prime Valuations',\n",
                "        'prompt': 'Find the highest power of 3 that divides 200!.',\n",
                "        'expected': '97'\n",
                "    },\n",
                "    {\n",
                "        'domain': 'Δ3 Chinese Remainder Theorem',\n",
                "        'prompt': 'Find the smallest positive integer x such that x = 4 (mod 7) and x = 9 (mod 11).',\n",
                "        'expected': '53'\n",
                "    },\n",
                "    {\n",
                "        'domain': 'Strategic Intent & Pragmatics',\n",
                "        'prompt': 'How do you respond politely to a demanding client?',\n",
                "        'expected': '<reg:intent>'\n",
                "    }\n",
                "]\n",
                "\n",
                "print('=' * 80)\n",
                "print('CROSS-FAMILY ARCHITECTURE TRANSFER EVALUATION: LLAMA 1.7B')\n",
                "print('=' * 80)\n",
                "\n",
                "eval_results = []\n",
                "total_tokens = 0\n",
                "\n",
                "for test in TEST_PROMPTS:\n",
                "    domain = test['domain']\n",
                "    p = test['prompt']\n",
                "    formatted_input = f'<|im_start|>system\\nYou are SymboLM, a biomimetic multi-register reasoning engine.<|im_end|>\\n' \\\n",
                "                      f'<|im_start|>user\\n{p}<|im_end|>\\n' \\\n",
                "                      f'<|im_start|>assistant\\n'\n",
                "    inputs = tokenizer(formatted_input, return_tensors='pt').to(model.device)\n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=96,\n",
                "            temperature=0.2,\n",
                "            do_sample=True,\n",
                "            pad_token_id=tokenizer.eos_token_id\n",
                "        )\n",
                "    \n",
                "    gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]\n",
                "    tok_count = len(gen_tokens)\n",
                "    total_tokens += tok_count\n",
                "    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()\n",
                "    \n",
                "    # Check register preservation\n",
                "    has_plan = '<reg:plan>' in gen_text\n",
                "    has_deduce = '<reg:deduce>' in gen_text\n",
                "    has_bypass = '<reg:bypass>' in gen_text\n",
                "    has_intent = '<reg:intent>' in gen_text\n",
                "    \n",
                "    print(f'\\n[Domain] {domain}')\n",
                "    print(f'Prompt  : {p}')\n",
                "    print(f'Tokens  : {tok_count} generated')\n",
                "    print(f'Response: {gen_text}')\n",
                "    print(f'Registers Detected: plan={has_plan}, deduce={has_deduce}, bypass={has_bypass}, intent={has_intent}')\n",
                "    \n",
                "    eval_results.append({\n",
                "        'domain': domain,\n",
                "        'prompt': p,\n",
                "        'tokens': tok_count,\n",
                "        'response': gen_text,\n",
                "        'registers': {'plan': has_plan, 'deduce': has_deduce, 'bypass': has_bypass, 'intent': has_intent}\n",
                "    })\n",
                "\n",
                "print('\\n' + '=' * 80)\n",
                "print('SUMMARY OF CROSS-FAMILY TRANSFER RESULTS:')\n",
                "print(f'Total Evaluated: {len(TEST_PROMPTS)} problems')\n",
                "print(f'Total Generated Tokens: {total_tokens} (Average: {total_tokens / len(TEST_PROMPTS):.1f} tok/problem)')\n",
                "print('=' * 80)\n",
                "\n",
                "with open('/kaggle/working/llama_1_7b_transfer_results.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump({'model': MODEL_ID, 'results': eval_results}, f, indent=2)\n",
                "print('[OK] Saved evaluation receipts to /kaggle/working/llama_1_7b_transfer_results.json')\n"
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
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_file = OUT_DIR / "symbolm_llama_3_2_transfer.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"[OK] Generated {nb_file}")
