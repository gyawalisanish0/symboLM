import json
from pathlib import Path

nb_path = Path("kaggle_grpo/symboLM_grpo_kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 1: Universal hardware check
nb["cells"][1]["source"] = [
    "# Cell 1: Environment & Hardware Verification\n",
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
    "    print('[Notice] No CUDA GPU found. Checking PyTorch/XLA TPU...')\n",
    "    try:\n",
    "        import torch_xla\n",
    "        import torch_xla.core.xla_model as xm\n",
    "        print('[OK] PyTorch/XLA detected! Active TPU device:', xm.xla_device())\n",
    "    except Exception as e:\n",
    "        print('[Notice] TPU/XLA check:', e)\n"
]

# Update Cell 2: Dependencies and conflict resolution
nb["cells"][2]["source"] = [
    "# Cell 2: Install required libraries & resolve package conflicts\n",
    "!pip install -q --upgrade pip\n",
    "!pip uninstall -y -q torchao\n",
    "!pip install -q transformers peft trl accelerate sympy datasets bitsandbytes\n"
]

# Update Cell 4: Dynamic SFT Adapter & Tokenizer Mounting
nb["cells"][4]["source"] = [
    "# Cell 4: Mount and unpack Stage 1 SFT Adapter & Tokenizer from Kaggle kernel source\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "target_dir = Path('/kaggle/working/symboLM/checkpoints/sft_adapter')\n",
    "target_dir.mkdir(parents=True, exist_ok=True)\n",
    "tok_target = Path('/kaggle/working/symboLM/checkpoints/tokenizer_extended')\n",
    "tok_target.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "found = False\n",
    "# 1. Search for existing extracted sft_adapter directory containing adapter_config.json\n",
    "for p in Path('/kaggle/input').glob('**/sft_adapter'):\n",
    "    if (p / 'adapter_config.json').exists():\n",
    "        print(f'[OK] Found extracted SFT adapter directory at: {p}')\n",
    "        for item in p.iterdir():\n",
    "            if item.is_file():\n",
    "                shutil.copy2(item, target_dir / item.name)\n",
    "        found = True\n",
    "        break\n",
    "\n",
    "# 2. If not found, search for any sft*.tar.gz archive\n",
    "if not found:\n",
    "    for arc in Path('/kaggle/input').glob('**/*sft*.tar.gz'):\n",
    "        print(f'[OK] Found SFT archive at: {arc}. Unpacking...')\n",
    "        os.system(f'tar -xzf \"{arc}\" -C \"{target_dir}\"')\n",
    "        found = True\n",
    "        break\n",
    "\n",
    "# 3. Copy tokenizer_extended if present\n",
    "for tp in Path('/kaggle/input').glob('**/tokenizer_extended'):\n",
    "    if (tp / 'tokenizer.json').exists():\n",
    "        print(f'[OK] Found tokenizer_extended directory at: {tp}')\n",
    "        for item in tp.iterdir():\n",
    "            if item.is_file():\n",
    "                shutil.copy2(item, tok_target / item.name)\n",
    "        break\n",
    "\n",
    "print('[Status] Target SFT Adapter Directory Contents:')\n",
    "for f in sorted(target_dir.iterdir()):\n",
    "    sz = f.stat().st_size / 1e6 if f.is_file() else 0\n",
    "    print(f'  {f.name} ({sz:.2f} MB)' if f.is_file() else f'  {f.name}/')\n"
]

# Update Cell 6: Hardware-adaptive launch
nb["cells"][6]["source"] = [
    "# Cell 6: Launch Stage 2 GRPO Policy Optimization on single GPU\n",
    "!CUDA_VISIBLE_DEVICES=0 python -m training.grpo_train \\\n",
    "    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\\n",
    "    --sft_adapter ./checkpoints/sft_adapter \\\n",
    "    --output_dir ./checkpoints/grpo_adapter \\\n",
    "    --data_dir ./data \\\n",
    "    --num_generations 4 \\\n",
    "    --learning_rate 0.000005 \\\n",
    "    --max_steps 250\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[OK] kaggle_grpo/symboLM_grpo_kaggle.ipynb updated successfully!")
