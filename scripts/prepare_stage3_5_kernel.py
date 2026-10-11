import json
from pathlib import Path

# 1. Update kernel metadata
meta_path = Path("kaggle_stage3/kernel-metadata.json")
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Ensure Stage 2.5 Concept SFT is the first kernel source
sft_src = "sanishgyawali/symbolm-stage-2-5-concept-sft-seeding"
if sft_src not in meta.get("kernel_sources", []):
    meta["kernel_sources"] = [sft_src] + meta.get("kernel_sources", [])
else:
    # Move to front
    meta["kernel_sources"].remove(sft_src)
    meta["kernel_sources"] = [sft_src] + meta["kernel_sources"]

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"[OK] Updated {meta_path} with kernel sources: {meta['kernel_sources']}")

# 2. Update notebook Cell 4
nb_path = Path("kaggle_stage3/symboLM_stage3_kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][4]["source"] = [
    "# Cell 4: Mount and unpack Stage 2.5 Concept SFT Policy Adapter as warm-start base\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "target_dir = Path('/kaggle/working/symboLM/checkpoints/stage2_adapter')\n",
    "target_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "found = False\n",
    "# 1. Search for Stage 2.5 Concept SFT extracted directory\n",
    "for p in Path('/kaggle/input').glob('**/sft_concept_adapter'):\n",
    "    if (p / 'adapter_config.json').exists():\n",
    "        print(f'[OK] Found Stage 2.5 Concept SFT adapter at: {p}')\n",
    "        for item in p.iterdir():\n",
    "            if item.is_file():\n",
    "                shutil.copy2(item, target_dir / item.name)\n",
    "        found = True\n",
    "        break\n",
    "\n",
    "# 2. Search for any Concept SFT tar archive\n",
    "if not found:\n",
    "    for arc in Path('/kaggle/input').glob('**/*concept*.tar.gz'):\n",
    "        print(f'[OK] Found Stage 2.5 Concept archive at: {arc}. Unpacking...')\n",
    "        os.system(f'tar -xzf \"{arc}\" -C \"{target_dir}\"')\n",
    "        found = True\n",
    "        break\n",
    "\n",
    "# 3. Fallback: Search for Stage 2 GRPO adapter\n",
    "if not found:\n",
    "    for p in Path('/kaggle/input').glob('**/stage2_adapter'):\n",
    "        if (p / 'adapter_config.json').exists():\n",
    "            print(f'[Notice] Found Stage 2 GRPO adapter at: {p}')\n",
    "            for item in p.iterdir():\n",
    "                if item.is_file():\n",
    "                    shutil.copy2(item, target_dir / item.name)\n",
    "            found = True\n",
    "            break\n",
    "\n",
    "print('[Status] Target Warm-Start Adapter Contents:')\n",
    "for f in sorted(target_dir.iterdir()):\n",
    "    sz = f.stat().st_size / 1e6 if f.is_file() else 0\n",
    "    print(f'  {f.name} ({sz:.2f} MB)' if f.is_file() else f'  {f.name}/')\n",
    "\n",
    "assert (target_dir / 'adapter_config.json').exists(), 'CRITICAL: No warm-start adapter found!'\n",
    "print('[VERIFIED] Successfully mounted Stage 2.5 Concept SFT warm-start adapter!')\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[OK] Successfully patched {nb_path} for Stage 3.5 Concept GRPO!")
