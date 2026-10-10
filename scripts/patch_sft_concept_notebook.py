"""
Patch kaggle_sft_concept/symboLM_sft_concept_kaggle.ipynb to resolve torchao version conflict
and enforce fail-fast verification on adapter generation.
"""

import json
from pathlib import Path

nb_path = Path("kaggle_sft_concept/symboLM_sft_concept_kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch Cell 2: Uninstall torchao and install required packages
nb["cells"][2]["source"] = [
    "# Cell 2: Install required libraries & resolve package conflicts\n",
    "!pip install -q --upgrade pip\n",
    "!pip uninstall -y -q torchao\n",
    "!pip install -q 'transformers>=4.48.0' 'peft>=0.14.0' 'accelerate>=1.2.0' 'bitsandbytes>=0.45.0' 'datasets>=3.2.0'\n"
]

# Patch Cell 7: Enforce strict assertion on adapter output before packaging
nb["cells"][7]["source"] = [
    "# Cell 7: Package and archive final Stage 2.5 policy checkpoints\n",
    "import os\n",
    "adapter_cfg = '/kaggle/working/symboLM/checkpoints/sft_concept_adapter/adapter_config.json'\n",
    "if not os.path.exists(adapter_cfg):\n",
    "    raise FileNotFoundError(f'CRITICAL: {adapter_cfg} not found! SFT training failed!')\n",
    "\n",
    "!tar -czf /kaggle/working/symboLM_sft_concept_policy.tar.gz -C /kaggle/working/symboLM/checkpoints/sft_concept_adapter .\n",
    "print('[COMPLETE] Stage 2.5 Concept SFT complete! Policy weights saved to /kaggle/working/symboLM_sft_concept_policy.tar.gz')\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("[OK] Successfully patched kaggle_sft_concept/symboLM_sft_concept_kaggle.ipynb with torchao conflict resolution!")
