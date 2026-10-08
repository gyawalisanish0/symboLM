#!/usr/bin/env bash
# SymboLM - Lightning AI Studio Bootstrap Script
# Author: Sanish Gyawali

set -e

echo "=========================================================="
echo "⚡ SymboLM Stage 1 SFT Setup on Lightning AI"
echo "=========================================================="

# 1. Update / clone repository
cd "$HOME"
if [ ! -d "symboLM" ]; then
    echo "Cloning symboLM repository..."
    git clone https://github.com/gyawalisanish0/symboLM.git
fi
cd symboLM
git pull origin main

# 2. Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# 3. Create checkpoints directories if missing
mkdir -p ./checkpoints/sft_adapter
mkdir -p ./checkpoints/tokenizer_extended

echo ""
echo "Setup complete!"
echo "If you have staged checkpoint-200 into ./checkpoints/sft_adapter/checkpoint-200,"
echo "start training with:"
echo ""
echo "python -m training.sft_train \\"
echo "    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\"
echo "    --data_dir ./data/symbolic_dataset \\"
echo "    --output_dir ./checkpoints/sft_adapter \\"
echo "    --epochs 3 \\"
echo "    --batch_size 2 \\"
echo "    --grad_accum 8 \\"
echo "    --lr 0.0002 \\"
echo "    --no_quant \\"
echo "    --resume_from_checkpoint ./checkpoints/sft_adapter/checkpoint-200"
echo "=========================================================="
