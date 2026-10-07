# SymboLM — Task Tracker

## Phase 0: Project Scaffold
- [x] Implementation plan approved
- [x] Create project directory structure
- [x] Write task.md

## Phase 1: Symbolic DSL Core
- [x] `symbolic/grammar.py` — DSL spec, validator, examples
- [x] `symbolic/tokenizer_ext.py` — extend base tokenizer with symbol tokens
- [x] `symbolic/converter.py` — English CoT → symbolic CoT conversion

## Phase 2: Data Pipeline
- [x] `data/build_dataset.py` — pull GSM8K/MATH, convert to symbolic format
- [x] `data/symbolic_dataset.py` — PyTorch Dataset class
- [x] `data/verify_answers.py` — ground truth answer checker

## Phase 3: Training
- [x] `training/reward.py` — multi-component reward function
- [x] `training/sft_train.py` — Stage 1 SFT (QLoRA 4-bit / Unsloth)
- [x] `training/grpo_train.py` — Stage 2 GRPO RL (DeepSeek R1 style)

## Phase 4: Colab Notebooks
- [x] `colab/01_setup.ipynb` — GPU check, Drive mount, package installs
- [x] `colab/02_data_prep.ipynb` — Tokenizer extension & dataset conversion
- [x] `colab/03_sft.ipynb` — Stage 1 SFT training
- [x] `colab/04_grpo.ipynb` — Stage 2 GRPO RL training
- [x] `colab/05_eval.ipynb` — Benchmarking & interactive playground

## Phase 5: Eval & Inference
- [x] `eval/benchmark.py` — pass@1 test accuracy
- [x] `eval/efficiency.py` — token count comparison & speed audit
- [x] `inference/generate.py` — streaming generation with KV cache
- [x] `inference/symbolic_to_text.py` — decompress symbolic trace to English

## Phase 6: Config & Utilities
- [x] `config.py` — central config & Colab T4 tuning
- [x] `requirements.txt` — dependencies
- [x] `README.md` — project guide & execution documentation
