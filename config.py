"""
SymboLM Global Configuration
=============================
Default settings, model configurations, and resource optimizations.
Supports Google Colab TPU v5e (v5e-1) with native bfloat16 and GPU fallbacks.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SymboLMConfig:
    # Model: DeepSeek-R1-Distill-Qwen-1.5B (native reasoning + math capability)
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Hardware target: Auto-detects TPU v5e / CUDA
    use_tpu: bool = False  # Set to True when running on TPU v5e
    precision: str = "bfloat16"  # "bfloat16" (TPU/modern GPU) or "float16" (T4)

    # Paths
    project_root: Path = Path(__file__).parent.resolve()
    data_dir: Path = project_root / "data" / "symbolic_dataset"
    checkpoints_dir: Path = project_root / "checkpoints"
    tokenizer_dir: Path = project_root / "checkpoints" / "tokenizer_extended"
    sft_output_dir: Path = project_root / "checkpoints" / "sft_adapter"
    grpo_output_dir: Path = project_root / "checkpoints" / "grpo_adapter"

    # Colab Google Drive path for persistent checkpoints
    colab_drive_checkpoint: str = "/content/drive/MyDrive/symboLM_checkpoints"

    # System prompt for symbolic reasoning
    system_prompt: str = (
        "You are SymboLM, an ultra-efficient reasoning model. "
        "Solve problems step-by-step using a compact symbolic language between <think> and </think>. "
        "Use '|' as step separators, operators like '→', '∴', '∵', 'let', 'hyp', and 'verify'. "
        "Never use conversational English filler in your thinking trace. "
        "Output the final numeric or exact answer enclosed strictly inside <ans> and </ans>."
    )

    # LoRA Config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # SFT Training Hyperparameters (TPU v5e-1 has 819 GB/s bandwidth -> can handle larger batch sizes!)
    sft_batch_size: int = 4            # Increased from 2 thanks to TPU v5e HBM2e bandwidth
    sft_grad_accum_steps: int = 4      # Effective batch size = 16
    sft_lr: float = 2e-4
    sft_epochs: int = 3
    sft_max_seq_len: int = 1024

    # GRPO Training Hyperparameters (R1-style)
    grpo_num_generations: int = 4      # G=4 completions per prompt
    grpo_batch_size: int = 2           # TPU throughput advantage
    grpo_grad_accum_steps: int = 8
    grpo_lr: float = 5e-6
    grpo_max_prompt_len: int = 512
    grpo_max_completion_len: int = 512
    grpo_beta: float = 0.04            # KL penalty coefficient
    grpo_temperature: float = 0.8

    # Cloud Persistence & Remote Hub Storage
    hf_repo_id: str = "gyawalisanish0/symboLM-checkpoints"
    hf_private_repo: bool = True
    enable_hf_sync: bool = True
    sync_every_n_steps: int = 25
    enable_local_pruning: bool = True
    keep_latest_checkpoints: int = 1
    prune_after_sync_only: bool = True


config = SymboLMConfig()
