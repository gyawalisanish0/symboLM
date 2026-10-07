"""
SymboLM Dataset Builder
========================
Downloads GSM8K and MATH datasets, converts to symbolic CoT format,
and produces train/val/test splits ready for SFT training.

Usage:
    python -m data.build_dataset --output ./data/symbolic_dataset

Output structure:
    data/symbolic_dataset/
        train.jsonl       (~7000 examples)
        val.jsonl         (~500 examples)
        test.jsonl        (~1319 examples, GSM8K test set)
        stats.json        (compression ratios, conversion rates)
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

from symbolic.converter import convert, ConversionResult
from symbolic.grammar import validate_trace, THINK_OPEN, THINK_CLOSE, ANS_OPEN, ANS_CLOSE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
MIN_CONFIDENCE = 0.5   # minimum converter confidence to include sample


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

def iter_gsm8k(split: str = "train") -> Iterator[dict]:
    """
    Yield GSM8K examples as dicts with keys:
        problem, english_cot, answer
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    for ex in ds:
        # GSM8K answer field: "#### 42" at the end
        answer_match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", ex["answer"])
        if not answer_match:
            continue
        answer = answer_match.group(1).replace(",", "")
        # CoT is everything before "####"
        cot = re.sub(r"####.*", "", ex["answer"]).strip()
        yield {
            "problem":     ex["question"],
            "english_cot": cot,
            "answer":      answer,
            "source":      "gsm8k",
        }


# ---------------------------------------------------------------------------
# MATH dataset loader (subset — levels 1-3 only for tractability)
# ---------------------------------------------------------------------------

def iter_math(split: str = "train", max_level: int = 3) -> Iterator[dict]:
    """
    Yield MATH dataset examples (levels 1–max_level).
    """
    try:
        ds = load_dataset("lighteval/MATH", split=split, trust_remote_code=True)
    except Exception:
        # Fallback dataset name
        try:
            ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
        except Exception:
            print("Warning: MATH dataset not available, skipping")
            return

    for ex in ds:
        level = int(re.search(r"\d+", str(ex.get("level", "1"))).group())
        if level > max_level:
            continue

        # Extract numeric answer from boxed{} notation
        sol = ex.get("solution", "")
        boxed = re.search(r"\\boxed\{([^}]+)\}", sol)
        if not boxed:
            continue
        answer = boxed.group(1).strip()

        yield {
            "problem":     ex["problem"],
            "english_cot": sol,
            "answer":      answer,
            "source":      f"math_level{level}",
        }


# ---------------------------------------------------------------------------
# Core conversion pipeline
# ---------------------------------------------------------------------------

def process_example(ex: dict) -> dict | None:
    """
    Convert one example to symbolic format.
    Returns None if conversion fails or quality is too low.
    """
    result: ConversionResult | None = convert(
        problem=ex["problem"],
        english_cot=ex["english_cot"],
        answer=ex["answer"],
        min_confidence=MIN_CONFIDENCE,
    )

    if result is None:
        return None

    # Validate the symbolic trace
    validation = result.validation or validate_trace(result.symbolic_cot)

    # Reject if answer doesn't match (critical quality gate)
    if validation.extracted_answer and validation.extracted_answer != ex["answer"]:
        # Try normalizing (float comparison)
        try:
            extracted = float(validation.extracted_answer)
            expected = float(ex["answer"])
            if abs(extracted - expected) > 1e-4:
                return None
        except ValueError:
            # Non-numeric answer — do string comparison
            if validation.extracted_answer.strip() != ex["answer"].strip():
                return None

    # Build full formatted output
    full_output = (
        f"{THINK_OPEN}\n"
        f"{result.symbolic_cot}\n"
        f"{THINK_CLOSE}\n"
        f"{ANS_OPEN}{ex['answer']}{ANS_CLOSE}"
    )

    return {
        "problem":          ex["problem"],
        "symbolic_cot":     result.symbolic_cot,
        "answer":           ex["answer"],
        "full_output":      full_output,
        "english_cot":      ex["english_cot"],
        "source":           ex.get("source", "unknown"),
        "conversion_method": result.method,
        "confidence":       result.confidence,
        "english_tokens":   len(ex["english_cot"].split()),
        "symbolic_tokens":  len(result.symbolic_cot.split()),
        "is_valid":         validation.is_valid,
        "english_leak":     validation.english_leak_score,
    }


# ---------------------------------------------------------------------------
# SFT prompt formatter
# ---------------------------------------------------------------------------

def format_sft_prompt(example: dict, system_prompt: str | None = None) -> dict:
    """
    Format an example into the chat template format expected by SFT training.

    Format:
        system: <system_prompt>
        user: <problem>
        assistant: <think>symbolic_cot</think><ans>answer</ans>
    """
    if system_prompt is None:
        system_prompt = (
            "You are SymboLM, a reasoning assistant that thinks in compact symbolic notation "
            "for maximum efficiency. Express your reasoning in symbolic steps separated by |, "
            "use ∴ to mark your conclusion, and wrap your answer in <ans> tags. "
            "Eliminate all filler words — every token must carry logical content."
        )

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": example["problem"]},
            {"role": "assistant", "content": example["full_output"]},
        ],
        "answer":         example["answer"],
        "source":         example["source"],
        "symbolic_tokens": example["symbolic_tokens"],
        "english_tokens":  example["english_tokens"],
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(
    output_dir: str | Path = "./data/symbolic_dataset",
    max_gsm8k_train: int = 7000,
    max_math_train: int = 2000,
    val_size: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Build the full symbolic dataset and save to output_dir.
    Returns stats dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    all_train: list[dict] = []
    all_test:  list[dict] = []
    stats = {
        "total_processed": 0,
        "total_converted": 0,
        "total_rejected":  0,
        "by_source":       {},
        "avg_compression": 0.0,
    }

    # ----------------------------------------------------------------
    # Process GSM8K train
    # ----------------------------------------------------------------
    if verbose:
        print("Loading GSM8K train...")
    gsm8k_converted = 0
    gsm8k_rejected  = 0

    for ex in tqdm(iter_gsm8k("train"), desc="GSM8K train", disable=not verbose):
        if len(all_train) >= max_gsm8k_train:
            break
        stats["total_processed"] += 1
        processed = process_example(ex)
        if processed:
            all_train.append(format_sft_prompt(processed))
            gsm8k_converted += 1
        else:
            gsm8k_rejected += 1

    stats["by_source"]["gsm8k_train"] = {
        "converted": gsm8k_converted,
        "rejected":  gsm8k_rejected,
    }

    # ----------------------------------------------------------------
    # Process GSM8K test
    # ----------------------------------------------------------------
    if verbose:
        print("Loading GSM8K test...")
    for ex in tqdm(iter_gsm8k("test"), desc="GSM8K test", disable=not verbose):
        processed = process_example(ex)
        if processed:
            all_test.append(format_sft_prompt(processed))

    stats["by_source"]["gsm8k_test"] = {"converted": len(all_test)}

    # ----------------------------------------------------------------
    # Process MATH dataset (lower levels)
    # ----------------------------------------------------------------
    if verbose:
        print("Loading MATH dataset (levels 1-3)...")
    math_converted = 0
    math_rejected  = 0

    for ex in tqdm(iter_math("train", max_level=3), desc="MATH", disable=not verbose):
        if math_converted >= max_math_train:
            break
        stats["total_processed"] += 1
        processed = process_example(ex)
        if processed:
            all_train.append(format_sft_prompt(processed))
            math_converted += 1
        else:
            math_rejected += 1

    stats["by_source"]["math_train"] = {
        "converted": math_converted,
        "rejected":  math_rejected,
    }

    # ----------------------------------------------------------------
    # Shuffle and split
    # ----------------------------------------------------------------
    random.shuffle(all_train)
    val_set   = all_train[:val_size]
    train_set = all_train[val_size:]

    stats["total_converted"] = len(train_set) + len(val_set)
    stats["total_rejected"]  = stats["total_processed"] - stats["total_converted"]

    # Compute average compression ratio
    if train_set:
        compressions = [
            1.0 - (ex.get("symbolic_tokens", 1) / max(ex.get("english_tokens", 1), 1))
            for ex in train_set
        ]
        stats["avg_compression"] = round(sum(compressions) / len(compressions), 3)

    # ----------------------------------------------------------------
    # Save to JSONL files
    # ----------------------------------------------------------------
    splits = {
        "train": train_set,
        "val":   val_set,
        "test":  all_test,
    }

    for split_name, data in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if verbose:
            print(f"Saved {len(data)} examples → {out_path}")

    # Save stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Dataset built successfully!")
        print(f"  Train : {len(train_set):,}")
        print(f"  Val   : {len(val_set):,}")
        print(f"  Test  : {len(all_test):,}")
        print(f"  Avg token compression: {stats['avg_compression']:.0%}")
        print(f"  Saved to: {output_dir}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SymboLM symbolic dataset")
    parser.add_argument("--output",          default="./data/symbolic_dataset")
    parser.add_argument("--max-gsm8k-train", type=int, default=7000)
    parser.add_argument("--max-math-train",  type=int, default=2000)
    parser.add_argument("--val-size",        type=int, default=500)
    args = parser.parse_args()

    build_dataset(
        output_dir=args.output,
        max_gsm8k_train=args.max_gsm8k_train,
        max_math_train=args.max_math_train,
        val_size=args.val_size,
    )
