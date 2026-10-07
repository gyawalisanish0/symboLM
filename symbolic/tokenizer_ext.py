"""
SymboLM Tokenizer Extension
============================
Adds all symbolic DSL tokens as single vocabulary entries so that
each operator (→, ∴, ∵, ∀, ∃, etc.) costs exactly ONE token.

Without this, unicode symbols tokenize into multiple byte-fallback tokens,
which defeats the efficiency purpose entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from symbolic.grammar import (
    ALL_SPECIAL_TOKENS,
    KEYWORDS,
    LOGICAL_OPS,
    THINK_CLOSE,
    THINK_OPEN,
    ANS_OPEN,
    ANS_CLOSE,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# All tokens to add to the vocabulary
# ---------------------------------------------------------------------------

def get_all_new_tokens() -> list[str]:
    """
    Return the complete list of tokens to inject into the tokenizer vocabulary.
    Order matters — earlier tokens get lower IDs.
    """
    tokens: list[str] = []

    # 1. Structural special tokens (most important — add first)
    tokens += [THINK_OPEN, THINK_CLOSE, ANS_OPEN, ANS_CLOSE]

    # 2. Logical / math operators (single unicode chars)
    tokens += list(LOGICAL_OPS.keys())

    # 3. DSL keywords (short ASCII strings)
    tokens += list(KEYWORDS.keys())

    # 4. Common composite patterns that appear frequently
    #    Adding these as single tokens gives a big efficiency boost
    common_composites = [
        "∴ ans=",   # conclusion + answer
        "→ans=",    # implies answer
        "verify(",  # start of verification
        ")✓",       # end of verification, passed
        ")✗",       # end of verification, failed
        "hyp(",     # hypothesis open
        "let(",     # assignment open
        "case(",    # case open
        "derive(",  # derivation open
        "∂f=",      # derivative expression
        "∀x∈",      # universal quantifier over set
        "∃x∈",      # existential quantifier over set
    ]
    tokens += common_composites

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


# ---------------------------------------------------------------------------
# Main extension function
# ---------------------------------------------------------------------------

def extend_tokenizer(
    model_name_or_path: str,
    save_path: str | Path,
    verbose: bool = True,
) -> PreTrainedTokenizerFast:
    """
    Load a tokenizer, inject all symbolic DSL tokens, save to disk.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        save_path: Where to save the extended tokenizer.
        verbose: Print token mapping info.

    Returns:
        The extended tokenizer (also saved to save_path).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading base tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    original_vocab_size = len(tokenizer)
    new_tokens = get_all_new_tokens()

    # Filter tokens already in the vocabulary
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]

    if verbose:
        print(f"Original vocab size : {original_vocab_size:,}")
        print(f"New tokens (total)  : {len(new_tokens)}")
        print(f"Tokens already in vocab: {len(new_tokens) - len(tokens_to_add)}")
        print(f"Tokens to add       : {len(tokens_to_add)}")

    # Add as regular tokens (not special tokens) so they participate
    # in normal attention — only <think>/<ans> are "special"
    special = [THINK_OPEN, THINK_CLOSE, ANS_OPEN, ANS_CLOSE]
    regular = [t for t in tokens_to_add if t not in special]

    if special:
        added_special = tokenizer.add_special_tokens(
            {"additional_special_tokens": special}
        )
        if verbose:
            print(f"Added special tokens: {added_special}")

    if regular:
        added_regular = tokenizer.add_tokens(regular)
        if verbose:
            print(f"Added regular tokens: {added_regular}")

    new_vocab_size = len(tokenizer)

    if verbose:
        print(f"New vocab size      : {new_vocab_size:,}")
        print(f"Vocabulary growth   : +{new_vocab_size - original_vocab_size}")

        # Show a sample of the new token IDs
        print("\nSample token → ID mappings:")
        sample_tokens = list(LOGICAL_OPS.keys())[:8] + [THINK_OPEN, THINK_CLOSE, ANS_OPEN, ANS_CLOSE]
        for tok in sample_tokens:
            tid = tokenizer.convert_tokens_to_ids(tok)
            # Check it's truly a single token (not split)
            encoded = tokenizer.encode(tok, add_special_tokens=False)
            single = len(encoded) == 1
            status = "✓ single token" if single else f"✗ splits into {len(encoded)} tokens"
            print(f"  '{tok}' → id={tid}  [{status}]")

    tokenizer.save_pretrained(save_path)

    # Save metadata about what was added
    meta = {
        "base_model": model_name_or_path,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "added_tokens": tokens_to_add,
        "symbolic_tokens": list(LOGICAL_OPS.keys()),
        "keyword_tokens": list(KEYWORDS.keys()),
        "special_tokens": special,
    }
    with open(save_path / "tokenizer_extension_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nTokenizer saved to: {save_path}")

    return tokenizer


# ---------------------------------------------------------------------------
# Semantic Warm-Start Embedding Initialization (Mitigating Symbol Drift)
# ---------------------------------------------------------------------------

SEMANTIC_ANCHORS = {
    "→": "implies",
    "∴": "therefore",
    "∵": "because",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "⊕": "xor",
    "≡": "equivalent",
    "≠": "different",
    "≤": "less",
    "≥": "greater",
    "∈": "in",
    "∉": "notin",
    "⊂": "subset",
    "∀": "forall",
    "∃": "exists",
    "✓": "correct",
    "✗": "wrong",
    "hyp": "assume",
    "derive": "calculate",
    "conclude": "conclude",
    "let": "let",
    "verify": "check",
    "backtrack": "retry",
    "case": "case",
    "base": "base",
    "step": "step",
    "ans": "answer",
}


def initialize_symbol_embeddings(model, tokenizer, verbose: bool = True) -> int:
    """
    Initializes newly added symbolic token embeddings using the vector
    of their closest English semantic equivalent rather than random noise.
    
    This grounds the symbols directly in pre-existing reasoning circuits,
    mitigating the Symbol Grounding Problem and cold-start semantic drift.
    """
    import torch

    embeddings = model.get_input_embeddings()
    weight = embeddings.weight.data
    initialized_count = 0

    for symbol, anchor_word in SEMANTIC_ANCHORS.items():
        sym_id = tokenizer.convert_tokens_to_ids(symbol)
        anchor_ids = tokenizer.encode(anchor_word, add_special_tokens=False)

        # Ensure valid token IDs and that symbol was added
        if sym_id is not None and sym_id < weight.shape[0] and anchor_ids:
            with torch.no_grad():
                # Average anchor token vectors (if word splits into multiple subwords)
                anchor_vec = weight[anchor_ids].mean(dim=0)
                # Copy into new symbol embedding
                weight[sym_id] = anchor_vec
                initialized_count += 1

    if verbose:
        print(f"[Semantic Warm-Start] Initialized {initialized_count} symbol embeddings from English semantic anchors.")

    return initialized_count


# ---------------------------------------------------------------------------
# Token efficiency audit
# ---------------------------------------------------------------------------

def audit_token_efficiency(tokenizer: PreTrainedTokenizerFast) -> dict:
    """
    Run a quick audit: for each example in the grammar, compare token
    counts for English CoT vs symbolic CoT using the ACTUAL tokenizer.
    """
    from symbolic.grammar import EXAMPLE_TRACES

    results = []
    for ex in EXAMPLE_TRACES:
        eng_ids = tokenizer.encode(ex["english_cot"], add_special_tokens=False)
        sym_ids = tokenizer.encode(ex["symbolic_cot"], add_special_tokens=False)
        ratio = 1.0 - len(sym_ids) / max(len(eng_ids), 1)
        results.append({
            "problem_short": ex["problem"][:50],
            "english_tokens": len(eng_ids),
            "symbolic_tokens": len(sym_ids),
            "compression_ratio": round(ratio, 3),
        })

    avg = sum(r["compression_ratio"] for r in results) / len(results)
    return {"per_example": results, "average_compression": round(avg, 3)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extend tokenizer with symbolic DSL tokens")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
        help="Base model to load tokenizer from",
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoints/tokenizer_extended",
        help="Where to save the extended tokenizer",
    )
    parser.add_argument("--audit", action="store_true", help="Run token efficiency audit after extension")
    args = parser.parse_args()

    tok = extend_tokenizer(args.model, args.save_path)

    if args.audit:
        print("\n--- Token Efficiency Audit ---")
        audit = audit_token_efficiency(tok)
        for row in audit["per_example"]:
            print(
                f"  English: {row['english_tokens']:3d} | "
                f"Symbolic: {row['symbolic_tokens']:3d} | "
                f"Saved: {row['compression_ratio']:.0%} | "
                f"{row['problem_short']}"
            )
        print(f"\n  Average compression: {audit['average_compression']:.0%}")
