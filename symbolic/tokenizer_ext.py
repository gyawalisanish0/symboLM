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
from typing import Iterable, Optional

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
    revision: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """
    Load a tokenizer, inject all symbolic DSL tokens, save to disk.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        save_path: Where to save the extended tokenizer.
        verbose: Print token mapping info.
        revision: Immutable Hugging Face revision/commit for the base tokenizer.

    Returns:
        The extended tokenizer (also saved to save_path).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading base tokenizer: {model_name_or_path}@{revision or 'default'}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
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
        "base_revision": revision,
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
    # Multi-token anchors: a single-token anchor here would make the added
    # output rows exact copies of an existing token's row.  If those rows stay
    # frozen, their logits are permanently tied and greedy decoding can only
    # select the lower-id token.  The trainable token deltas are the primary
    # fix; these distinct anchors also give them a non-degenerate warm start.
    "<ans>": "final answer",
    "</ans>": "answer end",
}


def initialize_symbol_embeddings(
    model,
    tokenizer,
    verbose: bool = True,
    added_tokens: Optional[Iterable[str]] = None,
) -> int:
    """
    Initialize genuinely *new* symbolic token rows from English anchors.

    Both the input embedding matrix and the output LM head are initialized when
    the model has untied embeddings.  The adapter freezes these modules, so the
    initialized rows must be written before ``attach_lora_adapter``; otherwise
    the output head would be saved with the random rows created by
    ``resize_token_embeddings``.

    Only tokens reported by ``tokenizer.get_added_vocab()`` (or explicitly
    passed in ``added_tokens``) are touched.  In particular, existing base
    tokens such as ``ans`` are never overwritten just because they appear in
    ``SEMANTIC_ANCHORS``.
    """
    import torch

    embeddings = model.get_input_embeddings()
    if embeddings is None:
        return 0
    input_weight = embeddings.weight.data

    added_vocab = set(tokenizer.get_added_vocab().keys())
    eligible = set(added_tokens) if added_tokens is not None else added_vocab
    eligible &= added_vocab

    output_embeddings = model.get_output_embeddings()
    output_weight = output_embeddings.weight.data if output_embeddings is not None else None
    tied = (
        output_weight is None
        or output_weight.data_ptr() == input_weight.data_ptr()
    )

    initialized_input = 0
    initialized_output = 0
    with torch.no_grad():
        for symbol, anchor_word in SEMANTIC_ANCHORS.items():
            if symbol not in eligible:
                continue
            sym_id = tokenizer.convert_tokens_to_ids(symbol)
            anchor_ids = tokenizer.encode(anchor_word, add_special_tokens=False)
            if (
                not isinstance(sym_id, int)
                or sym_id < 0
                or sym_id >= input_weight.shape[0]
                or sym_id == getattr(tokenizer, "unk_token_id", None)
                or not anchor_ids
            ):
                continue
            anchor_ids = [
                int(token_id)
                for token_id in anchor_ids
                if 0 <= int(token_id) < input_weight.shape[0]
            ]
            if not anchor_ids:
                continue

            # Average anchor token vectors if the English word is split into
            # multiple subword tokens.
            input_weight[sym_id] = input_weight[anchor_ids].mean(dim=0)
            initialized_input += 1

            if not tied and sym_id < output_weight.shape[0]:
                head_anchor_ids = [
                    int(token_id)
                    for token_id in anchor_ids
                    if 0 <= int(token_id) < output_weight.shape[0]
                ]
                if head_anchor_ids:
                    output_weight[sym_id] = output_weight[head_anchor_ids].mean(dim=0)
                    initialized_output += 1

    if verbose:
        mode = "tied input/output" if tied else "input + output head"
        print(
            f"[Semantic Warm-Start] Initialized {initialized_input} genuinely "
            f"added symbol rows ({mode}; output rows={initialized_output}) "
            "from English semantic anchors."
        )

    return initialized_input


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
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face revision/commit for the base tokenizer",
    )
    parser.add_argument("--audit", action="store_true", help="Run token efficiency audit after extension")
    args = parser.parse_args()

    tok = extend_tokenizer(args.model, args.save_path, revision=args.revision)

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
