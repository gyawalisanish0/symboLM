"""
SymboLM Register Tokenizer Manager
==================================
Extends standard autoregressive tokenizers with dedicated, atomic tokens
for SymboLM Register v2.1 cognitive delimiters. Eliminates subword fragmentation,
collapsing wrapper costs from 14 tokens down to exactly 2 tokens.

Author: Sanish Gyawali (Rupandehi, Nepal)
AI Systems Collaborator: Antigravity (Google DeepMind)
"""

from typing import List, Tuple, Optional, Any
import torch

REGISTER_SPECIAL_TOKENS: List[str] = [
    "<reg:plan>", "</reg:plan>",
    "<reg:deduce>", "</reg:deduce>",
    "<reg:verify>", "</reg:verify>",
    "<reg:bypass>", "</reg:bypass>",
    "<reg:intent>", "</reg:intent>",
]


def get_register_tokens() -> List[str]:
    """Return the list of 10 canonical SymboLM register delimiter tokens."""
    return list(REGISTER_SPECIAL_TOKENS)


def detect_model_family(tokenizer: Any) -> str:
    """Detects whether tokenizer belongs to LLaMA, Qwen, Gemma, or Mistral family."""
    tok_class = tokenizer.__class__.__name__.lower()
    if "qwen" in tok_class:
        return "Qwen"
    elif "llama" in tok_class:
        return "LLaMA"
    elif "gemma" in tok_class:
        return "Gemma"
    elif "mistral" in tok_class:
        return "Mistral"
    return "Universal CausalLM"



def setup_register_tokenizer(
    tokenizer: Any,
    model: Optional[Any] = None,
    init_with_mean: bool = True
) -> Tuple[Any, Optional[Any]]:
    """
    Register SymboLM cognitive delimiters as explicit atomic special tokens.

    Args:
        tokenizer: Hugging Face PreTrainedTokenizer or PreTrainedTokenizerFast.
        model: Optional Hugging Face PreTrainedModel whose embeddings to resize.
        init_with_mean: If True and model is provided, initializes new token
                        embeddings with the mean of existing embeddings.

    Returns:
        Tuple of (tokenizer, model).
    """
    # 1. Add special tokens
    tokens_to_add = [tok for tok in REGISTER_SPECIAL_TOKENS if tok not in tokenizer.get_vocab()]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    # 2. Resize model embeddings if model is provided
    if model is not None and num_added > 0:
        old_embeddings = model.get_input_embeddings()
        old_vocab_size, embedding_dim = old_embeddings.weight.shape
        model.resize_token_embeddings(len(tokenizer))
        new_embeddings = model.get_input_embeddings()

        if init_with_mean:
            with torch.no_grad():
                mean_embedding = old_embeddings.weight[:old_vocab_size].mean(dim=0)
                new_embeddings.weight[old_vocab_size:] = mean_embedding

    return tokenizer, model


def measure_wrapper_token_count(tokenizer: Any, register_name: str = "bypass", content: str = "K") -> int:
    """
    Measure total tokens required to represent an encapsulated register.
    E.g., `<reg:bypass>K</reg:bypass>`.
    """
    text = f"<reg:{register_name}>{content}</reg:{register_name}>"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)
