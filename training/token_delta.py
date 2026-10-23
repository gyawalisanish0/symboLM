"""Trainable deltas for genuinely added tokenizer rows.

The tokenizer extension adds new tokens.  ``initialize_symbol_embeddings``
warm-starts their input and output rows from English anchors, and
``attach_lora_adapter`` freezes the copied ``modules_to_save`` matrices.

Freezing those matrices is useful for artifact reproducibility, but it has one
fatal edge case: if two added tokens are initialized from the same anchor, their
output rows are byte-identical.  A frozen LM head cannot tell them apart, so
their logits are always tied and greedy decoding selects the lower token id.
That is exactly what happened in r5: ``<ans>`` and ``</ans>`` were exact copies
of the base ``answer`` row, so the model emitted ``answer`` instead of the
terminal markers.

This module keeps the frozen full matrices but adds a tiny trainable delta for
each genuinely added token:

* ``embed_delta`` is added to the input embedding rows.
* ``head_delta`` is added to the output logits at the same token ids.

The deltas are applied with forward hooks, so the 151k x 1536 matrices are not
copied on every forward.  Before the final save, the deltas are merged into the
active ``modules_to_save`` weights.  The adapter format is unchanged, which
means the existing evaluation and package gates keep working.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn


def _module_weight(module: Optional[nn.Module]) -> Optional[torch.Tensor]:
    """Return the active weight for a plain module or PEFT save wrapper."""
    if module is None:
        return None
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"Module {type(module).__name__} has no weight tensor")
    return weight


class TokenDeltaAdapter(nn.Module):
    """A small trainable perturbation for added input/output embedding rows."""

    def __init__(
        self,
        token_ids: Iterable[int],
        input_module: nn.Module,
        output_module: nn.Module,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        ids = sorted({int(token_id) for token_id in token_ids})
        if not ids:
            raise ValueError("TokenDeltaAdapter requires at least one token id")
        if min(ids) < 0:
            raise ValueError(f"Token ids must be non-negative: {ids}")

        input_weight = _module_weight(input_module)
        output_weight = _module_weight(output_module)
        if input_weight is None or output_weight is None:
            raise RuntimeError("TokenDeltaAdapter requires input and output embedding modules")
        if input_weight.data_ptr() == output_weight.data_ptr():
            raise RuntimeError(
                "TokenDeltaAdapter does not support tied input/output weights. "
                "The r6 adapter requires an untied LM head."
            )

        vocab_size = max(int(input_weight.shape[0]), int(output_weight.shape[0]))
        if max(ids) >= vocab_size:
            raise ValueError(
                f"Token id {max(ids)} is outside the resized vocabulary ({vocab_size})"
            )

        self.token_ids = ids
        self.num_tokens = len(ids)
        self._merged = False
        self._handles: list[Any] = []

        input_device = input_weight.device
        output_device = output_weight.device
        self.embed_delta = nn.Parameter(
            torch.zeros(self.num_tokens, input_weight.shape[1], dtype=dtype, device=input_device)
        )
        self.head_delta = nn.Parameter(
            torch.zeros(self.num_tokens, output_weight.shape[1], dtype=dtype, device=output_device)
        )

        # Buffers are non-persistent: they are routing tables, not artifact data.
        token_ids_tensor = torch.tensor(ids, dtype=torch.long, device=input_device)
        self.register_buffer("_token_ids", token_ids_tensor, persistent=False)
        index_map = torch.full((vocab_size,), -1, dtype=torch.long, device=input_device)
        index_map[token_ids_tensor] = torch.arange(
            self.num_tokens, dtype=torch.long, device=input_device
        )
        self.register_buffer("_index_map", index_map, persistent=False)

    def attach(self, model: nn.Module) -> "TokenDeltaAdapter":
        """Register the embedding and LM-head hooks on ``model``."""
        if self._handles:
            return self
        if self._merged:
            raise RuntimeError("Cannot attach a TokenDeltaAdapter after its deltas were merged")

        input_module = model.get_input_embeddings()
        output_module = model.get_output_embeddings()
        if input_module is None or output_module is None:
            raise RuntimeError("Model must expose input and output embeddings")

        self._handles.append(input_module.register_forward_hook(self._embedding_hook))
        self._handles.append(output_module.register_forward_hook(self._lm_head_hook))
        return self

    def _embedding_hook(self, module, inputs, output):
        if self._merged or not inputs or not torch.is_tensor(inputs[0]):
            return output

        input_ids = inputs[0]
        index_map = self._index_map.to(input_ids.device)
        delta_index = index_map[input_ids]
        valid = delta_index >= 0
        if not torch.any(valid):
            return output

        delta = self.embed_delta.to(device=output.device, dtype=output.dtype)
        gathered = delta[delta_index.clamp_min(0)]
        return output + gathered * valid.unsqueeze(-1).to(output.dtype)

    def _lm_head_hook(self, module, inputs, output):
        if self._merged or not inputs or not torch.is_tensor(inputs[0]):
            return output

        hidden = inputs[0]
        delta = self.head_delta.to(device=hidden.device, dtype=hidden.dtype)
        delta_logits = torch.matmul(hidden, delta.t())
        token_ids = self._token_ids.to(device=output.device)

        if hidden.dim() == 3:
            index = token_ids.view(1, 1, -1).expand(
                delta_logits.shape[0], delta_logits.shape[1], -1
            )
            return output.scatter_add(2, index, delta_logits.to(output.dtype))
        if hidden.dim() == 2:
            index = token_ids.view(1, -1).expand(delta_logits.shape[0], -1)
            return output.scatter_add(1, index, delta_logits.to(output.dtype))
        raise RuntimeError(f"Unsupported LM-head hidden-state shape: {tuple(hidden.shape)}")

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def merge(self, model: nn.Module) -> Dict[str, Any]:
        """Merge the learned deltas into the frozen ``modules_to_save`` rows."""
        if self._merged:
            return self.summary()

        self.remove_hooks()
        input_module = model.get_input_embeddings()
        output_module = model.get_output_embeddings()
        input_weight = _module_weight(input_module)
        output_weight = _module_weight(output_module)
        if input_weight is None or output_weight is None:
            raise RuntimeError("Cannot merge token deltas without input and output weights")
        if input_weight.data_ptr() == output_weight.data_ptr():
            raise RuntimeError("Refusing to merge token deltas into tied weights")

        with torch.no_grad():
            input_ids = self._token_ids.to(input_weight.device)
            input_weight.index_add_(
                0,
                input_ids,
                self.embed_delta.to(device=input_weight.device, dtype=input_weight.dtype),
            )
            output_ids = self._token_ids.to(output_weight.device)
            output_weight.index_add_(
                0,
                output_ids,
                self.head_delta.to(device=output_weight.device, dtype=output_weight.dtype),
            )

        self._merged = True
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        """Return compact, JSON-safe evidence for the run manifest."""
        return {
            "present": True,
            "merged": bool(self._merged),
            "token_count": int(self.num_tokens),
            "token_ids": list(self.token_ids),
            "embed_delta_l2": float(self.embed_delta.detach().float().norm().item()),
            "head_delta_l2": float(self.head_delta.detach().float().norm().item()),
            "embed_delta_max_abs": float(self.embed_delta.detach().float().abs().max().item()),
            "head_delta_max_abs": float(self.head_delta.detach().float().abs().max().item()),
        }


def attach_token_delta_adapter(
    model: nn.Module,
    token_ids: Iterable[int],
) -> TokenDeltaAdapter:
    """Attach trainable deltas for the genuinely added tokenizer rows."""
    adapter = TokenDeltaAdapter(
        token_ids,
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    )
    model.symbol_token_delta = adapter
    adapter.attach(model)
    return adapter


def merge_token_deltas(model: nn.Module) -> Dict[str, Any]:
    """Merge the attached token deltas, if any, and return manifest evidence."""
    adapter = getattr(model, "symbol_token_delta", None)
    if adapter is None:
        return {"present": False, "merged": False}
    return adapter.merge(model)


def assert_terminal_marker_rows_distinct(model: nn.Module, tokenizer) -> Dict[str, Any]:
    """Fail closed if the saved answer markers still have tied output rows."""
    token_names = ("<ans>", "</ans>", "answer")
    token_ids: Dict[str, int] = {}
    for name in token_names:
        token_id = tokenizer.convert_tokens_to_ids(name)
        if not isinstance(token_id, int) or token_id < 0:
            raise RuntimeError(f"Tokenizer cannot resolve {name!r}")
        token_ids[name] = token_id

    output_weight = _module_weight(model.get_output_embeddings())
    input_weight = _module_weight(model.get_input_embeddings())
    if output_weight is None or input_weight is None:
        raise RuntimeError("Cannot verify terminal-marker rows without embedding weights")

    checks: Dict[str, Any] = {"distinct": True, "token_ids": token_ids}
    pairs = (
        ("<ans>", "answer"),
        ("</ans>", "answer"),
        ("<ans>", "</ans>"),
    )
    with torch.no_grad():
        for label, weight in (("output", output_weight), ("input", input_weight)):
            rows = {
                name: weight[token_id].detach()
                for name, token_id in token_ids.items()
            }
            for left, right in pairs:
                equal = torch.equal(rows[left], rows[right])
                max_abs = float((rows[left].float() - rows[right].float()).abs().max().item())
                checks[f"{label}:{left}_vs_{right}"] = {
                    "equal": bool(equal),
                    "max_abs_diff": max_abs,
                }
                if equal:
                    checks["distinct"] = False
    if not checks["distinct"]:
        raise RuntimeError(
            "Terminal answer marker rows are still exactly tied; refusing to "
            f"package the adapter: {checks}"
        )
    return checks
