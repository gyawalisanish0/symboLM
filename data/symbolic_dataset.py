"""
SymboLM PyTorch Dataset & Collator
==================================

The old dataset wrapped targets in literal ``<｜begin of sentence｜>`` and
``<｜end of sentence｜>`` strings.  Those are *not* special tokens; the tokenizer
treats them as seven ordinary tokens.  The model therefore never learned a real
BOS/EOS and looped after ``</reg:verify>``.

This version builds the exact prompt/target pair with the tokenizer's real
special tokens, includes the system prompt used at evaluation time, and refuses
to train on targets that do not declare a terminal answer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from symbolic.prompts import (
    REGISTER_SYSTEM_PROMPT,
    build_chat_prompt,
    build_full_text,
    classify_target_body,
    normalize_target_body,
)


class SymbolicSFTDataset(Dataset):
    """
    Reads JSONL records with either ``full_target`` or ``messages``.

    Each item is rendered as::

        <BOS><system><|User|><prompt><|Assistant|><think>\n
        <target body>
        <EOS>

    The prompt is masked with -100 so the loss is calculated only on the target.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        system_prompt: str = REGISTER_SYSTEM_PROMPT,
        strict_format: bool = True,
        allow_dialogue: bool = False,
        allow_contrastive: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.strict_format = strict_format
        self.allow_dialogue = allow_dialogue
        self.allow_contrastive = allow_contrastive

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        raw_records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))

        self.records: List[Dict[str, Any]] = []
        self.skipped: Counter = Counter()
        self.kinds: Counter = Counter()

        for item in raw_records:
            try:
                _, _, assistant = self._parts(item)
                body = normalize_target_body(assistant)
                kind, _reason = classify_target_body(body)
                if kind == "dialogue" and not allow_dialogue:
                    self.skipped["dialogue_out_of_scope"] += 1
                    continue
                if kind == "contrastive" and not allow_contrastive:
                    self.skipped["contrastive_out_of_scope"] += 1
                    continue
                if strict_format and kind not in {"reasoning", "bypass"}:
                    self.skipped[f"invalid:{kind}"] += 1
                    continue
                self.records.append(item)
                self.kinds[kind] += 1
            except Exception as exc:
                self.skipped[f"render_error:{type(exc).__name__}"] += 1

        if not self.records:
            raise ValueError(
                f"No usable SFT records in {path}. "
                f"Skipped: {dict(self.skipped)}. "
                "Run scripts/prepare_research_sft.py to build a canonical split."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _parts(self, item: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return (system, user, assistant) for a record."""
        if "messages" in item:
            messages = item["messages"]
            system = self.system_prompt
            user = ""
            assistant = ""
            for message in messages:
                role = message.get("role")
                content = message.get("content", "") or ""
                if role == "system" and content:
                    system = content
                elif role == "user":
                    user = content
                elif role == "assistant":
                    assistant = content
            if not user or not assistant:
                raise ValueError("messages record is missing user or assistant content")
            return system, user, assistant

        if "full_target" in item:
            return (
                item.get("system_prompt", self.system_prompt),
                item.get("prompt", ""),
                item.get("full_target", ""),
            )

        raise ValueError(f"Unrecognized dataset record keys: {sorted(item.keys())}")

    def get_prompt(self, idx: int) -> str:
        """Prompt string for generation checks (no assistant target)."""
        system, user, _ = self._parts(self.records[idx])
        return build_chat_prompt(self.tokenizer, system, user, open_think=True)

    def get_target_body(self, idx: int) -> str:
        _, _, assistant = self._parts(self.records[idx])
        return normalize_target_body(assistant)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        system, user, assistant = self._parts(item)
        target_body = normalize_target_body(assistant)

        if self.strict_format:
            kind, reason = classify_target_body(target_body)
            if kind not in {"reasoning", "bypass"}:
                raise ValueError(
                    f"Invalid target at index {idx} (kind={kind}, reason={reason})"
                )

        prompt_str = build_chat_prompt(self.tokenizer, system, user, open_think=True)
        full_str = build_full_text(self.tokenizer, prompt_str, target_body)

        prompt_encoded = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        full_encoded = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = full_encoded["input_ids"]
        attention_mask = full_encoded["attention_mask"]
        prompt_ids = prompt_encoded["input_ids"]

        if len(prompt_ids) >= self.max_length:
            raise ValueError(
                f"Prompt at index {idx} is longer than max_length={self.max_length}"
            )
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Prompt is not a token prefix of the full training string at index {idx}. "
                "This would corrupt the label mask."
            )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and input_ids[-1] != eos_id:
            raise ValueError(
                f"Target at index {idx} was truncated before the real EOS token "
                f"(max_length={self.max_length}). Increase max_length or shorten the target."
            )

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "records": len(self.records),
            "kinds": dict(self.kinds),
            "skipped": dict(self.skipped),
            "max_length": self.max_length,
            "strict_format": self.strict_format,
        }


class SymbolicDataCollator:
    """Pads input_ids, attention_mask, and labels to batch max_len.
    On TPU, supports fixed max_length or multiples of 64/128 to prevent XLA
    dynamic shape recompilation.
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        if self.max_length is not None:
            max_len = max(max_len, self.max_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            batch_attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch_labels.append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
