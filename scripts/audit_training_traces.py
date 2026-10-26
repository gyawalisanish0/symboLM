"""Audit and repair the r6 SFT traces for r7 Verified-Trace Repair.

This script is deliberately fail-closed.  It audits every source row, rejects
rows with a wrong terminal answer or malformed contract, and only retains a
reasoning row as *verified* when all of the following hold:

* the terminal answer is correct and matches the ground truth;
* the register trace is structurally valid;
* the stated reduction (``∴ ans=...``) matches the terminal answer;
* a closed arithmetic equality in the deduce chain evaluates true and agrees
  with the terminal answer;
* the verify block evaluates true, references the answer, and contains an
  independent arithmetic or combinatorial check.

A weak, tautological, false, unrelated, or unparsed verify block is repaired
from a closed equality in the deduce chain when possible.  Otherwise the verify
block is removed and the row is written only to ``answer_checked_only.jsonl``.
It is never written to the verified train/val split.

Atomic bypass rows have no verify block by contract.  They are retained only as
``verified_bypass_contract`` and are excluded from the reasoning
verification-validity rate.

Dry run (default; writes only the audit report)::

    .venv\\Scripts\\python.exe scripts\\audit_training_traces.py

Write the audited split (still does not train)::

    .venv\\Scripts\\python.exe scripts\\audit_training_traces.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sympy  # noqa: E402

from eval.strict_scorer import (  # noqa: E402
    score_declared_answer,
    validate_register_trace,
    verify_answer,
)
from symbolic.prompts import (  # noqa: E402
    classify_target_body,
    extract_target_answer,
    normalize_target_body,
)

AUDIT_VERSION = "r7-verified-trace-audit-v1"
VERIFIED_COVERAGE = {"verified_reasoning", "verified_bypass_contract"}
SOURCE_REQUIRED_FIELDS = (
    "id",
    "prompt",
    "full_target",
    "ground_truth",
    "target_kind",
    "split",
)
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


class AuditError(RuntimeError):
    """Raised when the audit cannot establish provenance or a valid contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise AuditError(f"Missing JSONL file: {source}")
    rows: List[Dict[str, Any]] = []
    with open(source, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"Malformed JSONL at {source}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise AuditError(f"Non-object JSONL row at {source}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)


def check_lf_file(path: str | Path) -> None:
    data = Path(path).read_bytes()
    if b"\r\n" in data:
        raise AuditError(f"CRLF line endings found in {path}; expected LF")
    if data and not data.endswith(b"\n"):
        raise AuditError(f"File does not end with a newline: {path}")


def load_source_manifest(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise AuditError(f"Missing source manifest: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"Malformed source manifest: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"Source manifest must be a JSON object: {source}")
    return payload


def validate_source_rows(
    rows: Sequence[Dict[str, Any]],
    split: str,
    expected_count: int,
) -> Dict[str, Any]:
    if len(rows) != expected_count:
        raise AuditError(
            f"{split} row-count mismatch: expected {expected_count}, got {len(rows)}"
        )
    ids: set[str] = set()
    kinds = Counter()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in SOURCE_REQUIRED_FIELDS if field not in row]
        if missing:
            raise AuditError(
                f"{split} row {index} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise AuditError(f"{split} row {index} has an empty id")
        if row_id in ids:
            raise AuditError(f"Duplicate {split} id: {row_id!r}")
        ids.add(row_id)
        if str(row.get("split", "")).strip() != split:
            raise AuditError(
                f"{split} row {row_id!r} declares split {row.get('split')!r}"
            )
        kind = str(row.get("target_kind", "")).strip()
        if kind not in {"reasoning", "bypass"}:
            raise AuditError(
                f"{split} row {row_id!r} has unknown target_kind {kind!r}"
            )
        kinds[kind] += 1
        for field in ("prompt", "full_target", "ground_truth"):
            if not str(row.get(field, "")).strip():
                raise AuditError(f"{split} row {row_id!r} has empty {field}")
        if "\r" in str(row["full_target"]):
            raise AuditError(f"{split} row {row_id!r} contains a carriage return")
    return {"rows": len(rows), "ids": ids, "kinds": dict(kinds)}


# ── Verification parsing and repair ──────────────────────────────────────────


def _strip_verify_wrapper(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("<reg:verify>") and value.endswith("</reg:verify>"):
        value = value[len("<reg:verify>") : -len("</reg:verify>")].strip()
    value = re.sub(r"^verify\((.*)\)\s*[✓✗]?\s*$", r"\1", value, flags=re.S)
    return value.strip()


def normalize_expression(raw: str) -> str:
    """Normalize the limited math notation used by the r6 verify blocks."""
    value = _strip_verify_wrapper(raw)
    value = value.replace("∧", " and ").replace("∨", " or ").replace("¬", " not ")
    value = value.replace("≠", " != ").replace("≤", " <= ").replace("≥", " >= ")
    value = value.replace("×", "*").replace("·", "*").replace("÷", "/")
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace("π", "pi").replace("√", "sqrt")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\div", "/").replace("\\pi", "pi")
    value = re.sub(r"\bmod\b", "%", value, flags=re.I)
    value = re.sub(r"\bphi\(([^()]*)\)", r"totient(\1)", value, flags=re.I)
    value = re.sub(r"φ\(([^()]*)\)", r"totient(\1)", value)
    value = re.sub(
        r"\bP\(([^()]*),([^()]*)\)",
        r"(factorial(\1)/factorial(\1-(\2)))",
        value,
    )
    value = re.sub(r"\bC\(([^()]*),([^()]*)\)", r"binomial(\1,\2)", value)
    for superscript, exponent in zip(SUPERSCRIPT_DIGITS, range(10)):
        value = value.replace(superscript, f"**{exponent}")
    value = value.replace("^", "**")
    value = re.sub(r"(?<![<>=!])=(?!=)", "==", value)
    return value.strip()


def evaluate_expression(expression: str) -> Optional[bool]:
    try:
        parsed = sympy.sympify(expression, evaluate=True)
        simplified = sympy.simplify(parsed)
        if simplified in (sympy.true, sympy.false, True, False):
            return bool(simplified)
        if getattr(simplified, "is_Boolean", False):
            return bool(simplified)
        return None
    except Exception:
        return None


def has_independent_arithmetic(expression: str) -> bool:
    if re.search(r"\b(factorial|binomial|totient|sqrt)\b", expression):
        return True
    stripped = re.sub(
        r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", "", expression
    )
    stripped = re.sub(
        r"==|<=|>=|!=|<|>|\(|\)|\s+|\band\b|\bor\b|\bnot\b",
        "",
        stripped,
    )
    return bool(re.search(r"[+\-*/%!]", stripped))


def references_answer(raw: str, answer: str) -> bool:
    if re.search(r"\b(ans|answer|det|result|value)\b", raw, re.I):
        return True
    if not answer:
        return False
    return bool(
        re.search(
            rf"(?<![\d.]){re.escape(answer)}(?![\d.])",
            raw,
        )
    )


def classify_verify(raw: str, answer: str) -> Dict[str, Any]:
    expression = normalize_expression(raw)
    expanded = re.sub(r"\b(ans|answer)\b", f"({answer})", expression, flags=re.I)
    expanded = re.sub(
        r"\b(det|result|value)\b", f"({answer})", expanded, flags=re.I
    )
    evaluated = evaluate_expression(expanded)
    has_alias = bool(re.search(r"\b(det|result|value)\b", raw, re.I))
    has_ans_variable = bool(re.search(r"\b(ans|answer)\b", raw, re.I))
    has_reference = references_answer(raw, answer)
    independent = has_independent_arithmetic(expanded)
    simple = re.sub(r"\s+", "", expanded)
    tautology = False
    if evaluated is True:
        if has_alias:
            tautology = True
        elif has_ans_variable and not independent and has_reference:
            tautology = True
        elif not independent and simple in {
            f"({answer})=={answer}",
            f"{answer}=={answer}",
            f"({answer})==({answer})",
        }:
            tautology = True

    if evaluated is False:
        status = "false"
    elif evaluated is None:
        status = "unparsed"
    elif not has_reference:
        status = "unrelated"
    elif tautology:
        status = "tautology"
    elif independent:
        status = "valid"
    else:
        status = "weak"

    return {
        "status": status,
        "raw": (raw or "").strip(),
        "expression": expression,
        "expanded": expanded,
        "evaluated": evaluated,
        "references_answer": has_reference,
        "independent_arithmetic": independent,
        "alias_tautology": has_alias,
        "answer": answer,
    }


def extract_block(body: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_therefore_answer(deduce: str) -> Optional[str]:
    match = re.search(r"∴\s*ans\s*=\s*([^\s|<]+)", deduce, re.I)
    return match.group(1).strip() if match else None


def _split_equality_chain(deduce: str) -> List[Tuple[str, str]]:
    head = re.split(r"∴\s*ans\s*=", deduce, flags=re.I)[0]
    pairs: List[Tuple[str, str]] = []
    for segment in re.split(r"[|→\n;]", head):
        pieces = re.split(r"(?<![<>=!≡])=(?!=)", segment)
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        for left, right in zip(pieces, pieces[1:]):
            pairs.append((left, right))
    return pairs


def is_closed_expression(raw: str) -> bool:
    expression = normalize_expression(raw)
    if not expression:
        return False
    if re.search(r"\b(ans|answer|det|result|value)\b", expression, re.I):
        return False
    if re.search(r"==|<=|>=|!=|<|>", expression):
        return False
    has_arithmetic = bool(
        re.search(r"\b(factorial|binomial|totient|sqrt)\b", expression)
        or re.search(r"[+\-*/%!]|\*\*", expression)
    )
    if not has_arithmetic:
        return False
    residual = re.sub(
        r"\b(factorial|binomial|totient|sqrt|pi)\b", "", expression
    )
    return not bool(re.search(r"[A-Za-z_]", residual))


def extract_last_closed_equality(
    deduce: str, answer: str
) -> Optional[Tuple[str, str, Optional[bool]]]:
    for left, right in reversed(_split_equality_chain(deduce)):
        if not verify_answer(right, answer)[0]:
            continue
        if not is_closed_expression(left):
            continue
        candidate = f"{normalize_expression(left)} == {answer}"
        return left, right, evaluate_expression(candidate)
    return None


def repair_verify_from_deduce(
    deduce: str, answer: str
) -> Optional[Tuple[str, str, str]]:
    """Return ``(new_block, lhs, rhs)`` when a closed deduce check exists."""
    for left, right in reversed(_split_equality_chain(deduce)):
        if not verify_answer(right, answer)[0]:
            continue
        if not is_closed_expression(left):
            continue
        normalized_left = normalize_expression(left)
        candidate = f"{normalized_left} == {answer}"
        if evaluate_expression(candidate) is not True:
            continue
        if not has_independent_arithmetic(candidate):
            continue
        new_block = (
            f"<reg:verify>verify({normalized_left} == {answer})✓</reg:verify>"
        )
        classification = classify_verify(new_block, answer)
        if classification["status"] == "valid":
            return new_block, left, right
    return None


def replace_verify(body: str, new_block: str) -> str:
    if "<reg:verify>" in body:
        updated = re.sub(
            r"<reg:verify>.*?</reg:verify>",
            lambda _match: new_block,
            body,
            count=1,
            flags=re.DOTALL,
        )
    else:
        updated = body.replace("</ans>", new_block + "\n</ans>", 1)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated.strip()


def remove_verify(body: str) -> str:
    cleaned = re.sub(
        r"\n?<reg:verify>.*?</reg:verify>\n?", "\n", body, flags=re.DOTALL
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_ambiguity(prompt: str) -> List[str]:
    flags: List[str] = []
    if (
        re.search(r"\bpositive integers?\b", prompt, re.I)
        and re.search(r"\b(find|determine|compute|what is)\b", prompt, re.I)
        and not re.search(
            r"\b(unique|distinct|smallest|largest|least|greatest|minimum|maximum)\b",
            prompt,
            re.I,
        )
    ):
        flags.append("positive_integer_solution_set_review")
    return flags


# ── Row audit ─────────────────────────────────────────────────────────────────


def _base_audit(
    row: Dict[str, Any],
    split: str,
    source_sha: str,
    coverage: str,
) -> Dict[str, Any]:
    original_full = str(row.get("full_target", ""))
    return {
        "audit_version": AUDIT_VERSION,
        "source_split": split,
        "source_sha256": source_sha,
        "source_row_sha256": sha256_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "coverage_level": coverage,
        "answer_checked": False,
        "answer_correct": False,
        "format_valid": False,
        "terminal_answer": None,
        "derivation_status": "not_checked",
        "arithmetic_status": "not_checked",
        "verification_scope": None,
        "verification_status": "not_checked",
        "original_verification_status": "not_checked",
        "repair_action": None,
        "repair_expression": None,
        "verification_valid": False,
        "bypass_contract_valid": False,
        "ambiguity_flags": [],
        "original_target_sha256": sha256_text(original_full),
        "audited_target_sha256": None,
        "rejection_reasons": [],
    }


def _reject(
    row: Dict[str, Any],
    split: str,
    source_sha: str,
    reasons: Sequence[str],
    *,
    answer: Optional[str] = None,
    ground_truth: Optional[str] = None,
    format_reason: Optional[str] = None,
    verify_status: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    row_out = dict(row)
    audit = _base_audit(row, split, source_sha, "rejected")
    audit["rejection_reasons"] = list(reasons)
    audit["answer_checked"] = False
    audit["answer_correct"] = False
    audit["format_valid"] = format_reason in {None, "reasoning", "bypass", "declared_answer"}
    audit["terminal_answer"] = answer
    audit["verification_status"] = verify_status or "rejected"
    audit["original_verification_status"] = verify_status or "rejected"
    row_out["audit"] = audit
    detail = {
        "id": str(row.get("id", "")),
        "split": split,
        "reasons": list(reasons),
        "answer": answer,
        "ground_truth": ground_truth if ground_truth is not None else row.get("ground_truth"),
        "format_reason": format_reason,
        "target_kind": row.get("target_kind"),
        "source_index": row.get("source_index"),
        "original_verification_status": verify_status,
    }
    return "rejected", row_out, detail


def audit_row(
    row: Dict[str, Any],
    split: str,
    source_sha: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    original_full = str(row.get("full_target", ""))
    body = normalize_target_body(original_full)
    kind, kind_reason = classify_target_body(body)
    declared_kind = str(row.get("target_kind", "")).strip()

    if kind not in {"reasoning", "bypass"}:
        return _reject(
            row,
            split,
            source_sha,
            [f"target_invalid:{kind_reason}"],
            verify_status="invalid_target",
        )
    if kind != declared_kind:
        return _reject(
            row,
            split,
            source_sha,
            [f"target_kind_mismatch:{declared_kind}!={kind}"],
            verify_status="target_kind_mismatch",
        )

    score = score_declared_answer(
        body,
        arm="C",
        ground_truth=str(row.get("ground_truth", "")),
        tokens_emitted=1,
        budget=1_000_000,
        eos_emitted=True,
        expected_kind=kind,
    )
    answer = score["answer"]
    if not score["is_correct"]:
        return _reject(
            row,
            split,
            source_sha,
            ["answer_mismatch"],
            answer=answer,
            format_reason=score["format_reason"],
            verify_status="answer_mismatch",
        )
    if not score["format_valid"]:
        return _reject(
            row,
            split,
            source_sha,
            [f"format_invalid:{score['format_reason']}"],
            answer=answer,
            format_reason=score["format_reason"],
            verify_status="format_invalid",
        )

    terminal = extract_target_answer(body)
    if terminal is None or not verify_answer(answer, terminal)[0]:
        return _reject(
            row,
            split,
            source_sha,
            ["terminal_answer_mismatch"],
            answer=answer,
            format_reason=score["format_reason"],
            verify_status="terminal_answer_mismatch",
        )

    if kind == "reasoning":
        deduce = extract_block(body, "reg:deduce")
        if not deduce:
            return _reject(
                row,
                split,
                source_sha,
                ["missing_deduce"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status="missing_deduce",
            )
        therefore = extract_therefore_answer(deduce)
        if not therefore:
            return _reject(
                row,
                split,
                source_sha,
                ["missing_stated_reduction"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status="missing_stated_reduction",
            )
        if not verify_answer(therefore, answer)[0]:
            return _reject(
                row,
                split,
                source_sha,
                ["stated_reduction_mismatch"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status="stated_reduction_mismatch",
            )
        if not verify_answer(therefore, str(row.get("ground_truth", "")))[0]:
            return _reject(
                row,
                split,
                source_sha,
                ["stated_reduction_ground_truth_mismatch"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status="stated_reduction_ground_truth_mismatch",
            )
        last_closed = extract_last_closed_equality(deduce, answer)
        if last_closed is None:
            arithmetic_status = "unchecked"
        else:
            _left, _right, evaluated = last_closed
            if evaluated is False:
                return _reject(
                    row,
                    split,
                    source_sha,
                    ["arithmetic_mismatch"],
                    answer=answer,
                    format_reason=score["format_reason"],
                    verify_status="arithmetic_mismatch",
                )
            arithmetic_status = "checked_true" if evaluated is True else "unparsed"
        derivation_status = "matched"
    else:
        deduce = ""
        arithmetic_status = "not_applicable"
        derivation_status = "not_applicable"

    verify_blocks = re.findall(r"<reg:verify>(.*?)</reg:verify>", body, re.DOTALL)
    if kind == "bypass":
        if verify_blocks:
            return _reject(
                row,
                split,
                source_sha,
                ["bypass_contains_verify"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status="bypass_contains_verify",
            )
        original_verify_status = "not_applicable"
        final_verify_status = "not_applicable"
        repair_action = None
        repair_expression = None
        audited_body = body
        coverage = "verified_bypass_contract"
    else:
        if verify_blocks:
            original_verify = classify_verify(verify_blocks[-1], answer)
            original_verify_status = original_verify["status"]
        else:
            original_verify = {
                "status": "missing",
                "raw": "",
                "evaluated": None,
                "references_answer": False,
                "independent_arithmetic": False,
            }
            original_verify_status = "missing"

        repair_action: Optional[str] = None
        repair_expression: Optional[str] = None
        final_verify_status: str
        audited_body: str
        coverage: str

        if arithmetic_status != "checked_true":
            audited_body = remove_verify(body) if verify_blocks else body
            final_verify_status = "removed" if verify_blocks else "missing"
            repair_action = f"removed_verify_arithmetic_{arithmetic_status}"
            coverage = "answer_checked_only"
        elif original_verify_status == "valid":
            audited_body = body
            final_verify_status = "valid"
            coverage = "verified_reasoning"
        else:
            repaired = repair_verify_from_deduce(deduce, answer)
            if repaired is not None:
                new_block, _left, _right = repaired
                audited_body = replace_verify(body, new_block)
                final_verify_status = "repaired"
                repair_action = (
                    f"replaced_{original_verify_status}_with_deduce_check"
                )
                repair_expression = new_block
                coverage = "verified_reasoning"
            else:
                audited_body = remove_verify(body) if verify_blocks else body
                final_verify_status = "removed" if verify_blocks else "missing"
                repair_action = f"removed_{original_verify_status}_verify"
                coverage = "answer_checked_only"

        audited_kind, audited_reason = classify_target_body(audited_body)
        if audited_kind != "reasoning":
            return _reject(
                row,
                split,
                source_sha,
                [f"audited_target_invalid:{audited_reason}"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status=original_verify_status,
            )
        audited_valid, audited_format_reason = validate_register_trace(
            audited_body, arm="C", expected_kind="reasoning"
        )
        if not audited_valid:
            return _reject(
                row,
                split,
                source_sha,
                [f"audited_format_invalid:{audited_format_reason}"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status=original_verify_status,
            )
        audited_score = score_declared_answer(
            audited_body,
            arm="C",
            ground_truth=str(row.get("ground_truth", "")),
            tokens_emitted=1,
            budget=1_000_000,
            eos_emitted=True,
            expected_kind="reasoning",
        )
        if not audited_score["is_correct"] or not audited_score["format_valid"]:
            return _reject(
                row,
                split,
                source_sha,
                ["audited_score_failed"],
                answer=answer,
                format_reason=score["format_reason"],
                verify_status=original_verify_status,
            )

    prefix = " thinking\n" if original_full.lstrip().startswith(" thinking") else ""
    audited_full = f"{prefix}{audited_body}" if prefix else audited_body
    ambiguous_flags = detect_ambiguity(str(row.get("prompt", "")))
    audit = _base_audit(row, split, source_sha, coverage)
    audit.update(
        {
            "answer_checked": True,
            "answer_correct": True,
            "format_valid": True,
            "terminal_answer": terminal,
            "derivation_status": derivation_status,
            "arithmetic_status": arithmetic_status,
            "verification_scope": (
                "reasoning_verify" if kind == "reasoning" else "bypass_contract"
            ),
            "verification_status": final_verify_status,
            "original_verification_status": original_verify_status,
            "repair_action": repair_action,
            "repair_expression": repair_expression,
            "verification_valid": coverage == "verified_reasoning",
            "bypass_contract_valid": coverage == "verified_bypass_contract",
            "ambiguity_flags": ambiguous_flags,
            "audited_target_sha256": sha256_text(audited_full),
        }
    )
    row_out = dict(row)
    row_out["full_target"] = audited_full
    row_out["audit"] = audit
    detail = {
        "id": str(row.get("id", "")),
        "split": split,
        "coverage_level": coverage,
        "answer": answer,
        "ground_truth": row.get("ground_truth"),
        "format_reason": score["format_reason"],
        "target_kind": kind,
        "original_verification_status": original_verify_status,
        "verification_status": final_verify_status,
        "repair_action": repair_action,
        "repair_expression": repair_expression,
        "arithmetic_status": arithmetic_status,
        "derivation_status": derivation_status,
        "ambiguous_flags": ambiguous_flags,
        "source_index": row.get("source_index"),
    }
    return coverage, row_out, detail


# ── Summary and receipts ─────────────────────────────────────────────────────


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    coverage = Counter(record["coverage"] for record in records)
    original_status = Counter()
    final_status = Counter()
    repair_actions = Counter()
    arithmetic = Counter()
    derivation = Counter()
    rejection_reasons = Counter()
    ambiguity = Counter()
    reasoning_audited = 0
    bypass_audited = 0
    original_valid = 0
    repaired = 0
    removed = 0
    ambiguous_rows = 0

    for record in records:
        row = record["row"]
        detail = record["detail"]
        kind = str(row.get("target_kind", ""))
        if kind == "reasoning":
            reasoning_audited += 1
        elif kind == "bypass":
            bypass_audited += 1
        audit = row.get("audit", {})
        original_status[str(audit.get("original_verification_status", "unknown"))] += 1
        final_status[str(audit.get("verification_status", "unknown"))] += 1
        repair_actions[str(audit.get("repair_action", "none"))] += 1
        arithmetic[str(audit.get("arithmetic_status", "unknown"))] += 1
        derivation[str(audit.get("derivation_status", "unknown"))] += 1
        if audit.get("original_verification_status") == "valid":
            original_valid += 1
        if audit.get("verification_status") == "repaired":
            repaired += 1
        if audit.get("verification_status") == "removed":
            removed += 1
        if audit.get("ambiguity_flags"):
            ambiguous_rows += 1
            for flag in audit["ambiguity_flags"]:
                ambiguity[flag] += 1
        if record["coverage"] == "rejected":
            for reason in detail.get("reasons", []):
                rejection_reasons[reason] += 1

    verified_reasoning = coverage.get("verified_reasoning", 0)
    verified_bypass = coverage.get("verified_bypass_contract", 0)
    answer_checked_only = coverage.get("answer_checked_only", 0)
    rejected = coverage.get("rejected", 0)
    retained = verified_reasoning + verified_bypass

    def rate(numerator: int, denominator: int) -> Optional[float]:
        return (numerator / denominator) if denominator else None

    return {
        "audited": len(records),
        "reasoning_audited": reasoning_audited,
        "bypass_audited": bypass_audited,
        "coverage_counts": dict(sorted(coverage.items())),
        "verified_reasoning": verified_reasoning,
        "verified_bypass_contract": verified_bypass,
        "answer_checked_only": answer_checked_only,
        "rejected": rejected,
        "retained_total": retained,
        "original_valid": original_valid,
        "repaired": repaired,
        "removed": removed,
        "original_verification_status_counts": dict(sorted(original_status.items())),
        "verification_status_counts": dict(sorted(final_status.items())),
        "repair_action_counts": dict(sorted(repair_actions.items())),
        "arithmetic_status_counts": dict(sorted(arithmetic.items())),
        "derivation_status_counts": dict(sorted(derivation.items())),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "ambiguity_flag_counts": dict(sorted(ambiguity.items())),
        "ambiguous_rows": ambiguous_rows,
        "verification_validity_rate_original": rate(original_valid, reasoning_audited),
        "verification_validity_rate_after_repair": rate(
            verified_reasoning, reasoning_audited
        ),
        "bypass_contract_rate": rate(verified_bypass, bypass_audited),
        "answer_checked_only_rate": rate(answer_checked_only, len(records)),
        "rejected_rate": rate(rejected, len(records)),
    }


def _report_payload(
    *,
    status: str,
    train_path: Path,
    val_path: Path,
    source_manifest_path: Path,
    source_manifest: Dict[str, Any],
    source_manifest_sha: str,
    train_sha: str,
    val_sha: str,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    train_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    total_summary: Dict[str, Any],
    max_examples: int,
    outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rejected_examples = [
        record["detail"]
        for record in records
        if record["coverage"] == "rejected"
    ][:max_examples]
    answer_checked_examples = [
        record["detail"]
        for record in records
        if record["coverage"] == "answer_checked_only"
    ][:max_examples]
    repaired_examples = [
        record["detail"]
        for record in records
        if record["coverage"] == "verified_reasoning"
        and record["detail"].get("verification_status") == "repaired"
    ][:max_examples]

    warnings = [
        "r7 train/val retain only verified_reasoning and "
        "verified_bypass_contract rows.",
        "answer_checked_only rows have the verify block removed and are "
        "excluded from the verified train/val split.",
        "Bypass rows are answer-and-format checked under the atomic contract; "
        "they are not counted in the reasoning verification-validity rate.",
        "This is a data audit, not a training run, benchmark result, or claim "
        "of accuracy/token efficiency.",
        "r7 training remains blocked until the v3 path closes and a Phase 1b "
        "preregistration exists.",
    ]

    return {
        "schema": "symboLM.r7_trace_audit.v1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_version": AUDIT_VERSION,
        "audit_script": _relative(Path(__file__)),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "source": {
            "train_path": _relative(train_path),
            "train_sha256": train_sha,
            "train_rows": len(train_rows),
            "val_path": _relative(val_path),
            "val_sha256": val_sha,
            "val_rows": len(val_rows),
            "source_manifest_path": _relative(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest": source_manifest,
        },
        "train": train_summary,
        "val": val_summary,
        "totals": total_summary,
        "rejected_examples": rejected_examples,
        "answer_checked_only_examples": answer_checked_examples,
        "repaired_examples": repaired_examples,
        "warnings": warnings,
        "claim_boundary": (
            "This audit does not establish benchmark accuracy, format value, "
            "token efficiency, latency, or mobile performance. No r7 artifact "
            "exists. Do not call answer_checked_only rows verified."
        ),
        "outputs": outputs,
    }


def build_manifest(
    *,
    train_path: Path,
    val_path: Path,
    source_manifest_path: Path,
    source_manifest: Dict[str, Any],
    source_manifest_sha: str,
    train_sha: str,
    val_sha: str,
    train_source_rows: Sequence[Dict[str, Any]],
    val_source_rows: Sequence[Dict[str, Any]],
    train_written: Sequence[Dict[str, Any]],
    val_written: Sequence[Dict[str, Any]],
    answer_checked_only: Sequence[Dict[str, Any]],
    rejected: Sequence[Dict[str, Any]],
    train_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    total_summary: Dict[str, Any],
    output_hashes: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "schema": "symboLM.verified_sft.v1",
        "status": "written",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_version": AUDIT_VERSION,
        "audit_script": _relative(Path(__file__)),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "source": {
            "train_path": _relative(train_path),
            "train_sha256": train_sha,
            "train_rows": len(train_source_rows),
            "val_path": _relative(val_path),
            "val_sha256": val_sha,
            "val_rows": len(val_source_rows),
            "source_manifest_path": _relative(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest": source_manifest,
        },
        "counts": {
            "train_written": len(train_written),
            "val_written": len(val_written),
            "answer_checked_only": len(answer_checked_only),
            "rejected": len(rejected),
            "train_verified_reasoning": train_summary["verified_reasoning"],
            "train_verified_bypass_contract": train_summary[
                "verified_bypass_contract"
            ],
            "val_verified_reasoning": val_summary["verified_reasoning"],
            "val_verified_bypass_contract": val_summary[
                "verified_bypass_contract"
            ],
        },
        "coverage": {
            "train": train_summary,
            "val": val_summary,
            "totals": total_summary,
        },
        "outputs": output_hashes,
        "rules": {
            "terminal_answer_required": True,
            "format_required": True,
            "stated_reduction_required": True,
            "closed_arithmetic_equality_required_for_verified_reasoning": True,
            "verify_must_evaluate_true": True,
            "verify_must_reference_answer": True,
            "verify_must_have_independent_arithmetic": True,
            "repair_from_deduce_allowed": True,
            "answer_checked_only_excluded_from_train_val": True,
            "bypass_contract_is_separate_from_reasoning_verify": True,
        },
        "warnings": [
            "This manifest describes a data audit, not a training result.",
            "answer_checked_only rows are not part of train.jsonl or val.jsonl.",
            "Bypass rows are verified under the atomic bypass contract, not by a "
            "reasoning verify block.",
        ],
        "claim_boundary": (
            "No r7 training or evaluation result exists. Do not cite this "
            "manifest as accuracy, token efficiency, or format-value evidence."
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "research_sft" / "train.jsonl",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=ROOT / "data" / "research_sft" / "val.jsonl",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data" / "research_sft" / "manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "research_sft_verified",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Audit report path. Defaults to results/r7_verified/AUDIT_DRY_RUN.json "
        "or AUDIT.json with --write.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the audited split. Still does not train.",
    )
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="Exit nonzero if any source row is rejected.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum examples included in the audit report.",
    )
    args = parser.parse_args(argv)

    if args.max_examples < 0:
        parser.error("--max-examples cannot be negative")

    train_path = args.train.resolve()
    val_path = args.val.resolve()
    source_manifest_path = args.source_manifest.resolve()
    report_path = args.report
    if report_path is None:
        report_path = (
            ROOT
            / "results"
            / "r7_verified"
            / ("AUDIT.json" if args.write else "AUDIT_DRY_RUN.json")
        )
    report_path = report_path.resolve()

    source_manifest = load_source_manifest(source_manifest_path)
    expected_train = int(source_manifest.get("train", 0))
    expected_val = int(source_manifest.get("val", 0))
    if expected_train <= 0 or expected_val <= 0:
        raise AuditError("Source manifest must declare positive train/val counts")

    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    if train_sha.upper() != str(source_manifest.get("train_sha256", "")).upper():
        raise AuditError(
            "Train SHA-256 mismatch: expected "
            f"{source_manifest.get('train_sha256')}, got {train_sha}"
        )
    if val_sha.upper() != str(source_manifest.get("val_sha256", "")).upper():
        raise AuditError(
            "Val SHA-256 mismatch: expected "
            f"{source_manifest.get('val_sha256')}, got {val_sha}"
        )
    if str(source_manifest.get("line_endings", "")).lower() != "lf":
        raise AuditError("Source manifest does not declare LF line endings")
    check_lf_file(train_path)
    check_lf_file(val_path)

    source_file = ROOT / str(source_manifest.get("source", ""))
    if not source_file.is_file():
        raise AuditError(f"Declared source file is missing: {source_file}")
    source_file_sha = sha256_file(source_file)
    if source_file_sha.upper() != str(
        source_manifest.get("source_sha256", "")
    ).upper():
        raise AuditError(
            "Source-file SHA-256 mismatch: expected "
            f"{source_manifest.get('source_sha256')}, got {source_file_sha}"
        )

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    train_meta = validate_source_rows(train_rows, "train", expected_train)
    val_meta = validate_source_rows(val_rows, "val", expected_val)
    overlap = train_meta["ids"] & val_meta["ids"]
    if overlap:
        raise AuditError(
            f"Train/val id overlap: {len(overlap)} ids; first: {sorted(overlap)[:5]}"
        )
    expected_kinds = source_manifest.get("target_kinds", {})
    actual_kinds = Counter()
    for row in list(train_rows) + list(val_rows):
        actual_kinds[str(row.get("target_kind"))] += 1
    if expected_kinds:
        for kind, count in expected_kinds.items():
            if int(count) != actual_kinds.get(str(kind), 0):
                raise AuditError(
                    f"Target-kind count mismatch for {kind!r}: "
                    f"expected {count}, got {actual_kinds.get(str(kind), 0)}"
                )

    records: List[Dict[str, Any]] = []
    for split, rows, source_sha in (
        ("train", train_rows, train_sha),
        ("val", val_rows, val_sha),
    ):
        for row in rows:
            coverage, row_out, detail = audit_row(row, split, source_sha)
            records.append(
                {
                    "split": split,
                    "coverage": coverage,
                    "row": row_out,
                    "detail": detail,
                }
            )

    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    train_summary = summarize(train_records)
    val_summary = summarize(val_records)
    total_summary = summarize(records)

    if args.write:
        train_written = [
            record["row"]
            for record in train_records
            if record["coverage"] in VERIFIED_COVERAGE
        ]
        val_written = [
            record["row"]
            for record in val_records
            if record["coverage"] in VERIFIED_COVERAGE
        ]
        answer_checked_only = [
            record["row"]
            for record in records
            if record["coverage"] == "answer_checked_only"
        ]
        rejected = [
            record["row"]
            for record in records
            if record["coverage"] == "rejected"
        ]
        if not train_written or not val_written:
            raise AuditError(
                "Refusing to write an empty verified train or val split"
            )
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = out_dir / "train.jsonl"
        val_out = out_dir / "val.jsonl"
        answer_checked_out = out_dir / "answer_checked_only.jsonl"
        rejected_out = out_dir / "rejected.jsonl"
        write_jsonl(train_out, train_written)
        write_jsonl(val_out, val_written)
        write_jsonl(answer_checked_out, answer_checked_only)
        write_jsonl(rejected_out, rejected)
        output_hashes = {
            "train": {
                "path": _relative(train_out),
                "sha256": sha256_file(train_out),
                "rows": len(train_written),
            },
            "val": {
                "path": _relative(val_out),
                "sha256": sha256_file(val_out),
                "rows": len(val_written),
            },
            "answer_checked_only": {
                "path": _relative(answer_checked_out),
                "sha256": sha256_file(answer_checked_out),
                "rows": len(answer_checked_only),
            },
            "rejected": {
                "path": _relative(rejected_out),
                "sha256": sha256_file(rejected_out),
                "rows": len(rejected),
            },
        }
        manifest = build_manifest(
            train_path=train_path,
            val_path=val_path,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            source_manifest_sha=sha256_file(source_manifest_path),
            train_sha=train_sha,
            val_sha=val_sha,
            train_source_rows=train_rows,
            val_source_rows=val_rows,
            train_written=train_written,
            val_written=val_written,
            answer_checked_only=answer_checked_only,
            rejected=rejected,
            train_summary=train_summary,
            val_summary=val_summary,
            total_summary=total_summary,
            output_hashes=output_hashes,
        )
        manifest_path = out_dir / "manifest.json"
        write_json(manifest_path, manifest)
        report = _report_payload(
            status="written",
            train_path=train_path,
            val_path=val_path,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            source_manifest_sha=sha256_file(source_manifest_path),
            train_sha=train_sha,
            val_sha=val_sha,
            train_rows=train_rows,
            val_rows=val_rows,
            records=records,
            train_summary=train_summary,
            val_summary=val_summary,
            total_summary=total_summary,
            max_examples=args.max_examples,
            outputs={
                **output_hashes,
                "manifest": {
                    "path": _relative(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
            },
        )
        write_json(report_path, report)
        print(
            f"[INFO] Wrote verified split: {_relative(train_out)} "
            f"({len(train_written)}), {_relative(val_out)} ({len(val_written)})",
            flush=True,
        )
        print(
            f"[INFO] answer_checked_only={len(answer_checked_only)} "
            f"rejected={len(rejected)}",
            flush=True,
        )
        print(f"[INFO] Manifest: {_relative(manifest_path)}", flush=True)
    else:
        report = _report_payload(
            status="dry_run",
            train_path=train_path,
            val_path=val_path,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            source_manifest_sha=sha256_file(source_manifest_path),
            train_sha=train_sha,
            val_sha=val_sha,
            train_rows=train_rows,
            val_rows=val_rows,
            records=records,
            train_summary=train_summary,
            val_summary=val_summary,
            total_summary=total_summary,
            max_examples=args.max_examples,
            outputs=None,
        )
        write_json(report_path, report)

    print(
        f"[INFO] Train: audited={train_summary['audited']} "
        f"verified_reasoning={train_summary['verified_reasoning']} "
        f"verified_bypass={train_summary['verified_bypass_contract']} "
        f"answer_checked_only={train_summary['answer_checked_only']} "
        f"rejected={train_summary['rejected']}",
        flush=True,
    )
    print(
        f"[INFO] Val: audited={val_summary['audited']} "
        f"verified_reasoning={val_summary['verified_reasoning']} "
        f"verified_bypass={val_summary['verified_bypass_contract']} "
        f"answer_checked_only={val_summary['answer_checked_only']} "
        f"rejected={val_summary['rejected']}",
        flush=True,
    )
    original_rate = total_summary["verification_validity_rate_original"]
    repaired_rate = total_summary["verification_validity_rate_after_repair"]
    print(
        "[INFO] Reasoning verification validity: "
        f"original={original_rate:.3f}" if original_rate is not None else
        "[INFO] Reasoning verification validity: original=n/a",
        flush=True,
    )
    print(
        "[INFO] Reasoning verification validity: "
        f"after_repair={repaired_rate:.3f}" if repaired_rate is not None else
        "[INFO] Reasoning verification validity: after_repair=n/a",
        flush=True,
    )
    print(f"[INFO] Audit report: {_relative(report_path)}", flush=True)
    if not args.write:
        print("[INFO] Dry run only; no audited split was written.", flush=True)

    if args.fail_on_reject and total_summary["rejected"]:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"[AUDIT ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
