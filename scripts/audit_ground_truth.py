"""
Audit the held-out benchmark and produce the v2 evaluation file.
===============================================================

What this script does
---------------------
1. Applies documented ground-truth overrides from
   ``data/ground_truth_overrides.json``.
2. Flags duplicate prompt templates (the 200-problem set contains repeated
   problems, so raw n=200 overstates the independent sample size).
3. Writes ``data/held_out_eval_200_v2.jsonl`` with provenance fields.
4. Writes a compact Markdown audit report and a JSON manifest with hashes.

It deliberately does not silently rewrite benchmark answers.  Every changed
answer must appear in the overrides file with a reason.

Usage::

    python scripts/audit_ground_truth.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.strict_scorer import apply_ground_truth_override, load_ground_truth_overrides  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "held_out_eval_200_clean.jsonl"
DEFAULT_OVERRIDES = ROOT / "data" / "ground_truth_overrides.json"
DEFAULT_OUTPUT = ROOT / "data" / "held_out_eval_200_v2.jsonl"
DEFAULT_REPORT = ROOT / "data" / "held_out_eval_200_v2_audit.md"
DEFAULT_MANIFEST = ROOT / "data" / "held_out_eval_200_v2_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit held-out ground truth")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = load_jsonl(args.input)
    overrides = load_ground_truth_overrides(args.overrides)

    # Group identical prompts.  These are not necessarily wrong, but they are
    # not independent samples and must be visible in the report.
    prompt_groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        prompt_groups[row["prompt"]].append(row["id"])

    out_rows: List[Dict[str, Any]] = []
    override_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []

    for row in rows:
        audited = apply_ground_truth_override(row, overrides)
        group = prompt_groups[audited["prompt"]]
        audited["duplicate_prompt"] = len(group) > 1
        audited["duplicate_group_size"] = len(group)
        audited["duplicate_group_ids"] = list(group)
        audited["needs_review"] = audited["id"] in overrides or len(group) > 1
        if audited["ground_truth_source"] == "override":
            override_rows.append(audited)
        if audited["duplicate_prompt"]:
            duplicate_rows.append(audited)
        out_rows.append(audited)

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in held-out set; refusing to continue.")

    unique_prompts = len(prompt_groups)
    duplicate_groups = {p: ids for p, ids in prompt_groups.items() if len(ids) > 1}
    duplicate_extra_rows = sum(len(v) - 1 for v in duplicate_groups.values())

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "n_rows": len(out_rows),
        "n_unique_prompts": unique_prompts,
        "n_duplicate_groups": len(duplicate_groups),
        "n_duplicate_extra_rows": duplicate_extra_rows,
        "n_overrides": len(override_rows),
        "override_ids": [row["id"] for row in override_rows],
        "categories": dict(Counter(row.get("category", "") for row in out_rows)),
        "years": dict(Counter(str(row.get("year", "")) for row in out_rows)),
    }

    if not args.check_only:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["output_sha256"] = sha256_file(args.output)

        lines = [
            "# Held-Out Benchmark v2 Audit",
            "",
            f"Generated: `{summary['created_utc']}`",
            "",
            f"- Rows: **{summary['n_rows']}**",
            f"- Unique prompts: **{unique_prompts}**",
            f"- Duplicate groups: **{len(duplicate_groups)}** "
            f"({duplicate_extra_rows} redundant rows)",
            f"- Ground-truth overrides: **{len(override_rows)}**",
            "",
            "## Ground-truth overrides",
            "",
        ]
        if override_rows:
            lines += ["| ID | Original | Corrected | Reason |", "|---|---|---|---|"]
            for row in override_rows:
                reason = row.get("ground_truth_override_reason", "").replace("|", "\\|")
                lines.append(
                    f"| `{row['id']}` | `{row['ground_truth_original']}` | "
                    f"`{row['ground_truth']}` | {reason} |"
                )
        else:
            lines.append("None.")

        lines += [
            "",
            "## Duplicate prompt groups",
            "",
            "Identical prompts in the benchmark are not independent samples. The v2 file "
            "flags them with `duplicate_prompt`, `duplicate_group_size`, and "
            "`duplicate_group_ids`; use a cluster-aware or unique-prompt sensitivity "
            "analysis alongside the primary n=200 analysis.",
            "",
            "| Group size | IDs | Prompt |",
            "|---:|---|---|",
        ]
        for prompt, group_ids in sorted(duplicate_groups.items(), key=lambda x: (-len(x[1]), x[0])):
            short = prompt.replace("|", "\\|")
            lines.append(f"| {len(group_ids)} | {', '.join('`'+i+'`' for i in group_ids)} | {short[:180]} |")

        lines += [
            "",
            "## Output files",
            "",
            f"- `{args.output}`",
            f"- `{args.manifest}`",
            "",
        ]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
