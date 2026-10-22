#!/usr/bin/env python
"""
Build a self-contained Kaggle dataset bundle for the v3 SFT kernel.

The bundle is optional. The kernel can clone the pinned git tag instead. A
bundle is useful when the branch/tag is not public yet, or when you want the
kernel inputs to be immutable even if the repository moves.

Output:
    experiments/kaggle/sft_v3/upload/
        dataset-metadata.json
        bundle_manifest.json
        repo_bundle.tar.gz
        README.md

Upload (after reviewing the bundle):
    kaggle datasets create -p experiments/kaggle/sft_v3/upload

Then add the dataset to kernel-metadata.json:
    "dataset_sources": ["sanishgyawali/symbolm-v3-sft-bundle"]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "kaggle" / "sft_v3" / "upload"
BUNDLE_NAME = "repo_bundle.tar.gz"
DATASET_ID = "sanishgyawali/symbolm-v3-sft-bundle"

EXCLUDE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "checkpoints",
    "vendor",
    "assets",
    "results",
    "scratch",
    "eval_cross_domain",
    "upload",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".part"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def tar_filter(info: tarfile.TarInfo):
    path = Path(info.name)
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return None
    if path.suffix in EXCLUDE_SUFFIXES:
        return None
    return info


def add_path(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    archive.add(path, arcname=arcname, filter=tar_filter)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    tar_path = OUT / BUNDLE_NAME
    with tarfile.open(tar_path, "w:gz") as archive:
        add_path(archive, ROOT / "config.py", "symboLM/config.py")
        add_path(archive, ROOT / "README.md", "symboLM/README.md")
        add_path(
            archive,
            ROOT / "PREREG_VALUATION_V2.md",
            "symboLM/PREREG_VALUATION_V2.md",
        )
        add_path(archive, ROOT / "pyproject.toml", "symboLM/pyproject.toml")
        add_path(archive, ROOT / "requirements.txt", "symboLM/requirements.txt")
        add_path(archive, ROOT / "symbolm.gbnf", "symboLM/symbolm.gbnf")
        add_path(archive, ROOT / "symbolic", "symboLM/symbolic")
        add_path(archive, ROOT / "training", "symboLM/training")
        # Do not ship the whole docs/ tree.  It contains grant/application
        # material that must not be uploaded to a Kaggle dataset.  Only the
        # current evidence and redirect notes are needed for context.
        for name in ("EVIDENCE_STATUS.md", "RESEARCH_REDIRECT.md"):
            add_path(archive, ROOT / "docs" / name, f"symboLM/docs/{name}")
        add_path(
            archive,
            ROOT / "experiments" / "kaggle" / "sft_v3",
            "symboLM/experiments/kaggle/sft_v3",
        )

        # Keep only code plus the canonical SFT/benchmark artifacts.  The
        # original 5,000-row curriculum is not needed to train or verify the
        # canonical split.
        add_path(
            archive,
            ROOT / "data" / "symbolic_dataset.py",
            "symboLM/data/symbolic_dataset.py",
        )
        for name in (
            "workflow_dataset.py",
            "build_concept_math_dataset.py",
            "build_contrastive_curriculum.py",
        ):
            path = ROOT / "data" / name
            if path.exists():
                add_path(archive, path, f"symboLM/data/{name}")
        for name in (
            "manifest.json",
            "train.jsonl",
            "val.jsonl",
        ):
            add_path(archive, ROOT / "data" / "research_sft" / name, f"symboLM/data/research_sft/{name}")
        for name in (
            "held_out_eval_200_v2.jsonl",
            "held_out_eval_200_v2_manifest.json",
            "held_out_eval_200_v2_audit.md",
            "ground_truth_overrides.json",
        ):
            path = ROOT / "data" / name
            if path.exists():
                add_path(archive, path, f"symboLM/data/{name}")

        for name in ("strict_scorer.py", "robust_eval.py"):
            path = ROOT / "eval" / name
            if path.exists():
                add_path(archive, path, f"symboLM/eval/{name}")
        for name in ("prepare_research_sft.py", "audit_ground_truth.py"):
            path = ROOT / "scripts" / name
            if path.exists():
                add_path(archive, path, f"symboLM/scripts/{name}")

    sft_manifest = json.loads(
        (ROOT / "data" / "research_sft" / "manifest.json").read_text(encoding="utf-8")
    )
    bundle_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "repo_bundle": BUNDLE_NAME,
        "repo_bundle_sha256": sha256_file(tar_path),
        "repo_bundle_bytes": tar_path.stat().st_size,
        "data": {
            "train_sha256": sft_manifest.get("train_sha256"),
            "val_sha256": sft_manifest.get("val_sha256"),
            "source_sha256": sft_manifest.get("source_sha256"),
            "train_rows": sft_manifest.get("train"),
            "val_rows": sft_manifest.get("val"),
        },
        "expected_benchmark_sha256": (
            "95E2B237AE2B724DC0856AEE7220EBD79436F91630F77BBB3BAD0913127F10ED"
        ),
        "notes": (
            "Extract repo_bundle.tar.gz into /kaggle/working; it creates "
            "/kaggle/working/symboLM. The kernel verifies the tar hash, "
            "SFT hashes, token prefix, label masks, and real EOS before training."
        ),
    }
    write_json(OUT / "bundle_manifest.json", bundle_manifest)

    dataset_metadata = {
        "title": "SymboLM v3 SFT Bundle",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }
    write_json(OUT / "dataset-metadata.json", dataset_metadata)

    readme = f"""# SymboLM v3 SFT Bundle

Generated: {bundle_manifest['created_utc']}
Git head: `{bundle_manifest['git_head']}`
Archive SHA-256: `{bundle_manifest['repo_bundle_sha256']}`

This dataset contains a snapshot of the code and canonical SFT data needed by
the v3 SFT kernel. It is optional: the kernel can clone the pinned git tag
instead.

## Upload

Review the contents first, then:

```bash
kaggle datasets create -p experiments/kaggle/sft_v3/upload
```

To update an existing dataset:

```bash
kaggle datasets version -p experiments/kaggle/sft_v3/upload -m "v3 SFT bundle"
```

After uploading, add this to `kernel-metadata.json`:

```json
"dataset_sources": ["{DATASET_ID}"]
```

The kernel will prefer `/kaggle/input/symbolm-v3-sft-bundle/repo_bundle.tar.gz`
when it is mounted, and otherwise clone the pinned git tag.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"[OK] Wrote {tar_path} ({tar_path.stat().st_size / (1024**2):.2f} MiB)")
    print(f"[OK] Archive SHA-256: {bundle_manifest['repo_bundle_sha256']}")
    print(f"[OK] Git head: {bundle_manifest['git_head']}")
    print(f"[OK] Upload directory: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
