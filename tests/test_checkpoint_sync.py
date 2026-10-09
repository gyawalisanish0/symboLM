"""
Unit tests for SymboLM Checkpoint Persistence & Pruning Engine
Authorship: Sanish Gyawali
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.checkpoint_sync import (
    LocalDiskPruner,
    SyncConfig,
    package_adapter_tarball,
)


def test_pruning_keeps_latest_and_prunes_older():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = SyncConfig(keep_latest_local=1, enable_pruning=True, prune_after_sync_only=False)
        pruner = LocalDiskPruner(cfg)

        ckpt1 = tmp_path / "checkpoint-25"
        ckpt1.mkdir()
        (ckpt1 / "adapter_model.safetensors").write_text("dummy1")
        pruner.register_checkpoint(ckpt1, synced=True)
        pruner.prune_old_checkpoints()
        assert ckpt1.exists()

        ckpt2 = tmp_path / "checkpoint-50"
        ckpt2.mkdir()
        (ckpt2 / "adapter_model.safetensors").write_text("dummy2")
        pruner.register_checkpoint(ckpt2, synced=True)
        pruner.prune_old_checkpoints()

        # ckpt1 should have been pruned, ckpt2 remains
        assert not ckpt1.exists()
        assert ckpt2.exists()


def test_safety_override_prune_after_sync_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = SyncConfig(keep_latest_local=1, enable_pruning=True, prune_after_sync_only=True)
        pruner = LocalDiskPruner(cfg)

        ckpt1 = tmp_path / "checkpoint-25"
        ckpt1.mkdir()
        pruner.register_checkpoint(ckpt1, synced=False)  # NOT synced

        ckpt2 = tmp_path / "checkpoint-50"
        ckpt2.mkdir()
        pruner.register_checkpoint(ckpt2, synced=True)

        pruner.prune_old_checkpoints(force=False)
        # ckpt1 was NOT synced, so it must be protected from deletion
        assert ckpt1.exists()
        assert ckpt2.exists()

        # When forced, ckpt1 should be deleted
        pruner.prune_old_checkpoints(force=True)
        assert not ckpt1.exists()
        assert ckpt2.exists()


def test_disable_pruning_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = SyncConfig(keep_latest_local=1, enable_pruning=False)
        pruner = LocalDiskPruner(cfg)

        ckpt1 = tmp_path / "checkpoint-25"
        ckpt1.mkdir()
        pruner.register_checkpoint(ckpt1, synced=True)

        ckpt2 = tmp_path / "checkpoint-50"
        ckpt2.mkdir()
        pruner.register_checkpoint(ckpt2, synced=True)

        pruner.prune_old_checkpoints()
        # Both must exist because pruning is disabled
        assert ckpt1.exists()
        assert ckpt2.exists()


def test_package_adapter_tarball():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"test": true}')
        (adapter_dir / "adapter_model.safetensors").write_text("weights")

        tar_file = tmp_path / "test_policy.tar.gz"
        out = package_adapter_tarball(adapter_dir, tar_file)
        assert out.exists()
        assert out.stat().st_size > 0


if __name__ == "__main__":
    test_pruning_keeps_latest_and_prunes_older()
    test_safety_override_prune_after_sync_only()
    test_disable_pruning_override()
    test_package_adapter_tarball()
    print("ALL CHECKPOINT SYNC & PRUNING UNIT TESTS PASSED!")
