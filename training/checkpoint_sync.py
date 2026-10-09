"""
SymboLM Checkpoint Persistence & Remote Cloud Synchronization
============================================================
Authorship & Principal Architect: Sanish Gyawali
AI Systems Collaborator: Antigravity (Google DeepMind)

Provides a fault-tolerant cloud persistence architecture:
1. Primary Storage Sync: Real-time synchronization to Hugging Face Hub (private repository).
2. Local Pruning for Disk Efficiency: Automatically purges older local checkpoints to 
   guarantee that container disk quotas (Kaggle 20 GB, Colab 30 GB) are never breached.
3. Overriding Mechanism: Flexible CLI flags, environment variables, and safety toggles
   (e.g., prune-after-sync-only verification, force pruning, keep-N limits, and repo overrides).
"""

from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class SyncConfig:
    """Configuration and override settings for checkpoint persistence."""

    hf_repo_id: str = "gyawalisanish0/symboLM-checkpoints"
    hf_token: Optional[str] = None
    private_repo: bool = True
    sync_to_hub: bool = True
    sync_every_n_steps: int = 25
    keep_latest_local: int = 1
    enable_pruning: bool = True
    prune_after_sync_only: bool = True
    disk_threshold_percent: float = 85.0

    @classmethod
    def from_env(cls) -> SyncConfig:
        """Loads configuration with environment variable overrides."""
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or cls._get_cached_hf_token()
        )
        return cls(
            hf_repo_id=os.environ.get("HF_REPO_ID", "gyawalisanish0/symboLM-checkpoints"),
            hf_token=token,
            private_repo=os.environ.get("HF_PRIVATE", "1").lower() not in ("0", "false", "no"),
            sync_to_hub=os.environ.get("ENABLE_HF_SYNC", "1").lower() not in ("0", "false", "no"),
            sync_every_n_steps=int(os.environ.get("SYNC_EVERY_N_STEPS", "25")),
            keep_latest_local=int(os.environ.get("KEEP_LATEST_CHECKPOINTS", "1")),
            enable_pruning=os.environ.get("ENABLE_LOCAL_PRUNING", "1").lower() not in ("0", "false", "no"),
            prune_after_sync_only=os.environ.get("PRUNE_AFTER_SYNC_ONLY", "1").lower() not in ("0", "false", "no"),
            disk_threshold_percent=float(os.environ.get("DISK_THRESHOLD_PERCENT", "85.0")),
        )

    @staticmethod
    def _get_cached_hf_token() -> Optional[str]:
        default_path = Path.home() / ".cache" / "huggingface" / "token"
        if default_path.exists():
            try:
                return default_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return None


class HuggingFaceSyncManager:
    """Manages secure communication and asynchronous uploads to Hugging Face Hub."""

    def __init__(self, config: SyncConfig):
        self.config = config
        self._api = None
        self._authenticated = False
        self._init_client()

    def _init_client(self):
        if not self.config.sync_to_hub:
            return

        if not self.config.hf_token:
            print("[HF Hub Sync] Notice: No HF_TOKEN supplied. Remote Hub sync will be skipped unless configured.")
            return

        try:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.config.hf_token)
            # Ensure repository exists
            self._api.create_repo(
                repo_id=self.config.hf_repo_id,
                private=self.config.private_repo,
                exist_ok=True,
                repo_type="model",
            )
            self._authenticated = True
            print(f"[HF Hub Sync] Authenticated. Primary remote storage target: https://huggingface.co/{self.config.hf_repo_id}")
        except Exception as e:
            print(f"[HF Hub Sync Warning] Initialization failed: {e}")
            self._authenticated = False

    def is_active(self) -> bool:
        return self._authenticated and self._api is not None

    def upload_checkpoint(self, local_path: Path, step_name: str) -> bool:
        """Uploads a specific checkpoint directory to Hugging Face Hub."""
        if not self.is_active():
            return False

        try:
            print(f"\n[HF Hub Sync] Initiating upload for {step_name} -> {self.config.hf_repo_id}...")
            self._api.upload_folder(
                folder_path=str(local_path),
                repo_id=self.config.hf_repo_id,
                path_in_repo=f"checkpoints/{step_name}",
                commit_message=f"SymboLM automated checkpoint: {step_name} (Architect: Sanish Gyawali)",
            )
            print(f"[HF Hub Sync] Successfully synced {step_name} to Hugging Face Hub!")
            return True
        except Exception as e:
            print(f"[HF Hub Sync Error] Failed to upload {step_name}: {e}")
            return False

    def upload_final_adapter(self, adapter_path: Path, commit_message: str = "SymboLM Final Policy Adapter") -> bool:
        """Uploads the final trained policy adapter to root of the repo."""
        if not self.is_active():
            return False

        try:
            print(f"\n[HF Hub Sync] Uploading final policy adapter to {self.config.hf_repo_id}...")
            self._api.upload_folder(
                folder_path=str(adapter_path),
                repo_id=self.config.hf_repo_id,
                path_in_repo="final_policy_adapter",
                commit_message=f"{commit_message} (Architect: Sanish Gyawali)",
            )
            print(f"[HF Hub Sync] Final policy adapter published to https://huggingface.co/{self.config.hf_repo_id}!")
            return True
        except Exception as e:
            print(f"[HF Hub Sync Error] Final adapter upload failed: {e}")
            return False


class LocalDiskPruner:
    """
    Prevents container disk exhaustion (Kaggle 20 GB ceiling) by pruning old checkpoints.
    Features safety override: only prunes if remote sync confirmed, unless forced.
    """

    def __init__(self, config: SyncConfig):
        self.config = config
        self.checkpoint_history: list[dict] = []  # List of {'path': Path, 'synced': bool}

    def register_checkpoint(self, path: Path, synced: bool):
        self.checkpoint_history.append({"path": path, "synced": synced})

    def prune_old_checkpoints(self, force: bool = False):
        if not self.config.enable_pruning and not force:
            return

        # Keep only the last N checkpoints
        while len(self.checkpoint_history) > self.config.keep_latest_local:
            candidate = self.checkpoint_history[0]
            ckpt_path: Path = candidate["path"]
            synced: bool = candidate["synced"]

            # Safety override: if prune_after_sync_only is True, skip deleting unsynced ckpts unless forced
            if self.config.prune_after_sync_only and not synced and not force:
                print(f"[Disk Guard Notice] Retaining {ckpt_path.name} locally because remote sync was not confirmed.")
                break

            self.checkpoint_history.pop(0)
            if ckpt_path.exists():
                try:
                    shutil.rmtree(ckpt_path)
                    print(f"\n[Disk Guard] Pruned {ckpt_path.name} to preserve container quota. (Synced: {synced})")
                except Exception as e:
                    print(f"\n[Disk Guard Warning] Failed to delete {ckpt_path}: {e}")

    def check_emergency_disk_limit(self, output_dir: Path):
        """Emergency disk usage guard: if disk usage exceeds threshold, prune oldest."""
        try:
            usage = shutil.disk_usage(output_dir)
            used_pct = (usage.used / usage.total) * 100.0
            if used_pct >= self.config.disk_threshold_percent:
                print(f"\n[Disk Guard Alert] High disk usage detected ({used_pct:.1f}% >= {self.config.disk_threshold_percent}%). Forcing prune.")
                self.prune_old_checkpoints(force=True)
        except Exception:
            pass


class HuggingFacePersistenceCallback(TrainerCallback):
    """
    Unified Trainer Callback that orchestrates:
    1. Primary remote sync to Hugging Face Hub on save.
    2. Intelligent local pruning with safety overrides.
    3. Final policy upload on train end.
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig.from_env()
        self.hub_manager = HuggingFaceSyncManager(self.config)
        self.pruner = LocalDiskPruner(self.config)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = Path(args.output_dir)
        step_name = f"checkpoint-{state.global_step}"
        current_ckpt = output_dir / step_name

        if not current_ckpt.exists():
            return

        synced = False
        # Sync if interval matches
        if state.global_step % self.config.sync_every_n_steps == 0:
            synced = self.hub_manager.upload_checkpoint(current_ckpt, step_name)

        # Register and prune
        self.pruner.register_checkpoint(current_ckpt, synced=synced)
        self.pruner.check_emergency_disk_limit(output_dir)
        self.pruner.prune_old_checkpoints()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = Path(args.output_dir)
        print("\n[HuggingFacePersistenceCallback] Training finished. Performing final artifact packaging & sync...")

        # Package local tarball for backup
        tar_target = output_dir.parent / "symboLM_final_policy.tar.gz"
        try:
            package_adapter_tarball(output_dir, tar_target)
        except Exception as e:
            print(f"[Archive Notice] Tarball creation skipped: {e}")

        # Publish final adapter to Hugging Face Hub
        if output_dir.exists():
            self.hub_manager.upload_final_adapter(output_dir)


def package_adapter_tarball(adapter_dir: Path | str, output_tar: Path | str) -> Path:
    """
    Compresses an adapter checkpoint directory into a deployment-ready tarball.
    """
    adapter_path = Path(adapter_dir)
    tar_path = Path(output_tar)
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "w:gz") as tar:
        for file in adapter_path.iterdir():
            tar.add(file, arcname=file.name)

    print(f"[Archive] Packaged {adapter_path} -> {tar_path} ({tar_path.stat().st_size / 1e6:.2f} MB)")
    return tar_path
