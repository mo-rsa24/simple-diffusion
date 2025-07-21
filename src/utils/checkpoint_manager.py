# CheckpointManager.py

import os
import json
import glob
from pathlib import Path
import random
import numpy as np
import torch
import subprocess
from datetime import datetime
from typing import Optional, Tuple, Any

def _get_git_commit_hash() -> str:
    """Return the short git commit hash if available."""
    try:
        return subprocess.check_output([
            "git",
            "rev-parse",
            "--short",
            "HEAD",
        ]).decode().strip()
    except Exception:
        return "unknown"

class CheckpointManager:
    """
    Manages saving, loading, and rotating model checkpoints.

    Checkpoints are stored under:
        <checkpoint_dir>/checkpoints/
    Each checkpoint is named: epoch_{epoch:03d}.ckpt

    A log file (log.json) in the checkpoints folder records metadata for each saved checkpoint.

    Usage:
        manager = CheckpointManager(
            run_id="run_01",
            checkpoint_dir="runs/exp_01/run_01",
            keep_last_k=3,
            logger=logger
        )
        manager.save(model, optimizer, scheduler, epoch=5, global_step=1000)
        model, optimizer, scheduler, epoch, global_step = manager.load_latest(
            model, optimizer, scheduler, map_location=device
        )
    """

    def __init__(
        self,
        run_id: str,
        checkpoint_dir: Path,
        keep_last_k: int = 3,
        logger: Any = None,
    ):
        """
        Args:
            run_id (str): Unique identifier for this training run.
            checkpoint_dir (str): Base directory under which checkpoints will be stored.
            keep_last_k (int): Number of most recent checkpoints to keep; older ones will be deleted.
            logger: Logging object for informational messages (expects .info(), .warning(), .error()).
        """
        self.run_id = run_id
        self.ckpt_dir = checkpoint_dir
        self.keep_last_k = int(keep_last_k)
        if self.keep_last_k < 1:
            raise ValueError(f"keep_last_k must be ≥ 1, got {self.keep_last_k}")

        self.logger = logger

        # Log file path
        self.log_path = os.path.join(self.ckpt_dir, "log.json")
        if not os.path.isfile(self.log_path):
            # Initialize empty log
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def _write_log(self, entry: dict):
        """Append an entry to the log.json file."""
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = []

        data.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _list_checkpoints(self) -> list:
        """
        List all checkpoint files and return a sorted list of tuples:
            [(epoch_int, filepath), ...], sorted ascending by epoch_int.
        """
        pattern = os.path.join(self.ckpt_dir, "epoch_*.ckpt")
        files = glob.glob(pattern)
        ckpts = []
        for fp in files:
            basename = os.path.basename(fp)
            try:
                # Expect format: epoch_{epoch:03d}.ckpt
                epoch_str = basename.split("_")[1].split(".")[0]
                epoch = int(epoch_str)
                ckpts.append((epoch, fp))
            except Exception:
                continue
        ckpts.sort(key=lambda x: x[0])
        return ckpts

    def _delete_checkpoint(self, filepath: str):
        """Delete a checkpoint file and remove from log.json."""
        try:
            os.remove(filepath)
            if self.logger:
                self.logger.info(f"🗑️ Deleted old checkpoint: {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Could not delete checkpoint {filepath}: {e}")

        # Also remove from log.json
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = []

        # Filter out entries matching this filepath
        updated = [entry for entry in data if entry.get("filepath") != filepath]
        with open(self.log_path, "w") as f:
            json.dump(updated, f, indent=2)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: Optional[int] = None,
    ):
        """
        Save model, optimizer, and scheduler state at given epoch/global_step.
        Applies rotation to keep only the most recent `keep_last_k` checkpoints.

        Args:
            model: PyTorch model whose state_dict will be saved.
            optimizer: Optimizer whose state_dict will be saved (optional).
            scheduler: LR scheduler whose state_dict will be saved (optional).
            epoch (int): Current epoch number (used for naming).
            global_step (int): Current global training step (optional).
        """
        filename = f"epoch_{epoch:03d}.ckpt"
        filepath = os.path.join(self.ckpt_dir, filename)

        # Build checkpoint dict
        checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch}
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if global_step is not None:
            checkpoint["global_step"] = global_step
        # RNG states for deterministic resume
        checkpoint["torch_rng"] = torch.get_rng_state()
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        checkpoint["numpy_rng"] = np.random.get_state()
        checkpoint["python_rng"] = random.getstate()
        # Save to disk
        try:
            torch.save(checkpoint, filepath)
            if self.logger:
                self.logger.info(f"💾 Saved checkpoint: {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to save checkpoint at epoch {epoch}: {e}")
            return

        # Append to log.json
        entry = {
            "filepath": filepath,
            "epoch": epoch,
            "timestamp": datetime.utcnow().isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _get_git_commit_hash(),
        }
        if global_step is not None:
            entry["global_step"] = global_step
        self._write_log(entry)

        # Rotate old checkpoints if necessary
        all_ckpts = self._list_checkpoints()
        if len(all_ckpts) > self.keep_last_k:
            # remove older epochs
            to_delete = all_ckpts[:-self.keep_last_k]
            for old_epoch, old_fp in to_delete:
                self._delete_checkpoint(old_fp)

    def _find_previous(self, current_epoch: int) -> Optional[str]:
        """
        Given a current epoch integer, return the filepath of the next-lower epoch checkpoint.
        Returns None if no previous found.
        """
        ckpts = self._list_checkpoints()
        # Extract epochs only
        epochs = [e for e, _ in ckpts]
        if current_epoch not in epochs:
            return None
        idx = epochs.index(current_epoch)
        if idx == 0:
            return None
        return ckpts[idx - 1][1]

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], int, Optional[int]]:
        """
        Load the most recent valid checkpoint. If loading the latest fails, fallback to previous.

        Args:
            model: Model to load state_dict into.
            optimizer: Optimizer to load state_dict into (optional).
            scheduler: Scheduler to load state_dict into (optional).
            map_location: Device mapping for torch.load (e.g. {'cuda:0':'cpu'} or 'cpu').

        Returns:
            (model, optimizer, scheduler, epoch_loaded, global_step_loaded)
        """
        ckpts = self._list_checkpoints()
        if not ckpts:
            raise FileNotFoundError("No checkpoints found to load.")

        # Attempt to load from latest down to oldest
        for epoch, fp in reversed(ckpts):
            try:
                return self.load_checkpoint(fp, model, optimizer, scheduler, map_location)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Failed to load checkpoint {fp}: {e}")
                # Try next-older checkpoint
                prev_fp = self._find_previous(epoch)
                if prev_fp is None:
                    break
                continue

        raise RuntimeError("No valid checkpoint could be loaded (all failed).")

    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], int, Optional[int]]:
        """
        Load a specific checkpoint file into model, optimizer, scheduler.

        Args:
            filepath: Path to the .ckpt file.
            model: Model to load into.
            optimizer: Optimizer to load into (optional).
            scheduler: Scheduler to load into (optional).
            map_location: Device mapping for torch.load.

        Returns:
            (model, optimizer, scheduler, epoch_loaded, global_step_loaded)
        Raises:
            Exception if file is missing, corrupted, or missing keys.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # Load checkpoint dict
        checkpoint = torch.load(filepath, map_location='cpu')

        # Validate keys
        if "model_state_dict" not in checkpoint or "epoch" not in checkpoint:
            raise KeyError(f"Checkpoint {filepath} is missing required keys.")

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer & scheduler if provided
        global_step = checkpoint.get("global_step", None)
        epoch_loaded = checkpoint["epoch"]

        if optimizer is not None:
            if "optimizer_state_dict" not in checkpoint:
                raise KeyError(f"Checkpoint {filepath} missing optimizer state.")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            if "scheduler_state_dict" not in checkpoint:
                raise KeyError(f"Checkpoint {filepath} missing scheduler state.")
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore RNG states if present
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"])
        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            cuda_rng_states = checkpoint["cuda_rng"]
            if len(cuda_rng_states) != torch.cuda.device_count():
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Checkpoint RNG states ({len(cuda_rng_states)}) do not match available GPUs ({torch.cuda.device_count()}). Skipping CUDA RNG restoration.")
            else:
                # The state must be a ByteTensor (uint8) on the correct CUDA device.
                # Since we loaded to CPU, we must move it back.
                try:
                    states_on_cuda = [state.to(f'cuda:{i}', dtype=torch.uint8) for i, state in
                                      enumerate(cuda_rng_states)]
                    torch.cuda.set_rng_state_all(states_on_cuda)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"⚠️ Could not restore CUDA RNG state due to an error: {e}. Skipping.")

        if "numpy_rng" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])

        if self.logger:
            self.logger.info(f"✅ Loaded checkpoint '{filepath}' at epoch {epoch_loaded}")

        return model, optimizer, scheduler, epoch_loaded, global_step