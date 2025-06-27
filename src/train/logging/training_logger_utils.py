# src/utils/training_logger_utils.py

from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
from torchvision.utils import save_image

from .log import log_grid_images, log_table_wandb
from ...utils.visualization import save_side_by_side_images


def log_training_start(logger, model_name, experiment_id, run_id, task, total_epochs, total_batches):
    logger.info(
        f"🚀 Training started\n"
        f"🔧 Model: {model_name}\n"
        f"🧪 Experiment: {experiment_id}\n"
        f"🏃‍♂️ Run ID: {run_id}\n"
        f"🤞🏽 Task: {task}\n"
        f"🖋️ Experiment Name: experiment_{experiment_id}_{task}\n"
        f"⏳ Total Epochs: {total_epochs}, Batches/Epoch: {total_batches}"
    )

def log_epoch_start(epoch, logger):
    logger.info(f"\n--- Epoch {epoch+1} Start ---")

def log_epoch_summary(logger, epoch, total_epochs, train_loss, val_loss=None, epoch_time=None):
    msg = f"Epoch {epoch}/{total_epochs} completed | Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    if epoch_time is not None:
        msg += f" | Time: {timedelta(seconds=int(epoch_time))}"
    logger.info(msg)



def log_training_end(logger, training_time):
    logger.info(
        f"🏁 Training finished!\n"
        f"🕒 Total Duration: {training_time:.2f}s"
    )


def log_batch_progress(logger, epoch, total_epochs, step, total_steps, loss, lr, time_elapsed, gpu_mem=None):
    hrs, rem = divmod(int(time_elapsed), 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    msg = (
        f"[Epoch {epoch}/{total_epochs}] "
        f"Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed_str}"
    )
    if gpu_mem is not None:
        msg += f" | GPU_mem: {gpu_mem:.0f} MB"
    logger.info(msg)

def log_training_stats(logger, epoch, avg_loss, duration, optimizer, writer=None, wandb_tracker=None):
    lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"【Epoch {epoch}】 Avg Loss: {avg_loss:.4f} | Duration: {timedelta(seconds=int(duration))} | LR: {lr:.2e}"
    )
    if writer:
        writer.add_scalar("train/avg_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/epoch_duration", duration, epoch)
    if wandb_tracker:
        wandb_tracker.log({
            "train/avg_loss": avg_loss,
            "train/lr": lr,
            "train/epoch_duration": duration
        }, step=epoch)


def log_batch(step, loss, lr, logger, writer=None, wandb_tracker=None):
    logger.info(f"[Step {step:6d}] loss={loss:.4f}  lr={lr:.6f}")
    mem_mb = 0.0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.debug(f"GPU memory allocated: {mem_mb:.1f} MB")
    if writer:
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/lr", lr, step)
        writer.add_scalar("train/memory_MB", mem_mb, step)
    if wandb_tracker:
        wandb_tracker.log({"train/loss":  loss, "train/lr":lr, "train/memory_MB":  mem_mb}, step=step)

def log_json(logger, message, **extra_data):
    logger.info(message, extra={"extra_data": extra_data})

def log_exception(logger, epoch=None, step=None, exception=None):
    msg = "❌ Exception during training"
    if epoch is not None:
        msg += f" at Epoch {epoch}"
    if step is not None:
        msg += f", Step {step}"
    logger.exception(f"{msg}: {str(exception)}")

def visualize_epoch(generated: torch.Tensor, real: torch.Tensor,  dirs: Dict, epoch: int=None, prefix: str = "generated", wandb_run = None, writer = None):
    samples_dir: Path = dirs.get("results_samples") / Path(f"epoch_{epoch}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    # --- 1) Save each generated image individually ---
    for i, img in enumerate(generated):
        save_image(img,  samples_dir / Path(f"{prefix}_{i:03d}.png"), normalize=True, value_range=(0, 1))

    # --- 2) Side-by-side comparisons (one per index) ---
    side_by_side_dir: Path = dirs.get("results_side_by_side") / Path(f"epoch_{epoch}")
    side_by_side_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side_images(real, generated, side_by_side_dir)

    # --- 3) TensorBoard logging ---
    if writer:
        log_grid_images(generated, real, epoch, writer)

    # --- 4) W&B logging ---
    if wandb_run:
        log_table_wandb(generated, real, epoch, wandb_run)
