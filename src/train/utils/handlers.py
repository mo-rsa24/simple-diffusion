import os

import torch

from src.monitoring.email_alert_mailtrap import alert_on_failure
from src.monitoring.fail_safe_guard import fail_safe_guard
from src.train.logging.training_logger_utils import log_json, log_grad_norms, log_batch_progress, \
    log_training_stats, log_epoch_summary


def _save_checkpoint(epoch, ckpt_manager, model, optimizer, scheduler, global_step, logger, wandb_tracker, save_path):
    ckpt_manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch + 1,
        global_step=global_step,
    )
    ckpt_path = os.path.join(save_path, "checkpoints", f"epoch_{epoch+1:03d}.ckpt")
    log_json(logger, "💾 Checkpoint saved", epoch=epoch+1, path=ckpt_path, lr=optimizer.param_groups[0]["lr"])
    if wandb_tracker:
        wandb_tracker.save(ckpt_path)


def log_batch(epoch, num_epochs, batch_idx, num_batches, loss, optimizer, logger, model, writer, wandb_tracker, global_step, device, config):
    if (batch_idx + 1) % int(config["training"].get("log_interval", 100)) == 0:
        log_batch_progress(
                logger, epoch=epoch+1, total_epochs=num_epochs,
                step=batch_idx+1, total_steps=num_batches, loss=loss,
                lr=optimizer.param_groups[0]["lr"],
                time_elapsed=0,  # Or actual time if desired
                gpu_mem=(torch.cuda.memory_allocated(device)/(1024**2)) if device.type == "cuda" else 0.0
            )
        log_grad_norms(logger, model, epoch=epoch+1, step=batch_idx+1, writer=writer, wandb_tracker=wandb_tracker)
        if wandb_tracker:
            wandb_tracker.log({
                "train/batch_loss": loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/gpu_mem_mb": (torch.cuda.memory_allocated(device)/(1024**2)) if device.type == "cuda" else 0.0
            }, step=global_step)

def log_epoch(epoch, num_epochs, avg_train_loss, optimizer, logger, writer, wandb_tracker, epoch_duration):
    log_epoch_summary(logger, epoch=epoch+1, total_epochs=num_epochs, train_loss=avg_train_loss, val_loss=None, epoch_time=epoch_duration)
    log_training_stats(epoch=epoch+1, avg_loss=avg_train_loss, duration=epoch_duration, optimizer=optimizer, logger=logger, writer=writer, wandb_tracker=wandb_tracker)
