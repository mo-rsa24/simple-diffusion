from pathlib import Path
from datetime import timedelta
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.composable_diffusion import ComposableDiffusionModel
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_batch, log_training_stats, log_training_end, \
    visualize_epoch
from src.utils.calculations import q_sample
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.sampling import composable_expert_sampler

def train(
    cfg,
    dirs: dict,
    model: ComposableDiffusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger,
    device,
    writer=None,
    wandb_run=None,
):
    optimizer = Adam(model.parameters(), **cfg.optimizer.params)

    beta_schedule = getattr(cfg.diffusion, "beta_schedule", "linear")
    _ = beta_schedule  # placeholder to show usage

    start_time = time.time()
    log_training_start(
        logger,
        model_name="ComposableDiffusionModel",
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader),
    )

    ckpt_mgr = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt", Path(cfg.dirs.ckpt_dir)), logger=logger)
    global_step = 0
    start_epoch = 1
    if cfg.training.resume_from:
        try:
            model, optimizer, _, last_epoch, global_step = ckpt_mgr.load_latest(model, optimizer, None, map_location=device)
            start_epoch = last_epoch + 1
            logger.info(f"Resuming from epoch {last_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                x = batch["image"].to(device)
                noise = torch.randn_like(x)
                t = torch.randint(0, cfg.diffusion.timesteps, (x.size(0),), device=device).long()
                x_noisy = q_sample(x, t, noise, cfg.diffusion.timesteps, cfg.diffusion.beta_start, cfg.diffusion.beta_end)
                optimizer.zero_grad(set_to_none=True)
                out = model(x_noisy, t)
                pred_noise_shape = out["shape"]
                pred_noise_color = out["color"]
                pred_noise_box = out["box"]

                # Get the masks from the batch
                digit_mask = batch['digit_mask'].to(device)
                bbox_mask = batch['bbox_mask'].to(device)

                # --- Calculate the loss for each expert on its specific region ---

                foreground_mask = (digit_mask + bbox_mask).clamp(0, 1)
                loss_shape = (F.mse_loss(pred_noise_shape, noise,
                                         reduction='none') * foreground_mask).sum() / foreground_mask.sum()
                loss_color = (F.mse_loss(pred_noise_color, noise,
                                         reduction='none') * digit_mask).sum() / digit_mask.sum()
                loss_box = (F.mse_loss(pred_noise_box, noise, reduction='none') * bbox_mask).sum() / bbox_mask.sum()

                # --- CHANGE: Add loss for the merged output ---
                pred_noise_merged = out["merged"]
                # This loss is unmasked, training the merge layers on the whole image
                loss_merged = F.mse_loss(pred_noise_merged, noise)

                # --- CHANGE: Update the total loss ---
                # Add the merged loss. You can add a weighting factor (e.g., 0.5) if needed.
                loss = loss_shape + loss_color + loss_box + loss_merged

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 2e-4), logger, writer=writer, wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            avg_loss = running_loss / len(train_loader)
            duration = time.time() - epoch_start
            log_training_stats(logger, epoch, avg_loss, duration, optimizer, writer=writer, wandb_tracker=wandb_run)

            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                real_batch = next(iter(val_loader))
                real_batch = real_batch['image'].to(device)

                generated = composable_expert_sampler(model, cfg, device)
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run = wandb_run, writer = writer)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:
        ckpt_mgr.save(model, optimizer, None, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(cfg.experiment_id, cfg.run_id, total_epochs=epoch,
                     duration=str(timedelta(seconds=int(total_time))))