import torch
import torch.nn.functional as F
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict
from box import Box
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.models.vanilla.composable_unet import ComposableUnet
from src.models.vanilla.ema import EMA
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.utils.calculations import q_sample
from src.utils.checkpoint_manager import CheckpointManager
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    visualize_epoch, log_json, log_training_end
from src.utils.sampling import composable_expert_sampler, composable_expert_sampler2  # Import the new sampler


def train(cfg: Box, dirs: Dict, model: ComposableUnet, ema: EMA, train_loader: DataLoader, logger, device, writer=None,
          wandb_run=None):
    """
    Main training loop for the ComposableUnet model.
    """
    model.train()
    if cfg.optimizer.type.lower() == "adam":
        optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")

    scheduler = None  # Add scheduler if needed
    scaler = GradScaler()
    start_time = time.time()
    log_training_start(logger, model_name="Composable Diffusion Model", experiment_id=cfg.experiment_id,
                       run_id=cfg.run_id, task=cfg.task, total_epochs=cfg.training.epochs,
                       total_batches=len(train_loader))

    checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt", Path(cfg.dirs.ckpt_dir)),
                                           logger=logger)
    global_step = 0
    start_epoch = 1

    if cfg.training.resume_from:
        try:
            model, optimizer, scheduler, last_epoch, global_step = checkpoint_manager.load_latest(model, optimizer,
                                                                                                  scheduler,
                                                                                                  map_location=device)
            start_epoch = last_epoch + 1
            logger.info(f"Resuming from epoch {last_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_start = time.time()
            log_epoch_start(epoch - 1, logger)

            running_loss = 0.0
            for step, train_batch in enumerate(train_loader, 1):
                optimizer.zero_grad()

                # --- Extract Data, Labels, and Masks ---
                batch = train_batch['image'].to(device)
                digit_labels = train_batch['digit_label'].to(device)
                digit_color_labels = train_batch['color_label'].to(device)
                bbox_color_labels = train_batch['bbox_label'].to(device)
                digit_masks = train_batch['digit_mask'].to(device)
                bbox_masks = train_batch['bbox_mask'].to(device)

                # Create background mask by finding pixels that are neither digit nor bbox
                background_masks = (1.0 - digit_masks - bbox_masks).clamp(min=0.0)

                # Stack masks for each expert: [digit, bbox, background]
                # Shape: (B, num_experts, 1, H, W)
                masks = torch.stack([digit_masks, bbox_masks, background_masks], dim=1)

                batch_size = batch.shape[0]
                noise = torch.randn_like(batch)
                t = torch.randint(0, cfg.diffusion.timesteps, (batch_size,), device=device).long()

                # --- Classifier-Free Guidance Training ---
                # Randomly drop conditions with a certain probability
                if cfg.training.get('cond_drop_prob', 0.1) > 0:
                    cond_mask = torch.rand(batch_size, device=device) < cfg.training.cond_drop_prob
                    # Use 10 as the null/unconditional class index
                    digit_labels[cond_mask] = 10
                    digit_color_labels[cond_mask] = 10
                    bbox_color_labels[cond_mask] = 10

                x_noisy = q_sample(x_start=batch, t=t, noise=noise, **cfg.diffusion)

                with autocast():
                    # --- Get Expert Predictions ---
                    pred_noise_experts_raw = model(x_noisy, t, digit_labels, digit_color_labels, bbox_color_labels)

                    # Reshape to (B, num_experts, C, H, W) to separate experts
                    b, _, h, w = pred_noise_experts_raw.shape
                    c = model.channels
                    pred_noise_experts = pred_noise_experts_raw.view(b, model.num_experts, c, h, w)

                    # --- Compute MASKED LOSS ---
                    # Expand noise to match expert predictions shape for broadcasting
                    target_noise = noise.unsqueeze(1).repeat(1, model.num_experts, 1, 1, 1)

                    # Apply masks to both predictions and targets
                    masked_preds = pred_noise_experts * masks
                    masked_targets = target_noise * masks

                    # Calculate loss on the masked regions
                    if cfg.diffusion.loss_type == 'l1':
                        loss = F.l1_loss(masked_preds, masked_targets)
                    elif cfg.diffusion.loss_type == 'l2':
                        loss = F.mse_loss(masked_preds, masked_targets)
                    else:  # huber
                        loss = F.smooth_l1_loss(masked_preds, masked_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update()

                running_loss += loss.item()
                global_step += 1

                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss.item(), cfg.optimizer.params.get("lr"), logger, writer=writer,
                              wandb_tracker=wandb_run)

            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time)

            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                # Create some fixed labels for consistent visualization
                vis_digits = torch.arange(10, device=device)
                vis_digit_colors = torch.randint(0, 10, (10,), device=device)
                vis_bbox_colors = torch.randint(0, 10, (10,), device=device)

                generated = composable_expert_sampler2(
                    model, cfg, device,
                    digit_labels=vis_digits,
                    digit_color_labels=vis_digit_colors,
                    bbox_color_labels=vis_bbox_colors,
                    guidance_scale=cfg.sampling.guidance_scale
                )
                real_batch = next(iter(train_loader))['image'].to(device)
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run=wandb_run, writer=writer)
                model.train()

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}", exc_info=True)
        checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
        raise

    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )
