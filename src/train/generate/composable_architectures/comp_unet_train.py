from pathlib import Path
import time
from datetime import timedelta
from typing import Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.train.logging.training_logger_utils import (
    log_training_start, log_epoch_start, log_batch, log_epoch_summary,
    visualize_epoch, log_json, log_training_end
)
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.utils.checkpoint_manager import CheckpointManager
from src.models.vanilla.diffusion import generate_batch
from src.utils.calculations import q_sample
from src.models.vanilla.ema import EMA
from src.utils.sampling import ddpm_sampler, sample_compositional_unet


def train(cfg, dirs: Dict, model, train_loader: DataLoader, val_loader: DataLoader, logger, device, writer=None, wandb_run=None):
    optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    start_time = time.time()
    log_training_start(
        logger,
        model_name="CompositionalUNet",
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader),
    )

    ckpt_mgr = CheckpointManager(run_id=cfg.run_id,
                                 checkpoint_dir=dirs.get("ckpt", Path(cfg.dirs.ckpt_dir)),
                                 logger=logger)
    global_step = 0
    start_epoch = 1
    if cfg.training.resume_from:
        try:
            model, optimizer, _, last_epoch, global_step = ckpt_mgr.load_latest(
                model, optimizer, None, map_location=device
            )
            start_epoch = last_epoch + 1
            logger.info(f"Resuming from epoch {last_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_start = time.time()
            log_epoch_start(epoch - 1, logger)
            running_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                x = batch["image"].to(device)
                bsz = x.size(0)
                noise = torch.randn_like(x)
                t = torch.randint(0, cfg.diffusion.timesteps, (bsz,), device=device).long()
                x_noisy = q_sample(x, t, noise,
                                   timesteps=cfg.diffusion.timesteps,
                                   beta_start=cfg.diffusion.beta_start,
                                   beta_end=cfg.diffusion.beta_end)

                # --- Define Concepts (One-hot vectors) ---
                # Assuming concept_dim=3 for the GatingNetwork
                # Shape=[1,0,0], Color=[0,1,0], Box=[0,0,1]
                shape_concept = F.one_hot(torch.tensor([0]), num_classes=3).float().squeeze(0).to(device)
                color_concept = F.one_hot(torch.tensor([1]), num_classes=3).float().squeeze(0).to(device)
                box_concept = F.one_hot(torch.tensor([2]), num_classes=3).float().squeeze(0).to(device)
                concepts = [shape_concept, color_concept, box_concept]

                # --- Forward Pass ---
                optimizer.zero_grad(set_to_none=True)
                # The model returns a dictionary of predictions
                noise_pred_dict = model(x_noisy, t, concepts)

                # --- Masked Loss Calculation (Crucial Part) ---
                # Get masks from the batch
                digit_mask = batch['digit_mask'].to(device)
                bbox_mask = batch['bbox_mask'].to(device)
                foreground_mask = (digit_mask + bbox_mask).clamp(0, 1)

                # Calculate loss for each expert on its specific region
                loss_shape = (F.mse_loss(noise_pred_dict["shape"], noise,
                                         reduction='none') * foreground_mask).sum() / foreground_mask.sum()
                loss_color = (F.mse_loss(noise_pred_dict["color"], noise,
                                         reduction='none') * digit_mask).sum() / digit_mask.sum()
                loss_box = (F.mse_loss(noise_pred_dict["box"], noise,
                                       reduction='none') * bbox_mask).sum() / bbox_mask.sum()

                # Also train the merged output to ensure averaging is learned
                loss_merged = F.mse_loss(noise_pred_dict["merged"], noise)

                # The total loss is the sum of the specialized expert losses + merged loss
                loss = loss_shape + loss_color + loss_box + loss_merged

                # --- Backward Pass and Optimization ---
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 0.0002), logger,
                              writer=writer, wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)
            """
             👉 Sample Here 👈
            """

            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                real_batch = next(iter(val_loader))
                real_batch = real_batch['image'].to(device)
                shape_concept = F.one_hot(torch.tensor([0]), num_classes=3).float().squeeze(0).to(device)
                color_concept = F.one_hot(torch.tensor([1]), num_classes=3).float().squeeze(0).to(device)
                box_concept = F.one_hot(torch.tensor([2]), num_classes=3).float().squeeze(0).to(device)

                # The sampler expects a list of the concepts to be used
                concepts_to_generate = [shape_concept, color_concept, box_concept]
                generated = sample_compositional_unet(model, cfg, device, concepts_to_generate)
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run = wandb_run, writer = writer)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.train()
    except Exception as e:
        ckpt_mgr.save(model, optimizer, None, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(cfg.experiment_id, cfg.run_id,
                     total_epochs=epoch, duration=str(timedelta(seconds=int(total_time))))