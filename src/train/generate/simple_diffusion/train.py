from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from src.config.configs import Config
from src.models.vanilla.ema import EMA
from src.models.vanilla.unet import Unet
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    visualize_epoch, log_json, log_training_end
from src.models.vanilla.diffusion import generate_batch
from src.utils.calculations import q_sample
from src.utils.checkpoint_manager import CheckpointManager
from datetime import timedelta
import time
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


def train(cfg: Config, dirs: Dict, model: Unet, ema: EMA, train_loader: DataLoader, logger, device, writer = None, wandb_run = None):
    if cfg.optimizer.type.lower() == "adam":
        optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
    scheduler = None  # add if needed
    scaler = GradScaler()
    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader)
    )

    checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt",Path(cfg.dirs.ckpt_dir)),  logger=logger)
    global_step = 0
    start_epoch = 1

    if cfg.training.resume_from:
        try:
            model, optimizer, scheduler, last_epoch, global_step = checkpoint_manager.load_latest(
                model, optimizer, scheduler, map_location=device
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
            for step, train_batch in enumerate(train_loader, 1):
                batch = train_batch['image']
                batch_size = int(batch.shape[0])
                batch = batch.to(device)
                noise = torch.randn_like(batch)
                t = torch.randint(0, cfg.diffusion.timesteps, (batch_size, ), device=device).long()

                optimizer.zero_grad()

                x_noisy = q_sample(
                    x_start=batch,
                    t=t,
                    noise=noise,
                    timesteps=cfg.diffusion.timesteps,
                    beta_start=cfg.diffusion.beta_start,
                    beta_end=cfg.diffusion.beta_end,
                )
                with autocast():
                  pred_noise = model(x_noisy, t)

                # 3) cast back to FP32 for loss
                pred_noise = pred_noise.float()
                if cfg.diffusion.loss_type == 'l1':
                    loss = F.l1_loss(noise, pred_noise)
                elif cfg.diffusion.loss_type == 'l2':
                    loss = F.mse_loss(noise, pred_noise)
                elif cfg.diffusion.loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, pred_noise)

                # backward with the scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # optional: free any cached fragments

                ema.update()
                torch.cuda.empty_cache()

                running_loss += loss.item()
                global_step += 1

                # ── Periodic Logging ────────────────────────────────
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss,  duration=epoch_time)
            real_batch = batch[:cfg.sampling.batch_size].to(device)  # take first 16 real images
            ema.apply_shadow()
            generated = generate_batch(
                model,
                image_size=cfg.dataset.image_size,
                batch_size=cfg.sampling.batch_size,
                channels=cfg.dataset.channels,
                timesteps=cfg.diffusion.timesteps,
                beta_start=cfg.diffusion.beta_start,
                beta_end=cfg.diffusion.beta_end,
            )
            ema.restore()
            if epoch % cfg.training.log_every_epoch == 0:
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run = wandb_run, writer = writer)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:
        checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )