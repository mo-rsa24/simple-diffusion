from pathlib import Path
from typing import Dict

import torch
from box import Box
from tqdm import tqdm

from src.config.configs import Config
from src.models.ldm.autoencoder import AutoencoderKL
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
     log_json, log_training_end, visualize_vae
from src.utils.checkpoint_manager import CheckpointManager
from datetime import timedelta
import time
from torch.optim import Adam
from torch.utils.data import DataLoader


def train(cfg: Box, dirs: Dict, model: AutoencoderKL, train_loader: DataLoader, val_loader: DataLoader, logger, device, writer = None, wandb_run = None):
    model.train()
    if cfg.optimizer.type.lower() == "adam":
        optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
    scheduler = None  # add if needed
    start_time = time.time()
    log_training_start(
        logger,
        model_name="Variational Autoencoder",
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
            total_loss, total_recon_loss, total_kl_loss = 0, 0, 0

            for step, train_batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                images = train_batch['image'].to(device)

                reconstructions, posterior = model(images)
                kl_loss = posterior.kl().mean()  # Average KL loss over the batch
                # kl_loss = kl_loss.mean()  # Average KL loss over the batch

                recon_loss = torch.nn.functional.mse_loss(reconstructions, images)
                loss = recon_loss + model.kl_weight * kl_loss
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
                # Log metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                global_step += 1

                # ── Periodic Logging ────────────────────────────────
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, total_loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_recon_loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_kl_loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            avg_loss = total_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_recon_loss, epoch_time=epoch_time)
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_kl_loss, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss,  duration=epoch_time)

            if epoch % cfg.training.log_every_epoch == 0:
                visualize_vae(cfg, epoch, model, val_loader, device, dirs, writer, wandb_run)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.train()
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