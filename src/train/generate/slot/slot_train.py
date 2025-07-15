# src/train/ldm_train.py

import torch
from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import time
import torch.nn.functional as F
from src.config.configs import Config
from src.utils.checkpoint_manager import CheckpointManager
from src.train.logging.training_logger_utils import (
    log_training_start, log_epoch_start, log_batch, log_epoch_summary,
    visualize_epoch, log_json, log_training_end
)
from src.monitoring.email_alert_mailtrap import alert_on_success


def train(cfg: Config, dirs, model, ema, train_loader, logger, device, writer=None, wandb_run=None):
    # Combine model, encoder, decoder params if you want to do end-to-end finetuning (otherwise, freeze encoder/decoder)
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer.params)
    noise_scale = cfg.model.slot.get("noise_scale", 0.1)
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

    checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt", cfg.dirs.ckpt_dir), logger=logger)
    global_step = 0
    start_epoch = 1

    if cfg.training.resume_from:
        try:
            model, optimizer, _, last_epoch, global_step = checkpoint_manager.load_latest(
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

            for step, train_batch in enumerate(train_loader, 1):
                batch = train_batch['image'].to(device)
                noise = torch.randn_like(batch).to(device)
                noisy_x = batch + noise_scale * noise
                recon = model(noisy_x)
                optimizer.zero_grad()

                with autocast():
                    loss = F.mse_loss(recon, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update()
                torch.cuda.empty_cache()

                running_loss += loss.item()
                global_step += 1

                # Periodic Logging & Checkpointing
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer, wandb_tracker=wandb_run)

                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    checkpoint_manager.save(model, optimizer, None, epoch, global_step)
                break
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)

            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)
            real_batch = batch[:cfg.sampling.batch_size].to(device)  # take first 4 real images
            recon = recon[:cfg.sampling.batch_size].to(device)  # take first 4 real images
            ema.apply_shadow()

            ema.restore()
            if epoch % cfg.training.log_every_epoch == 0:
                visualize_epoch(recon, real_batch, dirs, epoch=epoch, wandb_run=wandb_run, writer=writer)
            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                checkpoint_manager.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    except Exception as e:
        checkpoint_manager.save(model, optimizer, None, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )
