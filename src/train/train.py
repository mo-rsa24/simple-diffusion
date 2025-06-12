from pathlib import Path
from typing import Dict

import torch

from src.config.configs import Config
from src.models.ema import EMA
from src.models.unet import Unet
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    visualize_epoch, log_json, log_training_end
from src.models.diffusion import p_losses, sample, generate_batch
from src.utils.checkpoint_manager import CheckpointManager
from datetime import timedelta
import time
from torch.optim import Adam
from torch.utils.data import DataLoader

def train(cfg: Config, dirs: Dict, model: Unet, ema: EMA, train_loader: DataLoader, logger, device, writer = None, wandb_run = None):
    if cfg.optimizer.type.lower() == "adam":
        optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
    scheduler = None  # add if needed

    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader)
    )

    checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt",Path(cfg.dirs.ckpt_dir)),  logger=logger)
    global_step = 0
    start_epoch = 1

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()
        log_epoch_start(epoch - 1, logger)

        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
          optimizer.zero_grad()

          batch_size = batch['image'].shape[0]
          batch = batch['image'].to(device)

          # Algorithm 1 line 3: sample t uniformally for every example in the batch
          t = torch.randint(0, cfg.diffusion.timesteps, (batch_size,), device=device).long()

          loss = p_losses(model, batch, t, loss_type=cfg.diffusion.loss_type, timesteps=cfg.diffusion.timesteps)
          loss.backward()
          optimizer.step()
          ema.update()

          running_loss += loss.item()
          global_step += 1

          # ── Periodic Logging ────────────────────────────────
          if global_step % cfg.training.log_every_step == 0:
              log_batch(step, loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer, wandb_tracker=wandb_run)
              checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
        log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss,  duration=epoch_time)
        real_batch = batch[:16].to(device)  # take first 16 real images
        ema.apply_shadow()
        generated = generate_batch(model, image_size=cfg.dataset.image_size, batch_size=4, channels=cfg.dataset.channels,timesteps=cfg.diffusion.timesteps)
        ema.restore()
        if epoch % cfg.training.log_every_epoch == 0:
            visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run = wandb_run, writer = writer)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )