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
from src.utils.sampling import ddpm_sampler


def train(cfg, dirs: Dict, model, ema: EMA, train_loader: DataLoader, logger, device, writer=None, wandb_run=None):
    optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    scaler = GradScaler() if getattr(cfg.training, "scaler", "") == "amp" else None
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
                conds = [batch.get("digit_label", None), batch.get("color_label", None), batch.get("bbox_label", None)]
                conds = [c.to(device) if torch.is_tensor(c) else None for c in conds]

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler is not None):
                    pred = model(x_noisy, t, conds)
                    loss = F.mse_loss(pred, noise)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                ema.update()
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
            model.eval()
            real_batch_data = next(iter(train_loader))
            real_batch = real_batch_data['image'].to(device)
            conds = [
                real_batch_data.get("digit_label", None),
                real_batch_data.get("color_label", None),
                real_batch_data.get("bbox_label", None)
            ]
            conds = [c.to(device) if torch.is_tensor(c) else None for c in conds]
            for i in range(len(conds)):
                if conds[i] is not None:
                    conds[i] = conds[i][:cfg.sampling.batch_size]
            generated = ddpm_sampler(model, cfg, device, conds)
            if epoch % cfg.training.log_every_epoch == 0:
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, wandb_run=wandb_run, writer=writer)

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