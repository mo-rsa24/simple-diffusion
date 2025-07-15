from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from src.config.configs import Config
from src.models.vpsde.ScoreSdeUNet import VPSDE, dsm_loss, pc_sampler, ScoreSdeUNet
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

def train(cfg: Config,
          dirs: Dict,
          model: ScoreSdeUNet,
          ema: EMA,
          train_loader: DataLoader,
          logger,
          device,
          writer=None,
          wandb_run=None):

    # optimiser / scaler identical to DDPM script
    optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    scaler    = GradScaler()
    sde       = model.sde

    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader),
    )

    ckpt_mgr   = CheckpointManager(run_id=cfg.run_id,
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

            for step, batch_dict in enumerate(train_loader, 1):
                x0   = batch_dict["image"].to(device)
                B    = x0.size(0)
                t    = torch.rand(B, device=device) * (1. - 1e-5)  # U(0,1)

                optimizer.zero_grad(set_to_none=True)

                loss = dsm_loss(model, x0, t, sde)      # <── the only change
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                ema.update()
                running_loss += loss.item()
                global_step += 1

                # ── periodic logging and checkpoint ───────────────
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss,
                              cfg.optimizer.params.get("lr", 1e-4),
                              logger, writer=writer, wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    ckpt_mgr.save(model, optimizer, None, epoch, global_step)
                break
            # ── epoch-level summaries ─────────────────────────────
            avg_loss  = running_loss / len(train_loader)
            epoch_dur = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs,
                              avg_loss, epoch_time=epoch_dur)
            log_json(logger, "Epoch Summary",
                     epoch=epoch, train_loss=avg_loss, duration=epoch_dur)

            # ── sampling preview (uses EMA weights) ──────────────
            real_batch = x0[:cfg.sampling.batch_size].to(device)
            ema.apply_shadow()
            sample = pc_sampler(model, sde,
                                shape=(cfg.sampling.batch_size,
                                       cfg.dataset.channels,
                                       cfg.dataset.image_size,
                                       cfg.dataset.image_size))
            ema.restore()

            if epoch % cfg.training.log_every_epoch == 0:
                visualize_epoch(sample, real_batch, dirs,
                                epoch=epoch, wandb_run=wandb_run, writer=writer)
            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    except Exception as e:
        ckpt_mgr.save(model, optimizer, None, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    # ── training finished ────────────────────────────────────
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(cfg.experiment_id, cfg.run_id,
                     total_epochs=epoch,
                     duration=str(timedelta(seconds=int(total_time))))