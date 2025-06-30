from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from src.config.configs import Config
from src.models.edm.EDM import EDMNoiseSchedule, edm_loss, edm_sampler
from src.models.vpsde.ScoreSdeUNet import VPSDE, dsm_loss, pc_sampler
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
          model,
          ema: EMA,
          train_loader: DataLoader,
          logger,
          device,
          writer=None,
          wandb_run=None):

    opt      = Adam(model.parameters(), **cfg.optimizer.params)
    scaler   = GradScaler()
    schedule = EDMNoiseSchedule(**cfg.diffusion.noise_schedule.params)
    sigma_data = cfg.diffusion.sigma_data

    start   = time.time()
    log_training_start(
        logger,
        model_name="EDM",
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
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_time = time.time()
        log_epoch_start(epoch - 1, logger)
        running = 0.0

        for step, batch in enumerate(train_loader, 1):
            x0 = batch["image"].to(device)
            B  = x0.size(0)
            sigma = schedule.sample(B, device)

            opt.zero_grad(set_to_none=True)
            loss = edm_loss(model, x0, sigma, sigma_data)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update()

            running += loss.item()
            global_step += 1

            if global_step % cfg.training.log_every_step == 0:
                log_batch(step, loss,
                          cfg.optimizer.params.get("lr", 1e-4),
                          logger, writer=writer, wandb_tracker=wandb_run)
                ckpt_mgr.save(model, opt, None, epoch, global_step)

        # epoch summary
        avg = running / len(train_loader)
        log_epoch_summary(logger, epoch, cfg.training.epochs,
                          avg, epoch_time=time.time() - epoch_time)
        log_json(logger, "Epoch Summary",
                 epoch=epoch, train_loss=avg,
                 duration=time.time() - epoch_time)

        # preview samples
        ema.apply_shadow()
        shape = (
            cfg.sampling.batch_size,
            cfg.dataset.channels,
            cfg.dataset.image_size,
            cfg.dataset.image_size,
        )
        sampler_cfg = getattr(cfg.sampling, "edm_sampler", {}).get("params", {})
        sample = edm_sampler(model, shape, **sampler_cfg)
        ema.restore()

        if epoch % cfg.training.log_every_epoch == 0:
            visualize_epoch(sample, x0[:sample.size(0)], dirs,
                            epoch=epoch, wandb_run=wandb_run, writer=writer)

    # training finished
    tot = time.time() - start
    log_training_end(logger, tot)
    alert_on_success(cfg.experiment_id, cfg.run_id,
                     total_epochs=epoch,
                     duration=str(timedelta(seconds=int(tot))))