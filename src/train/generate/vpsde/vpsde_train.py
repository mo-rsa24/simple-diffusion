from pathlib import Path
from typing import Dict
import torch.nn.functional as F
import torch
from box import Box
from src.models.vpsde.ColoredMNISTScoreModel import ColoredMNISTScoreModel, VPSDE
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    visualize_epoch, log_json, log_training_end
from src.utils.checkpoint_manager import CheckpointManager
from datetime import timedelta
import time
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.utils.sampling import ScoreModelSampler


def train(cfg: Box,
          dirs: Dict,
          model: ColoredMNISTScoreModel,
          sde: VPSDE,
          train_loader: DataLoader,
          val_loader: DataLoader,
          logger,
          device,
          writer=None,
          wandb_run=None):

    optimizer = Adam(model.parameters(), **cfg.optimizer.params)
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
    sampler = ScoreModelSampler(sde=sde)
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
                optimizer.zero_grad()
                x0   = batch_dict["image"].to(device)
                t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
                noise = torch.randn_like(x0)
                sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
                predicted_noise = model(xt, t.float())
                if cfg.diffusion.loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif cfg.diffusion.loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif cfg.diffusion.loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                global_step += 1

                # ── periodic logging and checkpoint ───────────────
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss,
                              cfg.optimizer.params.get("lr", 1e-4),
                              logger, writer=writer, wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            # ── epoch-level summaries ─────────────────────────────
            avg_loss  = running_loss / len(train_loader)
            epoch_dur = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs,
                              avg_loss, epoch_time=epoch_dur)
            log_json(logger, "Epoch Summary",
                     epoch=epoch, train_loss=avg_loss, duration=epoch_dur)

            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                real_batch = next(iter(val_loader))
                real_batch = real_batch['image'][:cfg.sampling.batch_size].to(device)

                generated = sampler.sample(model, real_batch.shape, cfg.diffusion.timesteps, device=device)
                prefix = f"{cfg.experiment_id}_run_{cfg.run_id}_sample_epoch_{epoch}"
                visualize_epoch(generated, real_batch, dirs, epoch=epoch, prefix=prefix, wandb_run=wandb_run,
                                writer=writer)
            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
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