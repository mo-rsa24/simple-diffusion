# src/train/ldm_train.py
from pathlib import Path
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import time

from src.models.ldm.autoencoder import AutoencoderKL, DiagonalGaussianDistribution
from src.models.ldm.diffusion import latent_sample
from src.models.vanilla.unet import Unet
from src.utils.calculations import q_sample
from src.utils.checkpoint_manager import CheckpointManager
from src.train.logging.training_logger_utils import (
    log_training_start, log_epoch_start, log_batch, log_epoch_summary,
    visualize_epoch, log_json, log_training_end
)
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.utils.sampling import ldm_sampler


def train(cfg, dirs, model: Unet, ema, vae: AutoencoderKL, train_loader, logger, device, writer=None, wandb_run=None):
    # Combine model, encoder, decoder params if you want to do end-to-end finetuning (otherwise, freeze encoder/decoder)
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer.params)
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
    vae_checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=Path(dirs.get('ckpt', cfg.dirs.ckpt_dir)) / 'vae' , logger=logger)
    vae_optimizer = torch.optim.Adam(vae.parameters(), **cfg.optimizer.params)
    try:
        # Load the state dictionary from the latest checkpoint file
        vae, vae_optimizer, scheduler, last_epoch, global_step = vae_checkpoint_manager.load_latest(
            vae, vae_optimizer, map_location=device)
        print("Successfully loaded VAE weights.")
    except FileNotFoundError:
        print(f"ERROR: No VAE checkpoint found in '{(Path(dirs.get('ckpt', cfg.dirs.ckpt_dir)) / 'vae')}'.")
        print("Please run the 'train_vae' task first to generate a checkpoint.")
        return
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
        model.train()
        vae.eval()
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_start = time.time()
            log_epoch_start(epoch - 1, logger)
            running_loss = 0.0

            for step, train_batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                batch = train_batch['image'].to(device)
                with torch.no_grad():
                    posterior: DiagonalGaussianDistribution= vae.encode(batch)
                    latents = posterior.sample() * 0.18215
                t = torch.randint(0, cfg.diffusion.timesteps, (latents.shape[0],), device=device).long()
                noise = torch.randn_like(latents)
                latents_noisy = q_sample(latents, t, noise, timesteps=cfg.diffusion.timesteps,
                                         beta_start=cfg.diffusion.beta_start, beta_end=cfg.diffusion.beta_end)
                with autocast():
                    predicted_noise = model(latents_noisy, t)
                predicted_noise = predicted_noise.float()
                if cfg.diffusion.loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif cfg.diffusion.loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif cfg.diffusion.loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)

                # backward with the scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                ema.update()
                torch.cuda.empty_cache()

                running_loss += loss.item()
                global_step += 1

                # Periodic Logging & Checkpointing
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)

                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    checkpoint_manager.save(model, optimizer, None, epoch, global_step)
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)

            # --- Visualization: Generate Samples in Latent Space, Decode, Log ---
            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                real_batch = next(iter(train_loader))
                real_batch = real_batch['image'].to(device)

                # Generate samples in latent space using UNet, then decode to image
                with torch.no_grad():
                    generated = ldm_sampler(model, cfg, device)
                visualize_epoch(generated[:cfg.sampling.batch_size], real_batch[:cfg.sampling.batch_size], dirs, epoch, wandb_run, writer)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                checkpoint_manager.save(model, optimizer, None, epoch, global_step)

            if device.type == "cuda":
                torch.cuda.empty_cache()
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
