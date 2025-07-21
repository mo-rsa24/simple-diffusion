from pathlib import Path
from typing import Dict

import torch
import torchvision
from box import Box
from src.models.ldm.autoencoder import AutoencoderKL
from src.models.vae.disentangled_vae import DisentangledVAE
from src.models.vae.loss import vae_disentanglement_loss
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    log_json, log_training_end, visualize_vae, visualize_epoch
from src.utils.checkpoint_manager import CheckpointManager
from datetime import timedelta
import time
from torch.optim import Adam
from torch.utils.data import DataLoader


def train_beta_vae(cfg: Box, dirs: Dict, model: DisentangledVAE, train_loader: DataLoader, val_loader: DataLoader, logger, device, writer=None, wandb_run=None):
    model.train()
    if cfg.optimizer.type.lower() == "adam":
        optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
    scheduler = None  # add if needed
    start_time = time.time()
    log_training_start(
        logger,
        model_name="Disentangled Beta-VAE",
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader)
    )

    checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt",Path(cfg.dirs.ckpt_dir)),  logger=logger)

    beta = cfg.optimizer.beta

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
            total_loss, total_recon_loss, total_kld = 0, 0, 0
            total_recon_digit, total_recon_bbox = 0, 0

            for step, train_batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                images = train_batch['image'].to(device)
                masks = {
                    'digit_mask': train_batch['digit_mask'].to(device),
                    'bbox_mask': train_batch['bbox_mask'].to(device)
                }

                # Forward pass
                model_output = model(images)

                # Calculate loss
                loss_dict = vae_disentanglement_loss(images, model_output, masks, beta=beta)
                loss = loss_dict['loss']

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_kld += loss_dict['kl_divergence'].item()

                # Log individual reconstruction losses
                total_recon_digit += loss_dict.get('recon_loss_digit', 0)
                total_recon_bbox += loss_dict.get('recon_loss_bbox', 0)

                global_step += 1

                # ── Periodic Logging ────────────────────────────────
                if global_step % cfg.training.log_every_step == 0:
                    log_batch(step, total_loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_recon_loss, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_kld, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_recon_digit, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                    log_batch(step, total_recon_bbox, cfg.optimizer.params.get("lr", 0.0003), logger, writer=writer,
                              wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    checkpoint_manager.save(model, optimizer, scheduler, epoch, global_step)
            avg_loss = total_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kld = total_kld / len(train_loader)
            avg_recon_digit = total_recon_digit / len(train_loader)
            avg_recon_bbox = total_recon_bbox / len(train_loader)
            epoch_time = time.time() - epoch_start
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_recon_loss, epoch_time=epoch_time)
            log_epoch_summary(logger, epoch, cfg.training.epochs, avg_kld, epoch_time=epoch_time)
            log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss,  duration=epoch_time)
            logger.info(
                f"Avg Recon Loss: {avg_recon_loss:.4f} (Digit: {avg_recon_digit:.4f}, BBox: {avg_recon_bbox:.4f}), Avg KL Div: {avg_kld:.4f}")

            if epoch % cfg.training.log_every_epoch == 0:
                model.eval()
                with torch.no_grad():
                    data_iter = iter(val_loader)
                    real_batch_1 = next(data_iter)['image'].to(device)
                    real_batch_2 = next(data_iter)['image'].to(device)

                    output1 = model(real_batch_1)
                    output2 = model(real_batch_2)

                    # --- Latent Swapping Visualization ---
                    # Take digit from batch 1 and box from batch 2
                    z_swapped = torch.cat([output1['z_digit'], output2['z_bbox']], dim=1)
                    reconstruction_swapped = model.decode(z_swapped)

                    # Create a grid: [Real Img 1, Its Recon, Real Img 2, Its Recon, Swapped Recon]
                    num_viz = min(4, real_batch_1.size(0))
                    viz_grid = torch.cat([
                        real_batch_1[:num_viz],
                        output1['reconstruction'][:num_viz],
                        real_batch_2[:num_viz],
                        output2['reconstruction'][:num_viz],
                        reconstruction_swapped[:num_viz]
                    ], dim=0)

                    grid_image = torchvision.utils.make_grid(viz_grid, nrow=num_viz)
                    visualize_epoch(reconstruction_swapped[:num_viz], real_batch_1[:num_viz], dirs, epoch=epoch, wandb_run=wandb_run, writer=writer)
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