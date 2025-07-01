from pathlib import Path
from datetime import timedelta
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.composable_diffusion import ComposableDiffusionModel
from src.models.vanilla.diffusion import generate_batch
from src.models.vanilla.ema import EMA
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_batch, log_training_stats, log_training_end
from src.utils.calculations import q_sample
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.visualization import visualize_images, save_side_by_side_images


def visualize_epoch(real: torch.Tensor, generated: torch.Tensor, dirs: dict, epoch: int):
    samples_dir = Path(dirs.get("results_samples")) / f"epoch_{epoch}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    grid_path = samples_dir / "generated.png"
    visualize_images(generated, save_path=str(grid_path), show=False, title="Compositional Diffusion")

    side_by_side_dir = Path(dirs.get("results_side_by_side")) / f"epoch_{epoch}"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side_images(real, generated, side_by_side_dir)


def train(
    cfg,
    dirs: dict,
    model: ComposableDiffusionModel,
    ema: EMA,
    train_loader: DataLoader,
    logger,
    device,
    writer=None,
    wandb_run=None,
):
    optimizer = Adam(model.parameters(), **cfg.optimizer.params)
    scaler = GradScaler() if getattr(cfg.training, "scaler", "none") == "amp" else None

    beta_schedule = getattr(cfg.diffusion, "beta_schedule", "linear")
    _ = beta_schedule  # placeholder to show usage

    start_time = time.time()
    log_training_start(
        logger,
        model_name="ComposableDiffusionModel",
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader),
    )

    ckpt_mgr = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt", Path(cfg.dirs.ckpt_dir)), logger=logger)
    global_step = 0
    start_epoch = 1
    if cfg.training.resume_from:
        try:
            model, optimizer, _, last_epoch, global_step = ckpt_mgr.load_latest(model, optimizer, None, map_location=device)
            start_epoch = last_epoch + 1
            logger.info(f"Resuming from epoch {last_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                x = batch["image"].to(device)
                noise = torch.randn_like(x)
                t = torch.randint(0, cfg.diffusion.timesteps, (x.size(0),), device=device).long()
                x_noisy = q_sample(x, t, noise, cfg.diffusion.timesteps, cfg.diffusion.beta_start, cfg.diffusion.beta_end)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler is not None):
                    out = model(x_noisy, t)
                    loss_shape = F.mse_loss(out["shape"], noise)
                    loss_color = F.mse_loss(out["color"], noise)
                    loss_box = F.mse_loss(out["box"], noise)
                    loss = loss_shape + loss_color + loss_box
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
                    log_batch(step, loss, cfg.optimizer.params.get("lr", 2e-4), logger, writer=writer, wandb_tracker=wandb_run)
                if cfg.training.save_every_step and global_step % cfg.training.save_every_step == 0:
                    ckpt_mgr.save(model, optimizer, None, epoch, global_step)

            avg_loss = running_loss / len(train_loader)
            duration = time.time() - epoch_start
            log_training_stats(logger, epoch, avg_loss, duration, optimizer, writer=writer, wandb_tracker=wandb_run)

            real_batch = x[: cfg.sampling.batch_size].to(device)
            ema.apply_shadow()
            generated = generate_batch(lambda x_, t_: model(x_, t_)["merged"],
                                       image_size=cfg.dataset.image_size,
                                       batch_size=cfg.sampling.batch_size,
                                       channels=cfg.dataset.channels,
                                       timesteps=cfg.diffusion.timesteps,
                                       beta_start=cfg.diffusion.beta_start,
                                       beta_end=cfg.diffusion.beta_end,
                                       device=device)
            ema.restore()

            if epoch % cfg.training.log_every_epoch == 0:
                visualize_epoch(real_batch, generated, dirs, epoch)

            if cfg.training.save_every_epoch and epoch % cfg.training.save_every_epoch == 0:
                ckpt_mgr.save(model, optimizer, None, epoch, global_step)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:
        ckpt_mgr.save(model, optimizer, None, epoch, global_step)
        logger.error(f"Training interrupted: {e}")
        raise
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(cfg.experiment_id, cfg.run_id, total_epochs=epoch,
                     duration=str(timedelta(seconds=int(total_time))))