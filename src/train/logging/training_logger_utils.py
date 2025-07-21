# src/utils/training_logger_utils.py
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision
import wandb
from box import Box
from matplotlib import pyplot as plt
from torchvision.utils import save_image, make_grid

from .log import log_grid_images, log_table_wandb
from ...utils.sampling import ito_sampler
from ...utils.visualization import save_side_by_side_images


def log_training_start(logger, model_name, experiment_id, run_id, task, total_epochs, total_batches):
    logger.info(
        f"🚀 Training started\n"
        f"🔧 Model: {model_name}\n"
        f"🧪 Experiment: {experiment_id}\n"
        f"🏃‍♂️ Run ID: {run_id}\n"
        f"🤞🏽 Task: {task}\n"
        f"🖋️ Experiment Name: experiment_{experiment_id}_{task}\n"
        f"⏳ Total Epochs: {total_epochs}, Batches/Epoch: {total_batches}"
    )

def log_epoch_start(epoch, logger):
    logger.info(f"\n--- Epoch {epoch+1} Start ---")

def log_epoch_summary(logger, epoch, total_epochs, train_loss, val_loss=None, epoch_time=None):
    msg = f"Epoch {epoch}/{total_epochs} completed | Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    if epoch_time is not None:
        msg += f" | Time: {timedelta(seconds=int(epoch_time))}"
    logger.info(msg)



def log_training_end(logger, training_time):
    logger.info(
        f"🏁 Training finished!\n"
        f"🕒 Total Duration: {training_time:.2f}s"
    )


def log_batch_progress(logger, epoch, total_epochs, step, total_steps, loss, lr, time_elapsed, gpu_mem=None):
    hrs, rem = divmod(int(time_elapsed), 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    msg = (
        f"[Epoch {epoch}/{total_epochs}] "
        f"Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed_str}"
    )
    if gpu_mem is not None:
        msg += f" | GPU_mem: {gpu_mem:.0f} MB"
    logger.info(msg)

def log_training_stats(logger, epoch, avg_loss, duration, optimizer, writer=None, wandb_tracker=None):
    lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"【Epoch {epoch}】 Avg Loss: {avg_loss:.4f} | Duration: {timedelta(seconds=int(duration))} | LR: {lr:.2e}"
    )
    if writer:
        writer.add_scalar("train/avg_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/epoch_duration", duration, epoch)
    if wandb_tracker:
        wandb_tracker.log({
            "train/avg_loss": avg_loss,
            "train/lr": lr,
            "train/epoch_duration": duration
        }, step=epoch)


def log_batch(step, loss, lr, logger, writer=None, wandb_tracker=None):
    logger.info(f"[Step {step:6d}] loss={loss:.4f}  lr={lr:.6f}")
    mem_mb = 0.0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.debug(f"GPU memory allocated: {mem_mb:.1f} MB")
    if writer:
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/lr", lr, step)
        writer.add_scalar("train/memory_MB", mem_mb, step)
    if wandb_tracker:
        wandb_tracker.log({"train/loss":  loss, "train/lr":lr, "train/memory_MB":  mem_mb}, step=step)

def log_json(logger, message, **extra_data):
    logger.info(message, extra={"extra_data": extra_data})

def log_exception(logger, epoch=None, step=None, exception=None):
    msg = "❌ Exception during training"
    if epoch is not None:
        msg += f" at Epoch {epoch}"
    if step is not None:
        msg += f", Step {step}"
    logger.exception(f"{msg}: {str(exception)}")

def visualize_epoch(generated: torch.Tensor, real: torch.Tensor,  dirs: Dict, epoch: int=None, prefix: str = "generated", wandb_run = None, writer = None):
    samples_dir: Path = dirs.get("results_samples") / Path(f"epoch_{epoch}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    # --- 1) Save each generated image individually ---
    for i, img in enumerate(generated):
        img = img.detach().cpu().clamp(-1, 1)
        img = (img + 1) / 2  # map [-1,1] -> [0,1]
        save_image(img, samples_dir / Path(f"{prefix}_{i:03d}.png"), normalize=False)

    # --- 2) Side-by-side comparisons (one per index) ---
    side_by_side_dir: Path = dirs.get("results_side_by_side") / Path(f"epoch_{epoch}")
    side_by_side_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side_images(real, generated, side_by_side_dir)

    # --- 3) TensorBoard logging ---
    if writer:
        log_grid_images(generated, real, epoch, writer)

    # --- 4) W&B logging ---
    if wandb_run:
        log_table_wandb(generated, real, epoch, wandb_run)


def generate_and_save_grid(individual_models, composed_samples, compose_model_name, sampling_params, save_path):
    """
    Generates and saves a grid of images for comparing individual and composed models.
    Args:
        individual_models (list): List of the individual pre-trained models.
        composed_samples (torch.Tensor): The samples generated by the main composition method.
        compose_model_name (str): The name of the composition method used (e.g., 'ito', 'composable_unet').
        sampling_params (dict): Dictionary of parameters for the sampler.
        save_path (str): The directory to save the image grid.
    """
    num_samples = sampling_params["shape"][0]

    # Generate samples from each individual model for comparison
    samples_from_individuals = []
    for model in individual_models:
        model.eval()
        samples = ito_sampler(models=[model], **sampling_params)
        samples_from_individuals.append(samples)

    # Naive overlay of the first two models
    naive_overlay = torch.clamp(samples_from_individuals[0] + samples_from_individuals[1], -1, 1)

    # Prepare tensors for the grid
    # Format: [Model A samples, Model B samples, Naive Overlay, Composed Samples]
    grid_tensors = samples_from_individuals + [naive_overlay, composed_samples]

    # Create a single tensor with all images
    full_grid_tensor = torch.cat(grid_tensors, dim=0)

    # Create and save the grid
    grid = torchvision.utils.make_grid(
        full_grid_tensor,
        nrow=num_samples,  # Each row will show one sample from each method
        normalize=True,
        scale_each=True
    )

    filename = f"superposition_grid_{compose_model_name}.png"
    filepath = os.path.join(save_path, filename)
    torchvision.utils.save_image(grid, filepath)
    print(f"Saved composition grid to {filepath}")

@torch.no_grad()
def visualize_vae(cfg: Box, epoch, vae_model, val_loader, device, dirs, writer, wandb_run=None):
    """
    Generates and logs a comprehensive suite of VAE visualizations.
    """
    vae_model.eval()
    print(f"\n--- Epoch {epoch}: Generating VAE visualizations ---")

    # Get a fixed batch from validation set for consistent visualization
    try:
        val_batch = next(iter(val_loader))
        images = val_batch['image'][: cfg.sampling.batch_size].to(device)
    except StopIteration:
        print("Validation loader is empty, skipping visualization.")
        return

    # --- 1. Reconstruction Quality Grid (Input -> Recon -> Residual) ---
    reconstructions, posterior = vae_model(images)
    residuals = (images - reconstructions).abs()
    # Clamp residuals for better visibility
    residuals = torch.clamp(residuals, 0, 1)
    comparison_grid = torch.cat([images, reconstructions, residuals])
    grid = make_grid(comparison_grid, nrow=images.size(0), normalize=True)
    samples_dir: Path = dirs.get("results_samples") / Path(f"epoch_{epoch}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_path = samples_dir / Path(f"vae_recons_epoch_{epoch}.png")
    save_image(grid,save_path , normalize=False)

    if writer:
        writer.add_image('VAE/Reconstruction Quality', grid, epoch)
    if wandb_run:
        wandb_run.log({"VAE/Reconstruction Quality": wandb.Image(save_path)}, step=epoch)

    # --- 2. Latent Prior vs. Posterior Histogram ---
    mus, logvars = posterior.mean, posterior.logvar
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mus.detach().cpu().numpy().flatten(), bins=50, alpha=0.7, label='Posterior µ', density=True)
    ax.hist(torch.exp(0.5 * logvars).detach().cpu().numpy().flatten(), bins=50, alpha=0.7, label='Posterior σ',
            density=True)
    # Overlay standard normal for comparison
    x = np.linspace(-3, 3, 100)
    ax.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2), 'r--', label='Prior N(0,1)')
    ax.legend()
    ax.set_title(f'Latent Distribution - Epoch {epoch}')
    latent_prior_dir: Path = dirs.get("results_latent_prior") / Path(f"epoch_{epoch}")
    latent_prior_dir.mkdir(parents=True, exist_ok=True)
    save_path = latent_prior_dir /Path(f"vae_latent_prior_epoch_{epoch}.png")
    plt.savefig(save_path)
    if writer:
        writer.add_figure('VAE/Latent Distribution', fig, epoch)
    if wandb_run:
        wandb_run.log({"VAE/Latent Distribution": wandb.Image(save_path)}, step=epoch)
    plt.close(fig)

    # --- 3. Random Samples from Prior ---
    z_channels = vae_model.decoder.z_channels
    latent_h = images.shape[2] // (2 ** len(vae_model.encoder.down))
    latent_w = images.shape[3] // (2 ** len(vae_model.encoder.down))
    z = torch.randn(images.size(0), z_channels, latent_h, latent_w, device=device)
    prior_samples = vae_model.decode(z)
    grid = make_grid(prior_samples, nrow=images.size(0), normalize=True)
    prior_samples_dir: Path = dirs.get("results_prior_samples") / Path(f"epoch_{epoch}")
    prior_samples_dir.mkdir(parents=True, exist_ok=True)
    save_path = prior_samples_dir / Path(f"vae_prior_samples_epoch_{epoch}.png")
    save_image(grid, save_path)
    if writer:
        writer.add_image('VAE/Prior Samples', grid, epoch)
    if wandb_run:
        wandb_run.log({"VAE/Prior Samples": wandb.Image(save_path)}, step=epoch)

    # --- 4. Latent Space Perturbations ---
    base_z = mus[0:1]  # Use the latent vector of the first image
    all_perts = [images[0:1]]  # Start with the original image
    for sigma in [0.1, 0.5, 1.0, 2.0]:
        z_perturbed = base_z + sigma * torch.randn_like(base_z)
        decoded_pert = vae_model.decode(z_perturbed)
        all_perts.append(decoded_pert)
    pert_grid = make_grid(torch.cat(all_perts), nrow=len(all_perts), normalize=True)
    latent_space_dir: Path = dirs.get("results_latent_space") / Path(f"epoch_{epoch}")
    latent_space_dir.mkdir(parents=True, exist_ok=True)
    save_path = latent_space_dir / Path(f"vae_perturbations_epoch_{epoch}.png")
    save_image(pert_grid, save_path)
    if writer:
        writer.add_image('VAE/Latent Perturbations (σ=0, 0.1, 0.5, 1.0, 2.0)', pert_grid, epoch)
    if wandb_run:
        wandb_run.log({"VAE/Latent Perturbations": wandb.Image(save_path)}, step=epoch)

    # --- 5. Reconstruction Error Distribution ---
    recon_errors = (images - reconstructions).pow(2).mean(dim=[1, 2, 3]).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(recon_errors, bins=30)
    ax.set_title(f'Reconstruction Error (MSE) Distribution - Epoch {epoch}')
    ax.set_xlabel('Per-Image MSE')
    ax.set_ylabel('Frequency')
    reconstruction_error_distribution_dir: Path = dirs.get("results_reconstruction_error_distribution") / Path(f"epoch_{epoch}")
    reconstruction_error_distribution_dir.mkdir(parents=True, exist_ok=True)
    save_path = reconstruction_error_distribution_dir / Path(f"vae_recon_error_dist_epoch_{epoch}.png")
    plt.savefig(save_path)
    if writer:
        writer.add_figure('VAE/Reconstruction Error Distribution', fig, epoch)
    if wandb_run:
        wandb_run.log({"VAE/Reconstruction Error Distribution": wandb.Image(save_path)}, step=epoch)
    plt.close(fig)

    print("All VAE visualizations generated and saved.")