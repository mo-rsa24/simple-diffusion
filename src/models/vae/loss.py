import torch
import torch.nn.functional as F


def vae_disentanglement_loss(original_image, model_output, masks, beta=4.0):
    """
    Calculates the loss for the Disentangled Beta-VAE.

    The loss is composed of:
    1. A masked reconstruction loss to associate parts of the latent space with spatial regions.
    2. A beta-weighted KL divergence term to encourage statistical disentanglement.

    Args:
        original_image (torch.Tensor): The input image batch.
        model_output (dict): The output dictionary from the DisentangledVAE model.
        masks (dict): A dictionary containing 'digit_mask' and 'bbox_mask'.
        beta (float): The weight for the KL divergence term.

    Returns:
        dict: A dictionary containing the total loss and its individual components.
    """
    reconstruction = model_output['reconstruction']
    mu_digit = model_output['mu_digit']
    logvar_digit = model_output['logvar_digit']
    mu_bbox = model_output['mu_bbox']
    logvar_bbox = model_output['logvar_bbox']

    digit_mask = masks['digit_mask']
    bbox_mask = masks['bbox_mask']

    # --- 1. Masked Reconstruction Loss ---
    # Calculate per-pixel reconstruction error
    recon_error = F.mse_loss(reconstruction, original_image, reduction='none')

    # Apply masks to isolate the loss for each concept
    recon_loss_digit = (recon_error * digit_mask).sum() / digit_mask.sum()
    recon_loss_bbox = (recon_error * bbox_mask).sum() / bbox_mask.sum()

    # Total reconstruction loss is the sum of the masked losses
    total_recon_loss = recon_loss_digit + recon_loss_bbox

    # --- 2. KL Divergence (Regularization) ---
    # The KL divergence encourages the learned latent distributions to be close to a standard normal distribution.
    # We calculate it for both parts of the latent space.
    kld_digit = -0.5 * torch.sum(1 + logvar_digit - mu_digit.pow(2) - logvar_digit.exp())
    kld_bbox = -0.5 * torch.sum(1 + logvar_bbox - mu_bbox.pow(2) - logvar_bbox.exp())

    total_kld = kld_digit + kld_bbox

    # --- 3. Final Beta-VAE Loss ---
    # The beta term puts more pressure on the KL divergence, forcing the model
    # to learn a more disentangled latent space.
    total_loss = total_recon_loss + beta * total_kld

    return {
        'loss': total_loss,
        'reconstruction_loss': total_recon_loss.detach(),
        'kl_divergence': total_kld.detach()
    }
