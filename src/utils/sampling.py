import torch
from tqdm import tqdm
import torch.nn.functional as F

from src.models.composable_diffusion import ComposableDiffusionModel
from src.models.ldm.autoencoder import AutoencoderKL
from src.models.vanilla.composable_unet import ComposableUnet


def get_diffusion_constants(cfg, device):
    """Helper function to get all diffusion schedule constants."""
    beta_start = cfg.diffusion.beta_start
    beta_end = cfg.diffusion.beta_end
    timesteps = cfg.diffusion.timesteps

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def get_schedule(timesteps, beta_min, beta_max, device):
    """
    Generates a linear noise schedule.
    """
    betas = torch.linspace(beta_min, beta_max, timesteps, device=device)
    alphas = 1. - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_hat


@torch.no_grad()
def ito_sampler(models, shape, timesteps, beta_min, beta_max, weights=None, device="cpu"):
    """
    Samples from a combined score function using the Itô Density Estimator method.
    This is a training-free composition method.
    Args:
        models (list): A list of pre-trained UNet models.
        shape (tuple): The shape of the output tensor (B, C, H, W).
        timesteps (int): The number of denoising steps.
        beta_min (float): The minimum beta value for the noise schedule.
        beta_max (float): The maximum beta value for the noise schedule.
        weights (list, optional): A list of weights for combining the models. Defaults to equal weighting.
        device (str, optional): The device to run on. Defaults to "cpu".
    Returns:
        torch.Tensor: The generated samples.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    betas, alphas, alphas_hat = get_schedule(timesteps, beta_min, beta_max, device)

    x = torch.randn(shape, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # The combined score is the weighted sum of individual model scores (predicted noise)
        combined_noise_pred = torch.zeros_like(x)
        for model, weight in zip(models, weights):
            model.eval()
            combined_noise_pred += weight * model(x, t_tensor)

        alpha_t = alphas[t]
        alpha_hat_t = alphas_hat[t]

        # Denoising step using the combined noise prediction
        # (xt - (1-alpha_t)/sqrt(1-alpha_hat_t) * pred_noise) / sqrt(alpha_t)
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * combined_noise_pred)

        if t > 0:
            # Add noise for the next step
            noise = torch.randn_like(x)
            sigma_t_sq = (1. - alphas_hat[t - 1]) / (1. - alphas_hat[t]) * betas[t]
            x += torch.sqrt(sigma_t_sq) * noise

    return x.clamp(-1, 1)


@torch.no_grad()
def composable_expert_sampler(model: ComposableDiffusionModel, cfg, device):
    """
    Sampler specifically for the ComposableDiffusionModel which returns a
    dictionary of expert predictions.

    This sampler uses the "merged" output from the model, which is assumed
    to be the intelligent combination of all expert noise predictions.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    posterior_variance = constants["posterior_variance"]

    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels
    timesteps = cfg.diffusion.timesteps

    # Start with random noise
    img = torch.randn((batch_size, channels, img_size, img_size), device=device)

    # Set the model to evaluation mode
    model.eval()

    for i in tqdm(reversed(range(timesteps)), desc="Composable Expert Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # Get the dictionary of predictions and extract the merged noise
        model_output = model(img, t)
        predicted_noise = model_output["merged"]

        # Standard DDPM denoising step (from ddpm_sampler)
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]

        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance[i]) * noise

    return img.clamp(-1, 1)

@torch.no_grad()
def composable_unet_sampler(model, shape, timesteps, beta_min, beta_max, device="cpu"):
    """
    Samples from a ComposableUNet.
    Args:
        model (nn.Module): The ComposableUNet model.
        shape (tuple): The shape of the output tensor (B, C, H, W).
        timesteps (int): The number of denoising steps.
        beta_min (float): The minimum beta value for the noise schedule.
        beta_max (float): The maximum beta value for the noise schedule.
        device (str, optional): The device to run on. Defaults to "cpu".
    Returns:
        torch.Tensor: The generated samples.
    """
    betas, alphas, alphas_hat = get_schedule(timesteps, beta_min, beta_max, device)

    x = torch.randn(shape, device=device)
    model.eval()

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Get the fused noise prediction from the composable model
        pred_noise = model(x, t_tensor)

        alpha_t = alphas[t]
        alpha_hat_t = alphas_hat[t]

        # Denoising step
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)

        if t > 0:
            # Add noise for the next step
            noise = torch.randn_like(x)
            sigma_t_sq = (1. - alphas_hat[t - 1]) / (1. - alphas_hat[t]) * betas[t]
            x += torch.sqrt(sigma_t_sq) * noise

    return x.clamp(-1, 1)

@torch.no_grad()
def ddpm_sampler(model, cfg, device, conds=None):
    """
    Standard DDPM sampler for noise-prediction models like UNet and CompositionalUNet.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]

    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels
    timesteps = cfg.diffusion.timesteps

    img = torch.randn((batch_size, channels, img_size, img_size), device=device)

    for i in tqdm(reversed(range(timesteps)), desc="DDPM Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        predicted_noise = model(img, t, conds)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]

        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = constants["posterior_variance"][i]
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img


@torch.no_grad()
def composable_ddpm_sampler(
        model: ComposableUnet,
        cfg,
        device,
        digit_labels,
        digit_color_labels,
        bbox_color_labels,
        guidance_scale=7.5
):
    """
    Conditional DDPM sampler for ComposableUnet using Classifier-Free Guidance.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    posterior_variance = constants["posterior_variance"]
    timesteps = cfg.diffusion.timesteps
    batch_size = digit_labels.shape[0]

    # Start with random noise
    img = torch.randn((batch_size, cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size), device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='Composable DDPM Sampling', total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # 1. Predict noise with conditioning (conditional pass)
        predicted_noise_cond = model(img, t, digit_labels, digit_color_labels, bbox_color_labels)

        # 2. Predict noise without conditioning (unconditional pass)
        # Create null labels (assuming class index 10 is the null token)
        uncond_digit_labels = torch.full_like(digit_labels, 10)
        uncond_digit_color_labels = torch.full_like(digit_color_labels, 10)
        uncond_bbox_color_labels = torch.full_like(bbox_color_labels, 10)

        predicted_noise_uncond = model(img, t, uncond_digit_labels, uncond_digit_color_labels, uncond_bbox_color_labels)

        # 3. Combine predictions using Classifier-Free Guidance
        noise_pred = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)

        # 4. Denoise for one step (p_sample logic)
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * noise_pred)

        if i == 0:
            img = model_mean
        else:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance[i]) * noise

    return img

@torch.no_grad()
def vpsde_sampler(model, cfg, device, conds=None, corrector_steps=1, snr=0.15):
    """
    Sampler for VP-SDE score-based models (e.g., ScoreSdeUNet).
    Uses a predictor-corrector loop.
    """
    constants = get_diffusion_constants(cfg, device)
    betas = constants["betas"]
    timesteps = cfg.diffusion.timesteps
    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels

    img = torch.randn((batch_size, channels, img_size, img_size), device=device)

    for i in tqdm(reversed(range(timesteps)), desc="VPSDE Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # --- Predictor Step (Reverse SDE) ---
        beta_t = betas[i]
        score = model(img, t, conds)
        drift = -0.5 * beta_t * img
        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion ** 2 * score
        img = img + drift
        if i != 0:
            img += diffusion * torch.randn_like(img)

        # --- Corrector Step (Langevin Dynamics) ---
        for _ in range(corrector_steps):
            noise = torch.randn_like(img)
            grad = model(img, t, conds)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            img = img + step_size * grad + torch.sqrt(2 * step_size) * noise

    return img


@torch.no_grad()
def edm_sampler(model, cfg, device, conds=None, s_churn=0., s_min=0., s_max=float('inf'), s_noise=1.):
    """
    Sampler for EDM models, which predict the denoised image x_0.
    Uses an ODE solver approach with a sigma schedule.
    """
    # EDM uses a sigma schedule instead of beta/alpha
    num_steps = cfg.diffusion.timesteps
    sigma_min = cfg.diffusion.sigma_min
    sigma_max = cfg.diffusion.sigma_max
    rho = 7.0  # A hyperparameter for the sigma schedule

    sigmas = torch.from_numpy(
        (sigma_max ** (1 / rho) + torch.arange(num_steps) / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    ).to(device)

    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels

    img = torch.randn((batch_size, channels, img_size, img_size), device=device) * sigmas[0]

    for i in tqdm(range(num_steps - 1), desc="EDM Sampling"):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        t = torch.full((batch_size,), i, device=device,
                       dtype=torch.long)  # EDM models might still use integer timesteps

        # Predict the denoised image
        denoised = model(img, t, conds)

        # ODE step (Heun's method)
        d = (img - denoised) / sigma_cur
        img_next = img + d * (sigma_next - sigma_cur)

        # Apply second-order correction
        if i < num_steps - 2:
            t_next = torch.full((batch_size,), i + 1, device=device, dtype=torch.long)
            denoised_next = model(img_next, t_next, conds)
            d_next = (img_next - denoised_next) / sigma_next
            img_next = img + (d + d_next) * ((sigma_next - sigma_cur) / 2)

        img = img_next

    return img


@torch.no_grad()
def composable_sampler(model, cfg, device, conds_list, weights):
    """
    Sampler for ComposableDiffusionModel.
    Combines noise predictions from multiple conditions.
    `conds_list`: A list of condition lists, e.g., [[cond_color], [cond_digit]].
    `weights`: A list of weights for combining the noise predictions.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]

    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels
    timesteps = cfg.diffusion.timesteps

    img = torch.randn((batch_size, channels, img_size, img_size), device=device)

    for i in tqdm(reversed(range(timesteps)), desc="Composable Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # Predict noise for each condition and combine them
        total_predicted_noise = torch.zeros_like(img)
        for conds, weight in zip(conds_list, weights):
            total_predicted_noise += weight * model(img, t, conds)

        predicted_noise = total_predicted_noise

        # Standard DDPM update step
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = constants["posterior_variance"][i]
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img


@torch.no_grad()
def cascaded_sampler(base_model, super_res_model, cfg_base, cfg_super_res, device, conds=None):
    """
    Sampler for a two-stage cascaded diffusion pipeline.
    """
    # 1. Generate low-resolution image with the base model
    print("--- Generating low-resolution base image ---")
    low_res_img = ddpm_sampler(base_model, cfg_base, device, conds)

    # 2. Upsample it to the target resolution
    low_res_upsampled = F.interpolate(
        low_res_img,
        size=cfg_super_res.dataset.image_size,
        mode='bilinear',
        align_corners=False
    )

    # 3. Use the super-resolution model to refine the upsampled image
    # The super-res model is conditioned on the upsampled image.
    print("--- Running super-resolution model ---")
    # This requires a modified sampler that takes the low-res image as an additional condition
    # For simplicity, we'll use a standard DDPM sampler where the super-res model
    # is architecturally designed to accept the low-res image.

    constants = get_diffusion_constants(cfg_super_res, device)
    betas = constants["betas"]
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    timesteps = cfg_super_res.diffusion.timesteps
    batch_size = cfg_super_res.sampling.batch_size

    img = torch.randn_like(low_res_upsampled)  # Start from noise at the high resolution

    for i in tqdm(reversed(range(timesteps)), desc="Cascaded Super-Res Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # The super_res_model must accept the noisy high-res image `img`
        # and the upsampled low-res image `low_res_upsampled` as conditions.
        predicted_noise = super_res_model(img, t, low_res_upsampled)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = constants["posterior_variance"][i]
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img


@torch.no_grad()
def ldm_sampler(ldm_unet, vae, cfg, device, conds=None):
    """
    Sampler for Latent Diffusion Models (LDMs).
    Operates in latent space and decodes at the end.
    """
    # The DDPM sampling logic is identical, but happens in the latent space
    latent_channels = vae.encoder.z_channels
    latent_size = cfg.dataset.image_size // 8  # Common downsampling factor

    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    timesteps = cfg.diffusion.timesteps
    batch_size = cfg.sampling.batch_size

    # Start with noise in the latent space
    latents = torch.randn((batch_size, latent_channels, latent_size, latent_size), device=device)

    for i in tqdm(reversed(range(timesteps)), desc="LDM Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        predicted_noise = ldm_unet(latents, t, conds)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]

        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (latents - coeff_pred_noise * predicted_noise)

        if i == 0:
            latents = model_mean
        else:
            posterior_variance_t = constants["posterior_variance"][i]
            noise = torch.randn_like(latents)
            latents = model_mean + torch.sqrt(posterior_variance_t) * noise

    # Decode the final latents back to pixel space
    # The scaling factor is specific to the VAE used (e.g., Stability AI's)
    latents = latents / 0.18215
    images = vae.decode(latents)
    return images


@torch.no_grad()
def composable_ldm_sampler(
        model: ComposableUnet,
        vae: AutoencoderKL,
        cfg,
        device,
        digit_labels,
        digit_color_labels,
        bbox_color_labels,
        guidance_scale=7.5
):
    """
    Conditional LDM sampler for ComposableUnet in latent space, using CFG.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    posterior_variance = constants["posterior_variance"]
    timesteps = cfg.diffusion.timesteps
    batch_size = digit_labels.shape[0]

    # Determine latent shape from a dummy forward pass through VAE
    dummy_input = torch.zeros(1, cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size, device=device)
    dummy_posterior = vae.encode(dummy_input)
    latent_shape = (batch_size,) + tuple(dummy_posterior.mean.shape[1:])

    # Start with random noise in the latent space
    z = torch.randn(latent_shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='Composable LDM Sampling', total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # 1. Predict noise with conditioning
        predicted_noise_cond = model(z, t, digit_labels, digit_color_labels, bbox_color_labels)

        # 2. Predict noise without conditioning
        uncond_digit_labels = torch.full_like(digit_labels, 10)
        uncond_digit_color_labels = torch.full_like(digit_color_labels, 10)
        uncond_bbox_color_labels = torch.full_like(bbox_color_labels, 10)
        predicted_noise_uncond = model(z, t, uncond_digit_labels, uncond_digit_color_labels, uncond_bbox_color_labels)

        # 3. Combine using CFG
        noise_pred = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)

        # 4. Denoise latent for one step
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        coeff_z = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_z * (z - coeff_pred_noise * noise_pred)

        if i == 0:
            z = model_mean
        else:
            noise = torch.randn_like(z)
            z = model_mean + torch.sqrt(posterior_variance[i]) * noise

    # Decode the final latent vector to get the image
    z = z / 0.18215  # Rescale latent before decoding
    image = vae.decode(z)

    return image

@torch.no_grad()
def classifier_guided_sampler(model, classifier, cfg, device, conds=None, guidance_scale=7.5):
    """
    DDPM sampler with classifier guidance.
    """
    constants = get_diffusion_constants(cfg, device)
    alphas = constants["alphas"]
    alphas_cumprod = constants["alphas_cumprod"]
    posterior_variance = constants["posterior_variance"]

    batch_size = cfg.sampling.batch_size
    img_size = cfg.dataset.image_size
    channels = cfg.dataset.channels
    timesteps = cfg.diffusion.timesteps

    # Assume conds[0] contains the target class labels for guidance
    target_classes = conds[0]

    img = torch.randn((batch_size, channels, img_size, img_size), device=device)

    for i in tqdm(reversed(range(timesteps)), desc="Classifier-Guided Sampling", total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # 1. Get the original noise prediction
        predicted_noise = model(img, t, conds)

        # 2. Compute the guidance gradient
        with torch.enable_grad():
            img_in = img.detach().requires_grad_(True)
            logits = classifier(img_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), target_classes.view(-1)]
            gradient = torch.autograd.grad(selected.sum(), img_in)[0]

        # 3. Modify the predicted noise with the guidance
        # The gradient is scaled by the guidance scale and posterior variance
        modified_noise = predicted_noise - torch.sqrt(posterior_variance[i]) * guidance_scale * gradient

        # 4. Denoise using the modified noise
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * modified_noise)

        if i == 0:
            img = model_mean
        else:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance[i]) * noise

    return img


@torch.no_grad()
def guided_sampler(model, classifier, cfg, device, conds=None, guidance_scale=7.5):
    """
    DDPM sampler with classifier guidance. Alias for classifier_guided_sampler.
    """
    return classifier_guided_sampler(model, classifier, cfg, device, conds, guidance_scale)


@torch.no_grad()
def moe_sampler(moe_model, cfg, device, conds=None):
    """
    Sampler for Mixture-of-Experts (MoE) diffusion models.
    The MoE model itself handles the routing to different experts based on timestep.
    """
    # The sampling logic is identical to DDPM. The MoE model's forward pass
    # internally selects the correct expert based on the timestep `t`.
    print("Using MoE sampler (equivalent to DDPM, expert routing is internal to the model)")
    return ddpm_sampler(moe_model, cfg, device, conds)
