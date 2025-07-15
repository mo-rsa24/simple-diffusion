import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.calculations import q_sample, get_betas, get_alphas, get_calculations, get_posterior_variance
from src.utils.image_manipulation import extract


@torch.no_grad()
def p_sample(model, x, t, t_index, timesteps: int = 300, beta_start: float = 0.0001, beta_end: float = 0.02):
    betas = get_betas(timesteps, beta_start, beta_end)
    alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas = get_alphas(betas=betas)
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_calculations(alphas_cumprod)
    posterior_variance = get_posterior_variance(betas, alphas_cumprod_prev, alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", timesteps: int = 300, beta_start: float = 0.0001, beta_end: float = 0.02):
    if noise is None:
        noise = torch.randn_like(x_start)

    # x_noisy = q_sample(x_start=x_start, t=t, noise=noise, timesteps=timesteps)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise, timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample_loop(
    model,
    shape,
    timesteps: int = 300,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    device: torch.device | None = None,
):
    """Reverse diffusion process for arbitrary score functions.

    The ``model`` argument can be either a ``nn.Module`` or any callable
    returning the predicted noise. When ``device`` is ``None`` and ``model``
    is a module, the parameters of ``model`` are used to infer the device.
    """
    if device is None:
        if hasattr(model, "parameters"):
            device = next(model.parameters()).device
        else:
            device = torch.device("cpu")

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, timesteps=timesteps,
                       beta_start=beta_start, beta_end=beta_end)
    return img


@torch.no_grad()
def sample(
    model,
    image_size,
    batch_size: int = 16,
    channels: int = 3,
    timesteps: int = 300,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    device: torch.device | None = None,
):
    return p_sample_loop(
        model,
        shape=(batch_size, channels, image_size, image_size),
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )


@torch.no_grad()
def generate_batch(model,
                   image_size: int,
                   batch_size: int = 16,
                   channels: int = 1,
                   timesteps: int = 300,
                   beta_start: float = 0.0001,
                   beta_end: float = 0.02,
                   device: torch.device = None) -> torch.Tensor:
    if device is None:
        if hasattr(model, "parameters"):
            device = next(model.parameters()).device
        else:
            device = torch.device("cpu")
    final = p_sample_loop(
        model,
        shape=(batch_size, channels, image_size, image_size),
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
    return final.to(device)
