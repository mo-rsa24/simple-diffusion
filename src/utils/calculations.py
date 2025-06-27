import torch
from src.models.builders.optimizers import linear_beta_schedule
import torch.nn.functional as F

from src.utils.image_manipulation import extract
from src.utils.transforms import reverse_transform

get_betas = lambda timesteps: linear_beta_schedule(timesteps=timesteps)
def get_alphas(betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    return alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas

get_calculations = lambda alphas_cumprod: (torch.sqrt(alphas_cumprod), torch.sqrt(1. - alphas_cumprod))
get_posterior_variance = lambda betas, alphas_cumprod_prev, alphas_cumprod: betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None, timesteps = 300):
    if noise is None:
        noise = torch.randn_like(x_start)
    betas = get_betas(timesteps)
    alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas = get_alphas(betas=betas)

    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod  = get_calculations(alphas_cumprod)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image
