import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.vanilla.diffusion import p_sample
from src.utils.calculations import q_sample

def latent_p_losses(denoise_model, encoder, x_start, t, loss_type="l2", timesteps=300):
    with torch.no_grad():
        z_start = encoder(x_start)

    noise = torch.randn_like(z_start)
    z_noisy = q_sample(x_start=z_start, t=t, noise=noise, timesteps=timesteps)
    predicted_noise = denoise_model(z_noisy, t)

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
def latent_sample(model, decoder, latent_shape, timesteps=300):
    device = next(model.parameters()).device
    z = torch.randn(latent_shape).to(device)

    for i in tqdm(reversed(range(0, timesteps)), desc='Latent Sampling', total=timesteps):
        t = torch.full((z.size(0),), i, device=device, dtype=torch.long)
        z = p_sample(model, z, t, i, timesteps=timesteps)

    recon = decoder(z)
    return recon
