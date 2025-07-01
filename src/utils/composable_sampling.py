from src.models.composable_diffusion import ComposableDiffusionModel
from src.models.vanilla.diffusion import p_sample_loop


def sample_with_fixed_slots(model: ComposableDiffusionModel, slot_tensors, timesteps: int, beta_start: float, beta_end: float):
    """Generate an image by fixing each slot tensor and running the reverse process."""
    merged = model.sample_slots(slot_tensors["shape"], slot_tensors["color"], slot_tensors["box"])
    return p_sample_loop(lambda x, t, t_index: merged, shape=merged.shape, timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)


def swap_slot_latents(latent_a, latent_b):
    return latent_b.clone(), latent_a.clone()