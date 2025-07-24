import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from box import Box
import math
from pathlib import Path
import time
from datetime import timedelta
from tqdm.auto import tqdm
import numpy as np


# ==============================================================================
# 1. MODEL AND SDE DEFINITIONS (Unchanged)
# ==============================================================================
class VPSDE:
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, num_timesteps: int = 1000, device='cpu'):
        self.beta_min, self.beta_max, self.num_timesteps, self.device = beta_min, beta_max, num_timesteps, device
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    # The reverse SDE requires f(x,t) and g(t)
    def f(self, x, t): return -0.5 * self.betas[t].view(-1, 1, 1, 1) * x

    def g(self, t): return torch.sqrt(self.betas[t])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__();
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device;
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :];
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__();
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1); self.transform = nn.ConvTranspose2d(out_ch, out_ch,
                                                                                                         4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1); self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1);
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch);
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)));
        time_emb = self.relu(self.time_mlp(t));
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb;
        h = self.bnorm2(self.relu(self.conv2(h)));
        return self.transform(h)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__();
        self.time_mlp = nn.Linear(time_emb_dim, out_ch);
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1);
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch);
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)));
        time_emb = self.relu(self.time_mlp(t));
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb;
        h = self.bnorm2(self.relu(self.conv2(h)));
        return h


class ScoreModel(nn.Module):
    """The U-Net model for predicting score (noise)."""

    def __init__(self, in_channels: int = 3, time_emb_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4),
                                      nn.ReLU(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.initial_conv = nn.Conv2d(in_channels, 32, 3, padding=1);
        self.down1 = Block(32, 64, time_emb_dim);
        self.down2 = Block(64, 128, time_emb_dim)
        self.bot1 = Block(128, 256, time_emb_dim);
        self.up_transpose_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1);
        self.up_block_1 = ConvBlock(256, 128, time_emb_dim)
        self.up_transpose_2 = nn.ConvTranspose2d(128, 64, 4, 2, 1);
        self.up_block_2 = ConvBlock(128, 64, time_emb_dim)
        self.up_transpose_3 = nn.ConvTranspose2d(64, 32, 4, 2, 1);
        self.up_block_3 = ConvBlock(64, 32, time_emb_dim);
        self.output = nn.Conv2d(32, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t);
        x1 = self.initial_conv(x);
        x2 = self.down1(x1, t_emb);
        x3 = self.down2(x2, t_emb);
        x_bot = self.bot1(x3, t_emb)
        u1 = self.up_transpose_1(x_bot);
        u1_cat = torch.cat([u1, x3], dim=1);
        u1_out = self.up_block_1(u1_cat, t_emb)
        u2 = self.up_transpose_2(u1_out);
        u2_cat = torch.cat([u2, x2], dim=1);
        u2_out = self.up_block_2(u2_cat, t_emb)
        u3 = self.up_transpose_3(u2_out);
        u3_cat = torch.cat([u3, x1], dim=1);
        u3_out = self.up_block_3(u3_cat, t_emb)
        return self.output(u3_out)


# ==============================================================================
# 2. CUSTOM DATASET (Unchanged)
# ==============================================================================
class ColoredMNIST(Dataset):
    def __init__(self, image_size, target_digits=None, color_override=None):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if target_digits is None or label in target_digits]
        self.mnist_dataset = mnist;
        self.color_override = color_override
        self.color_map = {0: (.5, .5, .5), 1: (0, .5, 1), 2: (0, .8, 0), 3: (0, .8, .8), 4: (1, .5, 0), 5: (1, 1, 0),
                          6: (1, 0, 0), 7: (1, 0, 1), 8: (.5, 0, 1), 9: (.6, .3, .1)}

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]];
        image_tensor = self.transform(image)
        color = self.color_override if self.color_override is not None else self.color_map[label]
        colored_image = image_tensor.repeat(3, 1, 1) * torch.tensor(color).view(3, 1, 1)
        return (colored_image * 2) - 1, label


# ==============================================================================
# 3. LAYOUTDIFF SAMPLER AND HELPERS
# ==============================================================================
class LayoutDiff:
    """A diffusion sampler that composes models based on spatial masks."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    @torch.no_grad()
    def sample(self, models: list[nn.Module], masks: list[torch.Tensor], shape: tuple, device: str) -> torch.Tensor:
        if len(models) != len(masks):
            raise ValueError("The number of models and masks must be equal.")

        x = torch.randn(shape, device=device)

        # Pre-calculate the final, non-overlapping mask for each model's score.
        # The last model in the list is treated as being on top.
        final_masks = [torch.zeros_like(m) for m in masks]
        occlusion_mask = torch.zeros_like(masks[0])  # Keeps track of area already claimed
        for i in range(len(masks) - 1, -1, -1):
            # The unique region is the mask's area minus what's already covered by models on top.
            unique_region = torch.clamp(masks[i] - occlusion_mask, 0, 1)
            final_masks[i] = unique_region
            occlusion_mask += unique_region

        # Add batch and channel dimensions for broadcasting and move to device
        final_masks = [m.to(device).unsqueeze(0).unsqueeze(0) for m in final_masks]

        # Standard reverse-time diffusion loop
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Layout-Aware Sampling", leave=False):
            t_idx = timesteps[i]
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            # --- Spatially-Aware Score Composition ---
            # Get score from each model and apply it only in its designated region.
            combined_noise_pred = torch.zeros_like(x)
            for model, mask in zip(models, final_masks):
                model.eval()
                # The model predicts the noise, which is proportional to the score.
                noise_pred = model(x, t.float())
                combined_noise_pred += noise_pred * mask

            # --- Standard DDPM-style Reverse Step ---
            # This is equivalent to the reverse SDE step but using the DDPM formulation
            # which is simpler to implement from the existing VPSDE class.
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_bar_t = self.sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)

            # Get the predicted x0 and then the mean of the posterior q(x_{t-1} | x_t, x_0)
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * combined_noise_pred) / sqrt_alpha_bar_t
            pred_x0 = torch.clamp(pred_x0, -1., 1.)

            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            alpha_bar_prev = self.sde.alphas_cumprod_prev[t].view(-1, 1, 1, 1)

            posterior_mean = (torch.sqrt(alpha_bar_prev) * beta_t / (
                        1. - self.sde.alphas_cumprod[t].view(-1, 1, 1, 1))) * pred_x0 + \
                             (torch.sqrt(self.sde.alphas[t].view(-1, 1, 1, 1)) * (1. - alpha_bar_prev) / (
                                         1. - self.sde.alphas_cumprod[t].view(-1, 1, 1, 1))) * x

            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(x)
                x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x_prev = posterior_mean

            x = x_prev

        return x.clamp(-1, 1)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: center = (int(w / 2), int(h / 2))
    if radius is None: radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return torch.from_numpy(mask.astype(float))


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir);
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, model_name):
        save_path = self.checkpoint_dir / f"{model_name}.pth";
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def load(self, model, model_name, device):
        load_path = self.checkpoint_dir / f"{model_name}.pth"
        if not load_path.exists(): raise FileNotFoundError(f"Checkpoint {load_path} not found.")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded model checkpoint from {load_path}");
        return model


def train(cfg, model, sde, train_loader, device, model_name, ckpt_mgr):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr)
    print(f"--- Starting Training for {model_name} ---")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"):
            optimizer.zero_grad();
            x0 = images.to(device)
            t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0);
            sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            predicted_noise = model(xt, t.float());
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward();
            optimizer.step()
    print(f"--- Finished Training for {model_name} ---");
    ckpt_mgr.save(model, model_name)


if __name__ == '__main__':
    # --- Configuration ---
    cfg = Box({
        "exp_name": "layout",
        "dataset": {
            "image_size": 32, "channels": 3,
            "model_A_digit": [6], "model_A_color": (1.0, 0.0, 0.0),  # Red 6
            "model_B_digit": [2], "model_B_color": (0.0, 0.8, 0.0),  # Green 2
        },
        "training": {"do_train": True, "epochs": 100, "batch_size": 128},
        "optimizer": {"params": {"lr": 2e-4}},
        "sampling": {"batch_size": 4}
    })

    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dirs = {'ckpt': Path('./checkpoints'), 'viz': Path('./visualizations')}
    dirs['ckpt'].mkdir(exist_ok=True);
    dirs['viz'].mkdir(exist_ok=True)
    ckpt_mgr = CheckpointManager(checkpoint_dir=dirs['ckpt'])
    sde = VPSDE(device=device)

    # --- Datasets and Training ---
    dataset_A = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.model_A_digit,
                             color_override=cfg.dataset.model_A_color)
    dataset_B = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.model_B_digit,
                             color_override=cfg.dataset.model_B_color)
    loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)

    model_A = ScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_A_name, model_B_name = f"model_A_{cfg.exp_name}", f"model_B_{cfg.exp_name}"

    if cfg.training.do_train:
        train(cfg, model_A, sde, loader_A, device, model_A_name, ckpt_mgr)
        train(cfg, model_B, sde, loader_B, device, model_B_name, ckpt_mgr)

    # --- Inference ---
    print("\n--- 🎬 Starting Layout-Aware Inference ---")
    model_A = ckpt_mgr.load(model_A, model_A_name, device)
    model_B = ckpt_mgr.load(model_B, model_B_name, device)

    # --- Create Masks for Layout Control ---
    H, W = cfg.dataset.image_size, cfg.dataset.image_size
    # Mask for Green 2 (background)
    mask_B = create_circular_mask(H, W, center=(W // 2, H // 2), radius=H // 3)
    # Mask for Red 6 (foreground, on top)
    mask_A = create_circular_mask(H, W, center=(W // 3, H // 3), radius=H // 4)

    # The order matters: last model/mask is on top.
    # Model B (green 2) is the background, Model A (red 6) is the foreground.
    models_for_layout = [model_B, model_A]
    masks_for_layout = [mask_B, mask_A]

    # --- Instantiate and Run the Sampler ---
    layout_sampler = LayoutDiff(sde)
    layout_samples = layout_sampler.sample(
        models=models_for_layout,
        masks=masks_for_layout,
        shape=(cfg.sampling.batch_size, cfg.dataset.channels, H, W),
        device=device
    )


    # --- Visualize the Results ---
    def norm(s):
        return (s.clamp(-1, 1) + 1) / 2


    # Also visualize the masks themselves to understand the layout
    mask_viz = torch.stack([mask_B, mask_A]).unsqueeze(1).repeat(1, 3, 1, 1).float()

    output_path = dirs['viz'] / f"experiment_{cfg.exp_name}_results.png"
    torchvision.utils.save_image(
        torch.cat([norm(layout_samples), mask_viz]),
        output_path,
        nrow=cfg.sampling.batch_size
    )
    print(f"Saved layout composition to {output_path}")
    print("Top row: Generated images. Bottom row: Masks used (Green=Model B, Red=Model A).")