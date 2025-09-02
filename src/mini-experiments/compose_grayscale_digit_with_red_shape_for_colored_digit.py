import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
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
# (VPSDE, SinusoidalPosEmb, ColoredMNISTScoreModel, etc. are included here)
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


class ColoredMNISTScoreModel(nn.Module):
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
# 2. CUSTOM DATASETS
# ==============================================================================
class GrayscaleMNIST(Dataset):
    """MNIST dataset filtered by digit, but kept as 3-channel grayscale."""

    def __init__(self, image_size, target_digits):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        mnist = datasets.MNIST(root='../../data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if label in target_digits]
        self.mnist_dataset = mnist

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]]
        image_tensor = self.transform(image)
        # Repeat the single channel to get a 3-channel grayscale image and normalize to [-1, 1]
        final_image = (image_tensor.repeat(3, 1, 1) * 2) - 1
        return final_image, label


class SimpleShapesDataset(Dataset):
    """A synthetic dataset of simple, solid-colored shapes."""

    def __init__(self, image_size, num_samples, shape_color):
        self.image_size = image_size
        self.num_samples = num_samples
        self.shape_color = torch.tensor(shape_color).float().view(3, 1, 1)  # [C, H, W]

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # Create a black canvas
        image = torch.zeros(3, self.image_size, self.image_size)
        # Draw a square of random size and position
        square_size = np.random.randint(self.image_size // 4, self.image_size // 2)
        x_start = np.random.randint(0, self.image_size - square_size)
        y_start = np.random.randint(0, self.image_size - square_size)
        image[:, y_start:y_start + square_size, x_start:x_start + square_size] = self.shape_color
        # Normalize to [-1, 1]
        return (image * 2) - 1, 0  # Label is unused


# ==============================================================================
# 3. TRAINING AND SAMPLING (Mostly Unchanged)
# ==============================================================================

class CheckpointManager:
    """A simple checkpoint manager."""

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


class SuperDiffSampler:
    """Implements the SUPERDIFF algorithm for composing two pre-trained models."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    @torch.no_grad()
    def sample(self, model1, model2, batch_size, shape, device, operation='OR', temp=1.0, bias=0.0):
        model1.eval();
        model2.eval();
        x = torch.randn((batch_size, *shape), device=device)
        log_q1 = torch.zeros(batch_size, device=device);
        log_q2 = torch.zeros(batch_size, device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc=f"SUPERDIFF Sampling ({operation})", leave=False):
            t_idx = timesteps[i];
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise1, noise2 = model1(x, t.float()), model2(x, t.float());
            score1, score2 = -noise1 / sqrt_one_minus_alpha_bar_t, -noise2 / sqrt_one_minus_alpha_bar_t
            if operation.upper() == 'OR':
                logits = torch.stack([log_q1, log_q2], dim=1);
                kappas = F.softmax(temp * logits + bias, dim=1);
                kappa1, kappa2 = kappas[:, 0].view(-1, 1, 1, 1), kappas[:, 1].view(-1, 1, 1, 1)
            elif operation.upper() == 'AND':
                # Heuristic to balance log-densities, pushing towards an equal density state.
                # A rigorous implementation solves the linear system in Prop. 6 of the paper [cite: 207-210].
                probs = F.softmax(torch.stack([-log_q1, -log_q2], dim=1), dim=1)
                kappa1, kappa2 = probs[:, 0].view(-1, 1, 1, 1), probs[:, 1].view(-1, 1, 1, 1)
            else:
                kappa1, kappa2 = 0.5, 0.5
            combined_score = kappa1 * score1 + kappa2 * score2;
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1);
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            mean = (1 / sqrt_alpha_t) * (x + beta_t * combined_score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1);
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            dx = x_prev - x;
            dtau = 1.0 / self.sde.num_timesteps;
            d = x.shape[1] * x.shape[2] * x.shape[3];
            div_f = -0.5 * beta_t.squeeze() * d

            def update_log_q(log_q, score):
                term1 = torch.sum(dx * score, dim=[1, 2, 3]);
                f_term = -0.5 * beta_t * x;
                g_sq_term = beta_t
                inner_prod_term = torch.sum((f_term - 0.5 * g_sq_term * score) * score, dim=[1, 2, 3])
                return log_q + term1 + (div_f + inner_prod_term) * dtau

            log_q1, log_q2 = update_log_q(log_q1, score1), update_log_q(log_q2, score2);
            x = x_prev
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_single_model(self, model, batch_size, shape, device):
        model.eval();
        x = torch.randn((batch_size, *shape), device=device);
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Single Model Sampling", leave=False):
            t_idx = timesteps[i];
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1);
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1);
            noise = model(x, t.float())
            score = -noise / sqrt_one_minus_alpha_bar_t;
            mean = (1 / sqrt_alpha_t) * (x + beta_t * score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1);
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            x = x_prev
        return x.clamp(-1, 1)


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================

### CHOOSE THE EXPERIMENT TO RUN ###
EXPERIMENT_TO_RUN = 'DISENTANGLE'  # Can be 'CIFAR10', 'MNIST', or 'DISENTANGLE'


def visualize_results(samples_a, samples_b, superdiff_samples, output_path):
    """Saves a grid comparing the three sets of samples."""

    def norm(s): return (s.clamp(-1, 1) + 1) / 2

    torchvision.utils.save_image(torch.cat([
        norm(samples_a), norm(samples_b), norm(superdiff_samples)
    ]), output_path, nrow=samples_a.shape[0])
    print(f"Saved final comparison to {output_path} (Top: Model A, Middle: Model B, Bottom: Composed)")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dirs = {'ckpt': Path('./checkpoints'), 'viz': Path('./visualizations')}
    dirs['ckpt'].mkdir(exist_ok=True);
    dirs['viz'].mkdir(exist_ok=True)
    ckpt_mgr = CheckpointManager(checkpoint_dir=dirs['ckpt'])
    sde = VPSDE(device=device)

    if EXPERIMENT_TO_RUN.upper() == 'DISENTANGLE':
        cfg = Box({
            "exp_name": "disentangle",
            "dataset": {"image_size": 32, "channels": 3, "content_digit": [7], "style_color": (1.0, 0.0, 0.0)},
            "training": {"do_train": True, "epochs": 2, "batch_size": 128}, "optimizer": {"params": {"lr": 2e-4}},
            "sampling": {"batch_size": 8}
        })
        print(f"--- 🧪 RUNNING DISENTANGLEMENT EXPERIMENT (Digit: {cfg.dataset.content_digit[0]}, Color: Red) ---")
        dataset_A = GrayscaleMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.content_digit)
        dataset_B = SimpleShapesDataset(cfg.dataset.image_size, num_samples=len(dataset_A),
                                        shape_color=cfg.dataset.style_color)
        loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
        loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)
        operation = 'AND'

    # Placeholder for other experiments
    else:
        raise ValueError("EXPERIMENT_TO_RUN must be 'DISENTANGLE' for this request.")

    model_A = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_A_name, model_B_name = f"model_A_{cfg.exp_name}", f"model_B_{cfg.exp_name}"

    if cfg.training.do_train:
        train(cfg, model_A, sde, loader_A, device, model_A_name, ckpt_mgr)
        train(cfg, model_B, sde, loader_B, device, model_B_name, ckpt_mgr)

    print("\n--- 🎬 Starting Inference Phase ---")
    model_A = ckpt_mgr.load(model_A, model_A_name, device)
    model_B = ckpt_mgr.load(model_B, model_B_name, device)

    sampler = SuperDiffSampler(sde)
    shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)

    print("Generating samples from Model A (Content: Grayscale 7s)...")
    samples_A = sampler.sample_single_model(model_A, cfg.sampling.batch_size, shape, device)

    print("Generating samples from Model B (Style: Red Squares)...")
    samples_B = sampler.sample_single_model(model_B, cfg.sampling.batch_size, shape, device)

    print(f"Generating composed samples using SUPERDIFF '{operation}'...")
    composed_samples = sampler.sample(model_A, model_B, cfg.sampling.batch_size, shape, device, operation)

    output_path = dirs['viz'] / f"experiment_{cfg.exp_name}_results.png"
    visualize_results(samples_A, samples_B, composed_samples, output_path)