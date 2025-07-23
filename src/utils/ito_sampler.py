import torch
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.models.vpsde.ColoredMNISTScoreModel import VPSDE


class SuperDiff:
    """
    PyTorch implementation of the SUPERDIFF algorithm for OR and AND composition.
    """

    def __init__(self,
                 score_models: list[nn.Module],
                 sde: VPSDE,
                 temperature: float = 1.0,
                 bias: float = 0.0):
        """
        Args:
            score_models (list[nn.Module]): A list of M pre-trained score models.
            sde (VPSDE): An instance of the SDE helper class.
            temperature (float): The temperature parameter T.
            bias (float): The bias parameter l.
        """
        self.score_models = nn.ModuleList(score_models)
        self.M = len(score_models)
        self.sde = sde
        self.T = temperature
        self.l = bias

    @torch.no_grad()
    def sample(self,
               op: str,
               shape: tuple,
               num_steps: int = 1000,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
        """
        Generates samples using the SUPERDIFF reverse-time SDE solver.

        Args:
            op (str): The composition operation, either 'OR' or 'AND'.
            shape (tuple): The shape of the desired output tensor, e.g., (B, C, H, W).
            num_steps (int): The number of discretization steps for the solver.
            device (str): The device to run the sampling on.

        Returns:
            torch.Tensor: The generated sample x_T.
        """
        if op.upper() not in ['OR', 'AND']:
            raise ValueError("Operation 'op' must be 'OR' or 'AND'.")
        op = op.upper()

        # 1. Initialization
        x = torch.randn(shape, device=device)  # Initial noise z ~ N(0, I)

        # Initialize log q_t^i(x_t) values for OR operation
        log_q_vals = torch.zeros(shape[0], self.M, device=device)

        time_steps = torch.linspace(0., 1., num_steps + 1, device=device)

        # 2. Main Sampling Loop (iterating τ from 0 to 1)
        for i in tqdm(range(num_steps), desc=f"SUPERDIFF Sampling ({op})", leave=False):
            tau = time_steps[i]
            d_tau = time_steps[i + 1] - tau

            # The corresponding time `t` in the forward SDE is `1 - τ`
            t = torch.full((shape[0],), 1. - tau, device=device)

            # --- Compute weights κ (kappa) ---
            if op == 'OR':
                # κ_t^i = softmax(T * log q_t^i(x_τ) + l)
                logits = self.T * log_q_vals + self.l
                kappa = torch.softmax(logits, dim=-1)
            else:  # op == 'AND'
                # Use simple average as described in the paper's text for Algorithm 1
                kappa = torch.full((shape[0], self.M), 1.0 / self.M, device=device)

            # --- Compute combined score u_t(x_τ) ---
            # Get score from each model.
            all_scores = torch.stack([model(x, t) for model in self.score_models], dim=1)

            # Reshape kappa for broadcasting: (B, M) -> (B, M, 1, 1, ...)
            kappa_reshaped = kappa.view(shape[0], self.M, *([1] * (x.dim() - 1)))
            u_t = torch.sum(kappa_reshaped * all_scores, dim=1)

            # --- Update x_τ using the reverse SDE step (Euler-Maruyama) ---
            f, g = self.sde.f(x, t), self.sde.g(t)
            g_reshaped = g.view(-1, *([1] * (x.dim() - 1)))

            drift = -f + g_reshaped.pow(2) * u_t
            noise = torch.randn_like(x)
            diffusion = g_reshaped * torch.sqrt(d_tau) * noise

            dx = drift * d_tau + diffusion

            # --- Update log q values for the next step (for OR case only) ---
            if op == 'OR':
                # Using the simplified update rule from the paper's appendix
                # d log q_i ≈ <dx, score_i>
                dx_flat = dx.flatten(start_dim=1)
                scores_flat = all_scores.flatten(start_dim=2)

                # Batch dot product via einsum
                d_log_q = torch.einsum('bd,bmd->bm', dx_flat, scores_flat)
                log_q_vals = log_q_vals + d_log_q

            # Apply the update
            x = x + dx

        return x