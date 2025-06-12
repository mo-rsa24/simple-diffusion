# src/utils/log.py

import wandb
import math
from torchvision.utils import make_grid

def log_grid_images(generated, real, global_step, writer):
    grid = make_grid(generated, nrow=int(math.sqrt(len(generated))), normalize=True)
    writer.add_image("generated/grid", grid, global_step)
    if real is not None:
        real_grid = make_grid(real[:len(generated)],
                              nrow=int(math.sqrt(len(generated))), normalize=True)
        writer.add_image("real/grid", real_grid, global_step)

def log_table_wandb(generated, real, global_step, wandb_run):
    cols = ["index", "real", "generated"]
    table = wandb.Table(columns=cols)
    n = min(len(generated), real.shape[0] if real is not None else len(generated))
    for i in range(n):
        real_img = wandb.Image(real[i].cpu()) if real is not None else None
        gen_img = wandb.Image(generated[i].cpu())
        table.add_data(i, real_img, gen_img)
    wandb_run.log({"samples_comparison": table}, step=global_step)

def log_metrics_wandb(wandb_tracker, metrics: dict, step: int):
    if wandb_tracker is None:
        return
    wandb_tracker.log(metrics, step=step)
