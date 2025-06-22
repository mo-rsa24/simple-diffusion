import argparse

import torch

from src.dataset.ChestXRay import get_xray_loaders
from src.dataset.MNIST import get_mnist_loaders
from src.models.ema import EMA
from src.models.unet import Unet
from src.monitoring.alert_notifier import send_failure_email
from src.train.logging.training_logger_utils import log_exception, log_batch
from src.train.train import train

from src.utils.setup import load_config, build_dirs, init_observers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", "-e", type=str, required=True)
    p.add_argument("--run_id", "-r", type=str, required=True)
    p.add_argument(
        "--dataset", "-d",
        choices=["TB", "PNEUMONIA", "MNIST"],
        required=True,
        help="Which dataset/config to use"
    )
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print merged config and exit"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_path = f"src/config/{args.dataset.lower()}.yml"
    cfg = load_config(config_path, args.experiment_id, args.run_id)
    dirs = build_dirs(cfg)
    logger, writer, wandb_run = init_observers(cfg, dirs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_mnist_loaders(cfg)
    model = Unet(**cfg.model.params).to(device)
    ema = EMA(model, decay=cfg.diffusion.ema_decay)

    try:
        train(cfg, dirs, model, ema, train_loader, logger, device, writer = writer, wandb_run = wandb_run)
    except Exception as e:
        log_exception(logger, exception= e)
        send_failure_email(run_id=cfg.run_id, reason=str(e), epoch=0)