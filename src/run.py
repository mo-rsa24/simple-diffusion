import argparse
import torch
from src.models.ema import EMA
from src.models.unet import Unet
from src.monitoring.alert_notifier import send_failure_email
from src.registry.mappings import DATASET_LOADERS, MODEL_REGISTRY
from src.task import generate_task, classify_task
from src.train.logging.training_logger_utils import log_exception
from src.utils.setup import load_config, build_dirs, init_observers
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", "-e", type=str, required=True)
    p.add_argument("--run_id", "-r", type=str, required=True)
    p.add_argument(
        "--dataset", "-d",
        choices=["TB", "PNEUMONIA", "MNIST", "MNIST_COLOR", "MNIST_BBOX"],
        required=True,
        help="Which dataset/config to use"
    )
    p.add_argument("--task", "-t", type=str, choices=["generate", "classify"], default="generate")
    p.add_argument("--color", "-c", type=str, choices=["fg", "bg"], default="fg")
    p.add_argument("--number", "-n", type=int, default=None)
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
    cfg = load_config(config_path, args.experiment_id, args.run_id, task=args.task)
    dirs = build_dirs(cfg)
    logger, writer, wandb_run = init_observers(cfg, dirs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        loaders = DATASET_LOADERS[args.dataset](cfg, number=args.number, task=args.task) # **vars(args)
        if args.dataset.startswith("MNIST") and args.dataset != "MNIST":
            train_loader, val_loader, test_loader = loaders.get(args.color)
        else:
            train_loader, val_loader, test_loader = loaders
        if args.task == "classify":
            model = MODEL_REGISTRY[args.dataset]().to(device)
            epochs, eval_interval = 5, 1
            classify_task(cfg, dirs, model, train_loader, val_loader, test_loader, device, logger, epochs,
                          eval_interval)
        elif args.task == "generate":
            model = Unet(**cfg.model.params).to(device)
            ema = EMA(model, decay=cfg.diffusion.ema_decay)
            generate_task(cfg, dirs, model, ema, train_loader, logger, device, writer, wandb_run)
    except Exception as e:
        log_exception(logger, exception=e)
        send_failure_email(run_id=cfg.run_id, reason=str(e), epoch=0)