from pathlib import Path

import wandb
import yaml
from tensorboard.plugin_util import experiment_id

from src.config.configs import Config, TrainingConfig, LoggingConfig, ObservabilityConfig, DiffusionConfig, ModelConfig, \
    OptimizerConfig, DatasetConfig, DirsConfig
from src.utils.logger import init_logger


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(
        experiment_id = data["experiment_id"],
        run_id        = data["run_id"],
        training      = TrainingConfig(**data["training"]),
        dirs          = DirsConfig(**data["dirs"]),
        logging       = LoggingConfig(**data["logging"]),
        observability = ObservabilityConfig(**data["observability"]),
        diffusion     = DiffusionConfig(**data["diffusion"]),
        model         = ModelConfig(**data["model"]),
        optimizer     = OptimizerConfig(**data["optimizer"]),
        dataset       = DatasetConfig(**data["dataset"])
    )


def build_dirs(cfg: Config) -> dict:
    """
    Create one subfolder per top‐level dir, then
    create nested subdirs under results_dir['base'].
    """
    paths = {}

    # 1) simple roots
    for key, root in [
        ("ckpt",    cfg.dirs.ckpt_dir),
        ("logs",    cfg.dirs.logs_dir),
        ("tb",      cfg.dirs.tensorboard_dir),
        ("wandb",   cfg.dirs.wandb_dir),
    ]:
        experiment = f"experiment_{cfg.experiment_id}"; run = f"run_{cfg.run_id}"
        p = Path(root) / Path(experiment) / Path(run)
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p

    # 2) results_dir with nesting
    rd = cfg.dirs.results_dir
    base = Path(rd["base"]) / Path(experiment) / Path(run)
    base.mkdir(parents=True, exist_ok=True)
    paths["results_base"] = base

    # create each subfolder under the 'base' path
    for name, rel in rd.items():
        if name == "base":
            continue
        # use the name of the rel-path as the folder name
        sub = base / Path(rel).name
        sub.mkdir(parents=True, exist_ok=True)
        paths[f"results_{name}"] = sub

    return paths

def init_observers(cfg: Config, dirs: dict):
    logger = init_logger(str(dirs["logs"]), log_to_stdout=True)
    writer = None
    if cfg.logging.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(dirs["logs"] / "tb"))
    wandb_tracker = None
    if cfg.logging.use_wandb:
        wandb_tracker = wandb.init(
            project=cfg.experiment_id,
            name=cfg.run_id,
            config=cfg.__dict__
        )
    return logger, writer, wandb_tracker