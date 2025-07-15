from pathlib import Path
import torch
import wandb
import yaml
from src.config.configs import Config, TrainingConfig, LoggingConfig, ObservabilityConfig, DiffusionConfig, ModelConfig, \
    OptimizerConfig, DatasetConfig, DirsConfig, SamplingConfig, SanityCheckConfig
from src.utils.env import is_cluster
from src.utils.logger import init_logger


def load_config(path: str, experiment_id: str, run_id:str, task: str = "generate") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(
        experiment_id = experiment_id,
        run_id        = run_id,
        task          = task,
        seed          = data["seed"],
        training      = TrainingConfig(**data["training"]),
        dirs          = DirsConfig(**data["dirs"]),
        logging       = LoggingConfig(**data["logging"]),
        observability = ObservabilityConfig(**data["observability"]),
        diffusion     = DiffusionConfig(**data["diffusion"]),
        model         = ModelConfig(**data["model"]),
        optimizer     = OptimizerConfig(**data["optimizer"]),
        dataset       = DatasetConfig(**data["dataset"]),
        sampling      = SamplingConfig(**data["sampling"]),
        sanity_checks = SanityCheckConfig(**data["sanity_checks"])
    )


# def build_dirs(cfg: Config, base_dir: Optional[str] = None) -> dict:
#     root_base = Path(cfg.dirs.cluster_base if is_cluster() else cfg.dirs.local_base)
#     experiment = f"experiment_{cfg.experiment_id}"
#     run = f"run_{cfg.run_id}"
#
#     paths = {}
#
#     # 1) Root-level folders
#     for key, attr in [
#         ("ckpt", cfg.dirs.ckpt_dir),
#         ("logs", cfg.dirs.logs_dir),
#         ("tb", cfg.dirs.tensorboard_dir),
#         ("wandb", cfg.dirs.wandb_dir),
#     ]:
#         abs_path = root_base / Path(attr) / cfg.task / experiment / run
#         abs_path.mkdir(parents=True, exist_ok=True)
#         paths[key] = abs_path
#
#     # 2) Results subfolders
#     results_cfg = cfg.dirs.results_dir
#     results_base = root_base / Path(results_cfg["base"]) / experiment / run
#     results_base.mkdir(parents=True, exist_ok=True)
#     paths["results_base"] = results_base
#
#     for name, rel_path in results_cfg.items():
#         if name == "base":
#             continue
#         full_path = results_base / Path(rel_path).name
#         full_path.mkdir(parents=True, exist_ok=True)
#         paths[f"results_{name}"] = full_path
#
#     return paths

from typing import Optional

def build_dirs(cfg: Config, base_dir: Optional[str] = None, gen_modeL: str = "vanilla") -> dict:
    if base_dir is not None:
        base = Path(base_dir)
        ckpt_dir = base / "checkpoints"
        logs_dir = base / "logs"
        tb_dir = logs_dir / "tb"
        wandb_dir = logs_dir / "wandb"
        for d in [ckpt_dir, logs_dir, tb_dir, wandb_dir]:
            d.mkdir(parents=True, exist_ok=True)

        paths = {
            "ckpt": ckpt_dir,
            "logs": logs_dir,
            "tb": tb_dir,
            "wandb": wandb_dir,
        }

        results_base = base / "results"
        results_base.mkdir(parents=True, exist_ok=True)
        paths["results_base"] = results_base

        for name, rel_path in cfg.dirs.results_dir.items():
            if name == "base":
                continue
            dest = results_base / Path(rel_path).name
            dest.mkdir(parents=True, exist_ok=True)
            paths[f"results_{name}"] = dest

        return paths

    root_base = Path(cfg.dirs.cluster_base if is_cluster() else cfg.dirs.local_base)
    experiment = f"{cfg.experiment_id}"
    run = f"run_{cfg.run_id}"
    task = cfg.task

    paths = {}

    # 1) Root-level folders
    for key, attr in [
        ("ckpt", cfg.dirs.ckpt_dir),
        ("logs", cfg.dirs.logs_dir),
        ("tb", cfg.dirs.tensorboard_dir),
        ("wandb", cfg.dirs.wandb_dir),
    ]:
        abs_path = root_base / "simple-diffusion" / experiment / run  / task / gen_modeL
        abs_path.mkdir(parents=True, exist_ok=True)
        paths[key] = abs_path

    # 2) Results subfolders
    results_cfg = cfg.dirs.results_dir
    results_base = root_base / 'simple-diffusion' / experiment / run  / task / gen_modeL / Path(results_cfg["base"])
    results_base.mkdir(parents=True, exist_ok=True)
    paths["results_base"] = results_base

    for name, rel_path in results_cfg.items():
        if name == "base":
            continue
        full_path = results_base / Path(rel_path).name
        full_path.mkdir(parents=True, exist_ok=True)
        paths[f"results_{name}"] = full_path

    return paths


def init_observers(cfg: Config, dirs: dict, task: str = "TB"):
    logger = init_logger(str(dirs["logs"]), log_to_stdout=True, log_level=cfg.logging.log_level)
    writer = None
    if cfg.logging.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(dirs["logs"] / "tb"))
        except Exception as e:
            logger.warning(f"TensorBoard unavailable: {e}")
    wandb_tracker = None
    if cfg.logging.use_wandb:
        try:
            wandb_tracker = wandb.init(
                project=f"chest-xray-{task}",
                name=f"experiment_{cfg.experiment_id}_run_{cfg.run_id}",
                config=cfg.__dict__
            )
            wandb.run.notes = f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        except Exception as e:
            logger.warning(f"WandB unavailable: {e}")
    return logger, writer, wandb_tracker