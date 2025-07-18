import collections
import collections.abc
import torch
import wandb
from src.config.configs import Config, TrainingConfig, LoggingConfig, ObservabilityConfig, DiffusionConfig, ModelConfig, \
    OptimizerConfig, DatasetConfig, DirsConfig, SamplingConfig, SanityCheckConfig
from src.utils.env import is_cluster
from src.utils.logger import init_logger
from box import Box
import yaml
from pathlib import Path
from typing import Dict, Optional


def deep_merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """Recursively merges dictionary d2 into a copy of d1."""
    d1 = d1.copy()
    for k, v in d2.items():
        if k in d1 and isinstance(d1.get(k), dict) and isinstance(v, collections.abc.Mapping):
            d1[k] = deep_merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1

def load_config(
    config_path: str,
    experiment_id: str,
    run_id: int,
    task: str,
    gen_model: str,
    profile_name: str = "default" # Default to the 'default' profile
) -> Box:
    """
    Loads and merges configurations from the user-provided YAML structure.

    Merge Order:
    1. Base config (all top-level keys except 'model' and 'profiles').
    2. Model-specific overrides (`model.<gen_model>`).
    3. Profile-specific overrides (`profiles.<profile_name>`).
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # PyYAML automatically handles anchors (&) and aliases (*)
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f) or {}

    # 1. Start with the base configuration (everything except model and profiles)
    base_cfg = {k: v for k, v in full_config.items() if k not in ['model', 'profiles']}

    # 2. Merge in the model-specific overrides
    model_specific_cfg = full_config.get('model', {}).get(gen_model)
    if model_specific_cfg:
        print(f"Applying overrides for model: '{gen_model}'")
        merged_cfg = deep_merge_dicts(base_cfg, model_specific_cfg)
    else:
        print(f"Warning: Model '{gen_model}' not found in config. Using base settings.")
        merged_cfg = base_cfg

    # 3. ✨ Apply profile-specific overrides intelligently
    profile_cfg = full_config.get('profiles', {}).get(profile_name)
    if profile_cfg:
        print(f"Applying overrides for profile: '{profile_name}'")
        for key, value in profile_cfg.items():
            # Case 1: Override a top-level block like 'dataset' or 'training'.
            if key in merged_cfg and isinstance(merged_cfg.get(key), dict):
                merged_cfg[key] = deep_merge_dicts(merged_cfg[key], value)
            # Case 2: Override a key within the top-level 'sanity' block.
            elif 'sanity' in merged_cfg and key in merged_cfg['sanity']:
                merged_cfg['sanity'][key] = value
            # Case 3: A new key, like 'description'. Add it to the root.
            else:
                merged_cfg[key] = value
    # --- Preserve original model block and add runtime args ---
    merged_cfg['model'] = full_config.get('model', {}) # Keep original model block

    # Add runtime arguments for easy access
    merged_cfg['experiment_id'] = experiment_id
    merged_cfg['run_id'] = run_id
    merged_cfg['task'] = task
    merged_cfg['gen_model'] = gen_model
    merged_cfg['profile'] = profile_name

    return Box(merged_cfg, default_box=True)

def load_config_(path: str, experiment_id: str, run_id:str, task: str = "generate") -> Config:
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
def save_config(cfg: Box, save_dir: Path, filename: str = "config.yml"):
    """
    Saves the final configuration Box object to a YAML file.

    Args:
        cfg (Box): The final, merged configuration object.
        save_dir (Path): The directory where the config should be saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    # Convert the Box object to a standard dictionary for clean YAML output
    config_dict = cfg.to_dict()

    print(f"Saving final configuration to: {save_path}")
    with open(save_path, 'w') as f:
        # Use sort_keys=False to maintain the original order
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

def build_dirs(cfg: Box, base_dir: Optional[str] = None, gen_modeL: str = "vanilla") -> dict:
    if base_dir is not None:
        base = Path(base_dir)
        ckpt_dir = base / "checkpoints"
        logs_dir = base / "logs"
        tb_dir = logs_dir / "tb"
        wandb_dir = logs_dir / "wandb"
        hyperparameters_dir = logs_dir / "hyperparameters"
        for d in [ckpt_dir, logs_dir, tb_dir, wandb_dir, hyperparameters_dir]:
            d.mkdir(parents=True, exist_ok=True)

        paths = {
            "ckpt": ckpt_dir,
            "logs": logs_dir,
            "tb": tb_dir,
            "wandb": wandb_dir,
            "hyperparameters": hyperparameters_dir,
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
        ("hyperparameters", cfg.dirs.hyperparameters_dir),
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