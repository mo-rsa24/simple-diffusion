from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TrainingConfig:
    epochs: int
    log_every_step: int
    log_every_epoch: int

@dataclass
class DirsConfig:
    ckpt_dir: str
    logs_dir: str
    results_dir:   Dict[str, str]
    tensorboard_dir: str
    wandb_dir: str

@dataclass
class LoggingConfig:
    use_tensorboard: bool
    use_wandb: bool

@dataclass
class ObservabilityConfig:
    feature_maps: bool
    batch_grid: bool
    sample_table: bool
    diffusion_process: bool

@dataclass
class DiffusionConfig:
    timesteps: int
    loss_type: str
    beta_start: float
    beta_end: float

@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]

@dataclass
class OptimizerConfig:
    type: str
    params: Dict[str, Any]

@dataclass
class DatasetConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: int
    pin_memory: bool
    channels: int

@dataclass
class Config:
    experiment_id: str
    run_id: str
    training: TrainingConfig
    logging: LoggingConfig
    dirs: DirsConfig
    observability: ObservabilityConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    dataset: DatasetConfig
