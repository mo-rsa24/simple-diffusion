from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TrainingConfig:
    epochs: int
    log_every_step: int
    log_every_epoch: int
    save_every_step: int = 0
    save_every_epoch: int = 1
    resume_from: bool = False

@dataclass
class DirsConfig:
    cluster_base: str
    local_base: str
    ckpt_dir: str
    logs_dir: str
    results_dir:   Dict[str, str]
    tensorboard_dir: str
    wandb_dir: str

@dataclass
class LoggingConfig:
    use_tensorboard: bool
    use_wandb: bool
    log_level: str = "INFO"

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
    ema_decay: float
    beta_end: float

@dataclass
class ModelConfig:
    type: str
    vpsde: Dict[str, Any]
    ldm: Dict[str, Any]
    edm: Dict[str, Any]
    slot: Dict[str, Any]
    vanilla: Dict[str, Any]

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
    augmentation: str
    normalization: str
    resize_strategy: str
    hist_eq: bool

@dataclass
class SamplingConfig:
    batch_size: int
    channels: int
    blit: bool
    repeat_delay: float

@dataclass
class Config:
    experiment_id: str
    run_id: str
    task: str
    seed: int
    training: TrainingConfig
    logging: LoggingConfig
    dirs: DirsConfig
    observability: ObservabilityConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    dataset: DatasetConfig
    sampling: SamplingConfig
