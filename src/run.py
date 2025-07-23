import argparse
import os
from pathlib import Path
import torch
from src.models.ldm.autoencoder import AutoencoderKL
from src.models.vanilla.ema import EMA
from src.models.vpsde.ColoredMNISTScoreModel import VPSDE
from src.monitoring.alert_notifier import send_failure_email
from src.registry.mappings import DATASET_LOADERS, CLASSIFIER_MODEL_REGISTRY, GENERATION_MODEL_REGISTRY
from src.task import classify_task
from src.train.generate.vae.train_beta_vae import train_beta_vae
from src.train.logging.training_logger_utils import log_exception
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.setup import load_config, build_dirs, init_observers, save_config
import pprint
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.prompt import Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", "-e", type=str, required=True)
    p.add_argument("--run_id", "-r", type=str, required=True)
    p.add_argument(
        "--dataset", "-d",
        choices=["TB", "PNEUMONIA", "MNIST", "MNIST_COLOR", "MNIST_BBOX", "MNIST_COMPOSABLE"],
        required=True,
        help="Which dataset/config to use"
    )
    p.add_argument("--task", "-t", type=str, choices=["generate", "classify"], default="generate")
    p.add_argument("--color", "-c", type=str, choices=["fg", "bg"], default="fg")
    p.add_argument("--number", "-n", type=int, default=None)
    p.add_argument(
        "--gen_model",
        "-g",
        type=str,
        choices=[
            "vanilla", # 👈
            "composable_vanilla", # 👈
            "ldm", # 👈
            "vpsde", # 👈
            "composable_ldm", # 👈
            "vae", # 👈
            "beta_vae", # 👈
            "composable", # 👈
            "comp_unet", # 👈
        ],
        default="vanilla",
        help="Which diffusion model architecture to use.",
    )
    p.add_argument('--loss_mask', type=str, default=None, choices=['digit', 'bbox'],
                   help='Apply loss only to a specific part of the image to train an "expert" model.')
    p.add_argument('--digit_color_label', type=int, default=None,
                   help='Apply a specific color to all the digits. Usually used when wanting to train experts')
    p.add_argument('--bbox_color_label', type=int, default=None,
                   help='Apply a specific color to all the bounding box. Usually used when wanting to train experts')
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoints")
    p.add_argument("--log-dir", type=str, default=None, help="Directory for training logs")
    p.add_argument("--array_index", type=int, default=None, help="SLURM array task index")
    p.add_argument("--log-level", type=str, default=None, help="Logging level")
    p.add_argument('--profile', type=str, default='default',
                        help="The experiment profile to use (e.g., 'sanity', 'debug').")

    p.add_argument(
        "--dry-run", action="store_true",
        help="Print merged config and exit"
    )
    return p.parse_args()


def display_dry_run_summary(args, cfg, base_dir):
    """
    Displays a rich summary of the configuration for a dry run.
    """
    if not RICH_AVAILABLE:
        print("Rich library not installed. Run 'pip install rich' for a better dry-run experience.")
        print("\n--- Arguments ---")
        import json
        print(json.dumps(vars(args), indent=2))
        print("\n--- Configuration ---")
        print(cfg)
        return

    console = Console()
    console.print(Panel(Text("🚀 Experiment Dry Run Summary", justify="center", style="bold cyan"),
                        border_style="cyan"))

    # --- Arguments and Paths Tables (Corrected) ---
    arg_table = Table(title="CLI Arguments", show_header=True, header_style="bold magenta", expand=True)
    arg_table.add_column("Argument", style="dim", no_wrap=True, ratio=1)
    arg_table.add_column("Value", style="bold", overflow="fold", ratio=3)
    arg_dict = vars(args)
    for arg, value in arg_dict.items():
        if value is not None and value is not False:
            arg_table.add_row(f"--{arg}", str(value))
    console.print(arg_table)

    # --- Derived Paths ---
    path_table = Table(title="Derived Paths", show_header=True, header_style="bold green", expand=True)
    path_table.add_column("Path Type", style="dim", no_wrap=True, ratio=1)
    path_table.add_column("Full Path", overflow="fold", ratio=3)

    dirs = build_dirs(cfg, base_dir=base_dir, gen_modeL=args.gen_model)
    for name, path in dirs.items():
        path_table.add_row(name.capitalize(), str(path))

    console.print(path_table)

    # --- Full Configuration Panel ---
    config_str = ""
    # Use OmegaConf's to_yaml() if available, as it's likely the config object type.
    if "omegaconf" in str(type(cfg)):
        from omegaconf import OmegaConf
        config_str = OmegaConf.to_yaml(cfg)
    else:
        # Fallback to pretty printing if not an OmegaConf object
        import pprint
        config_str = pprint.pformat(cfg)

    syntax = Syntax(config_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="[bold yellow]Full Merged Configuration[/bold yellow]", border_style="yellow"))
    console.print("[bold green]✅ Config verified. No job was submitted.[/bold green]")

if __name__ == "__main__":
    args = parse_args()
    config_path = f"src/config/{args.dataset.lower()}.yml"
    # cfg = load_config(config_path, args.experiment_id, args.run_id, task=args.task)
    cfg = load_config(
        config_path=config_path,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        task=args.task,
        gen_model=args.gen_model,
        profile_name=args.profile
    )
    if args.log_level:
        cfg.logging.log_level = args.log_level
    cfg.logging.use_wandb = args.use_wandb or cfg.logging.use_wandb
    cfg.logging.use_tensorboard = args.use_tensorboard or cfg.logging.use_tensorboard
    cfg.training.resume_from = args.resume or cfg.training.resume_from
    from src.utils.env import set_global_seeds

    set_global_seeds(cfg.seed)
    base_dir = None
    if args.checkpoint_dir:
        base_dir = Path(args.checkpoint_dir).resolve().parent
    elif args.log_dir:
        base_dir = Path(args.log_dir).resolve().parent
    elif args.resume and os.environ.get("BASE_DIR"):
        base_dir = Path(os.environ["BASE_DIR"])  # default from launcher

    dirs = build_dirs(cfg, base_dir=base_dir, gen_modeL=args.gen_model)
    if args.checkpoint_dir:
        dirs["ckpt"] = Path(args.checkpoint_dir)
    if args.log_dir:
        dirs["logs"] = Path(args.log_dir)
    logger, writer, wandb_run = init_observers(cfg, dirs)
    if args.array_index is not None:
        logger.info(f"SLURM array index: {args.array_index}")
    if args.dry_run:
        display_dry_run_summary(args, cfg, base_dir)
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_config(cfg, dirs.get('hyperparameters'), os.path.basename(config_path))
    try:
        extra = {}
        if args.dataset == "MNIST_COMPOSABLE":
            extra["variant"] = "foreground" if args.color == "fg" else "background"
        loaders = DATASET_LOADERS[args.dataset](
            cfg,
            number=args.number,
            task=args.task,
            digit_color_label=args.digit_color_label,
            bbox_color_label=args.bbox_color_label,
            **extra
        )

        if args.dataset.startswith("MNIST") and args.dataset != "MNIST":
            train_loader, val_loader, test_loader = loaders.get(args.color)
        else:
            train_loader, val_loader, test_loader = loaders
        if args.task == "classify":
            model = CLASSIFIER_MODEL_REGISTRY[args.dataset]().to(device)
            eval_interval = 1
            classify_task(
                cfg,
                dirs,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                logger,
                eval_interval,
            )
        elif args.task == "generate":
            if args.gen_model == "vanilla":
                model_params = dict(cfg.model.vanilla.architecture)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["vanilla"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                if args.loss_mask:
                    from src.train.generate.simple_diffusion.train_disentangled import train as train_disentangled
                    train_disentangled(cfg, dirs, model, ema, train_loader, logger, device, args.loss_mask, writer, wandb_run)
                else:
                    from src.train.generate.simple_diffusion.train import train as pixel_train
                    pixel_train(cfg, dirs, model, train_loader, val_loader,  logger, device, writer, wandb_run)
            elif args.gen_model == "composable_vanilla":
                model_params = dict(cfg.model.composable_vanilla.architecture)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["composable_vanilla"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.simple_diffusion.composable_train import train as composable_train
                composable_train(cfg, dirs, model, ema, train_loader, logger, device, writer, wandb_run)
            elif args.gen_model == "vpsde":
                model_params = dict(cfg.model.vpsde.architecture)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["vpsde"]["scoreModel"](**model_params).to(device)
                from src.train.generate.vpsde.vpsde_train import train as vpsde_train
                sde = VPSDE(num_timesteps=cfg.diffusion.timesteps, device=device)
                vpsde_train(cfg, dirs, model, sde,  train_loader, val_loader, logger, device, writer=writer, wandb_run=wandb_run)
            elif args.gen_model == "vae":
                model_params = dict(cfg.model.vae.architecture)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["vae"]["vae"](**model_params).to(device)
                from src.train.generate.vae.vae_train import train as vae_train
            elif args.gen_model == "beta_vae":
                model_params = dict(cfg.model.beta_vae.architecture)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["beta_vae"]["vae"](**model_params).to(device)
                train_beta_vae(cfg, dirs, model, train_loader, val_loader, logger, device, writer=writer, wandb_run=wandb_run)
            elif args.gen_model == "ldm":
                model_params = dict(cfg.model.ldm.architecture)  # copy so we don't modify original config

                vae_params = dict(cfg.model.vae.architecture)
                vae: AutoencoderKL = GENERATION_MODEL_REGISTRY["vae"]["vae"](**vae_params).to(device)
                vae_path = Path(Path(dirs.get('ckpt', cfg.dirs.ckpt_dir)).__str__().replace('ldm', 'vae'))
                try:
                    vae_checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=vae_path,
                                                               logger=logger)
                    vae_optimizer = torch.optim.Adam(vae.parameters(), **cfg.optimizer.params)
                    vae, vae_optimizer, scheduler, last_epoch, global_step = vae_checkpoint_manager.load_latest(
                        vae, vae_optimizer, map_location=device)
                except Exception as e:
                    logger.warning(f"Could not resume training: {e}")

                encoder = vae.encoder
                decoder = vae.decoder
                encoder.eval()
                decoder.eval()

                latent_channels = vae.encoder.z_channels # print model_params['channels'] here:
                model_params['channels'] = latent_channels
                model = GENERATION_MODEL_REGISTRY["ldm"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.ldm.ldm_train import train as ldm_train
                ldm_train(cfg, dirs, model, ema, vae, train_loader, val_loader, logger, device, writer, wandb_run)
            elif args.gen_model == "composable_ldm":
                model_params = dict(cfg.model.composable_ldm.architecture)  # copy so we don't modify original config

                vae_params = dict(cfg.model.vae.architecture)
                vae: AutoencoderKL = GENERATION_MODEL_REGISTRY["vae"]["vae"](**vae_params).to(device)
                vae_path = Path(Path(dirs.get('ckpt', cfg.dirs.ckpt_dir)).__str__().replace('composable_ldm', 'vae'))
                try:
                    vae_checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=vae_path,
                                                               logger=logger)
                    vae_optimizer = torch.optim.Adam(vae.parameters(), **cfg.optimizer.params)
                    vae, vae_optimizer, scheduler, last_epoch, global_step = vae_checkpoint_manager.load_latest(
                        vae, vae_optimizer, map_location=device)
                except Exception as e:
                    logger.warning(f"Could not resume training: {e}")

                encoder = vae.encoder
                decoder = vae.decoder
                encoder.eval()
                decoder.eval()

                latent_channels = vae.encoder.z_channels # print model_params['channels'] here:
                model_params['channels'] = latent_channels
                model = GENERATION_MODEL_REGISTRY["composable_ldm"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.ldm.composable_ldm_train import train as composable_ldm_train
                composable_ldm_train(cfg, dirs, model, ema, vae, train_loader, val_loader, logger, device, writer, wandb_run)
            elif args.gen_model == "composable":
                model_params = dict(cfg.model.composable.architecture)
                model = GENERATION_MODEL_REGISTRY["composable"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable.composable_train import train as composable_train

                composable_train(cfg, dirs, model, train_loader,val_loader,
                                 logger, device, writer, wandb_run)
            elif args.gen_model == "comp_unet":
                model_params = dict(cfg.model.comp_unet.architecture)
                model = GENERATION_MODEL_REGISTRY["comp_unet"]["unet"](**model_params).to(device)
                from src.train.generate.composable_architectures import comp_unet_train

                comp_unet_train(cfg, dirs, model, train_loader, val_loader,
                                logger, device, writer, wandb_run)
    except Exception as e:
        log_exception(logger, exception=e)
        send_failure_email(run_id=cfg.run_id, reason=str(e), epoch=0)