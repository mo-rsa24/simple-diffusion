import argparse
import os
from pathlib import Path
import torch
from src.models.composable_diffusion.ComposableExpertUnet import ComposableExpertUnet
from src.monitoring.alert_notifier import send_failure_email
from src.registry.mappings import DATASET_LOADERS, GENERATION_MODEL_REGISTRY
from torch.optim import Adam
from src.train.logging.training_logger_utils import log_exception, generate_and_save_grid
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.sampling import ito_sampler, composable_unet_sampler
from src.utils.setup import load_config, build_dirs, init_observers, save_config


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
            "vanilla",
            "composable_vanilla",
            "ldm",
            "composable_ldm",
            "vae",
            "slot",
            "edm",
            "vpsde",
            "composable",
            "comp_unet",
            "cascaded",
            "guided",
            "moe",
            "classifier_guided",
        ],
        default="vanilla",
        help="Which diffusion model architecture to use.",
    )
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoints")
    p.add_argument("--log-dir", type=str, default=None, help="Directory for training logs")
    p.add_argument("--array_index", type=int, default=None, help="SLURM array task index")
    p.add_argument("--log-level", type=str, default=None, help="Logging level")
    # Inherited arguments from run.py
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    p.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Weights for Itô sampler. Must sum to 1.')
    p.add_argument('--compose_model', type=str, default='ito', choices=['ito', 'composable_unet'],
                        help='Composition model to use')
    p.add_argument('--compose_method', type=str, default='additive', choices=['additive', 'gating'],
                        help='Composition method for ComposableUNet')
    p.add_argument('--profile', type=str, default='default',
                        help="The experiment profile to use (e.g., 'sanity', 'debug').")

    p.add_argument(
        "--dry-run", action="store_true",
        help="Print merged config and exit"
    )
    return p.parse_args()

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
    scheduler = None  # add if needed
    dirs = build_dirs(cfg, base_dir=base_dir, gen_modeL=args.gen_model)
    if args.checkpoint_dir:
        dirs["ckpt"] = Path(args.checkpoint_dir)
    if args.log_dir:
        dirs["logs"] = Path(args.log_dir)
    logger, writer, wandb_run = init_observers(cfg, dirs)
    if args.array_index is not None:
        logger.info(f"SLURM array index: {args.array_index}")
    if args.dry_run:
        print(cfg)
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
            **extra
        )

        if args.dataset.startswith("MNIST") and args.dataset != "MNIST":
            train_loader, val_loader, test_loader = loaders.get(args.color)
        else:
            train_loader, val_loader, test_loader = loaders
        logger.info("Loading models for superposition...")
        models = []
        for i in range(2):
            if args.gen_model == "vanilla":
                model_params = dict(cfg.model.vanilla.architecture)  # copy so we don't modify original config
                models.append(GENERATION_MODEL_REGISTRY["vanilla"]["unet"](**model_params).to(device))
            elif args.gen_model == "composable_vanilla":
                model_params = dict(
                    cfg.model.composable_vanilla.architecture)  # copy so we don't modify original config
                models.append(GENERATION_MODEL_REGISTRY["composable_vanilla"]["unet"](**model_params).to(device))

        digit_5_model_optimizer, digit_2_model_optimizer = None, None
        if cfg.optimizer.type.lower() == "adam":
            digit_5_model_optimizer = Adam(models[0].parameters(), **cfg.optimizer.params)
            digit_2_model_optimizer = Adam(models[1].parameters(), **cfg.optimizer.params)

        try:
            digit_5_checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=dirs.get("ckpt",Path(cfg.dirs.ckpt_dir)),  logger=logger)
            digit_5_model, digit_5_model_optimizer, scheduler, last_epoch, global_step = digit_5_checkpoint_manager.load_latest(models[0], digit_5_model_optimizer, scheduler, map_location=device)
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

        try:
            digit_2_path = Path(dirs.get("ckpt",Path(cfg.dirs.ckpt_dir)).__str__().replace('color_mnist_digit_5', 'color_mnist_digit_2'))
            digit_2_checkpoint_manager = CheckpointManager(run_id=cfg.run_id, checkpoint_dir=digit_2_path, logger=logger)
            digit_2_model, digit_2_model_optimizer, scheduler, last_epoch, global_step = digit_2_checkpoint_manager.load_latest(models[1], digit_2_model_optimizer, scheduler, map_location=device)
        except Exception as e:
            logger.warning(f"Could not resume training: {e}")

        """
        # Check To See Checkpoints were saved and loaded appropriately 
        
        from src.utils.sampling import ddpm_sampler
        from src.utils.visualization import visualize_images
        digit_5 = ddpm_sampler(models[0], cfg, device)
        digit_2 = ddpm_sampler(models[1], cfg, device)
        visualize_images(digit_5)
        visualize_images(digit_2)
        """
        # Load multiple checkpoints
        logger.info(f"Successfully loaded models.")
        # === Prepare for Composition ===
        if args.compose_model == 'composable_unet':
            if len(models) != 2:
                raise ValueError("ComposableUNet requires exactly two models.")
            logger.info(f"Creating ComposableUNet with fusion type: {args.compose_method}")
            composition_model = ComposableExpertUnet(
                unet_a=models[0],
                unet_b=models[1],
                fusion_type=args.compose_method
            ).to(device)
        else:
            composition_model = None  # Not needed for ito_sampler

        # === Run Sampling ===
        logger.info(f"Starting sampling with compose_model='{args.compose_model}'")

        # Define common sampling parameters
        sampling_params = {
            "shape": (cfg.dataset.sampling.batch_size, cfg.dataset.channels, cfg.dataset.image_size,
                      cfg.dataset.image_size),
            "timesteps": cfg.diffusion.timesteps,
            "beta_min": cfg.diffusion.beta_start,
            "beta_max": cfg.diffusion.beta_end,
            "device": device
        }

        if args.compose_model == 'ito':
            weights = [0.4, 0.6]
            logger.info(f"Using ito_sampler with weights: {weights}")
            composed_samples = ito_sampler(
                models,
                torch.Size((4,3, 32,32)),
                cfg.diffusion.timesteps,
                cfg.diffusion.beta_start,
                cfg.diffusion.beta_end,
                weights=weights,
                device=device,
            )
        elif args.compose_model == 'composable_unet':
            logger.info("Using composable_unet_sampler")
            # composed_samples = composable_unet_sampler(
            #     model=composition_model,
            #     **sampling_params
            # )
            composed_samples = composable_unet_sampler(
                composition_model,
                torch.Size((4, 3, 32, 32)),
                cfg.diffusion.timesteps,
                cfg.diffusion.beta_start,
                cfg.diffusion.beta_end,
                device=device,
            )
        else:
            raise ValueError(f"Unknown compose_model: {args.compose_model}")

        logger.info("Sampling complete.")

        # === Visualization ===
        logger.info("Generating and saving visualization grid...")
        generate_and_save_grid(
            individual_models=models,
            composed_samples=composed_samples,
            compose_model_name=args.compose_model,
            sampling_params=sampling_params,
            save_path=dirs.get("results_samples")
        )
        logger.info(f"Visualization grid saved in {logger.get_image_dir()}")

    except Exception as e:
        log_exception(logger, exception=e)
        send_failure_email(run_id=cfg.run_id, reason=str(e), epoch=0)