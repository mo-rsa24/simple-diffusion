import argparse
import os
from pathlib import Path
import torch
from src.models.vanilla.ema import EMA
from src.monitoring.alert_notifier import send_failure_email
from src.registry.mappings import DATASET_LOADERS, CLASSIFIER_MODEL_REGISTRY, GENERATION_MODEL_REGISTRY
from src.task import classify_task
from src.train.logging.training_logger_utils import log_exception
from src.utils.setup import load_config, build_dirs, init_observers

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
            "ldm",
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

    p.add_argument(
        "--dry-run", action="store_true",
        help="Print merged config and exit"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_path = f"src/config/{args.dataset.lower()}.yml"
    cfg = load_config(config_path, args.experiment_id, args.run_id, task=args.task)
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
        print(cfg)
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                model_params = dict(cfg.model.vanilla)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["vanilla"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.simple_diffusion.train import train as pixel_train
                pixel_train(cfg, dirs, model, ema, train_loader, logger, device, writer, wandb_run)
            elif args.gen_model == "ldm":
                model_params = dict(cfg.model.ldm)  # copy so we don't modify original config
                latent_dim = model_params.pop("latent_dim")# fallback if missing
                model_params["channels"] = latent_dim
                encoder = GENERATION_MODEL_REGISTRY["ldm"]["encoder"](
                    in_channels=cfg.dataset.channels, latent_dim=latent_dim).to(device)
                decoder = GENERATION_MODEL_REGISTRY["ldm"]["decoder"](
                    latent_dim=latent_dim, out_channels=cfg.dataset.channels).to(device)
                model = GENERATION_MODEL_REGISTRY["ldm"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.ldm.ldm_train import train as ldm_train
                ldm_train(cfg, dirs, model, ema, encoder, decoder, train_loader, logger, device, writer, wandb_run)
            elif args.gen_model == "slot":
                model_params = dict(cfg.model.slot)  # copy so we don't modify original config
                model = GENERATION_MODEL_REGISTRY["slot"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.slot.slot_train import train as slot_train
                slot_train(cfg, dirs, model, ema, train_loader, logger, device, writer=None, wandb_run=None)
            elif args.gen_model == "vpsde":
                model_params = dict(cfg.model.vpsde)
                model = GENERATION_MODEL_REGISTRY["vpsde"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.vpsde.vpsde_train import train as vpsde_train
                vpsde_train(cfg, dirs, model, ema, train_loader,
                          logger, device, writer, wandb_run)
            elif args.gen_model == "edm":
                model_params = dict(cfg.model.edm)
                model = GENERATION_MODEL_REGISTRY["edm"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.edm.edm_train import train as edm_train
                edm_train(cfg, dirs, model, ema, train_loader,
                          logger, device, writer, wandb_run)
            elif args.gen_model == "composable":
                model_params = dict(cfg.model.composable)
                model = GENERATION_MODEL_REGISTRY["composable"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable.composable_train import train as composable_train

                composable_train(cfg, dirs, model, ema, train_loader,
                                 logger, device, writer, wandb_run)
            elif args.gen_model == "comp_unet":
                model_params = dict(cfg.model.composable)
                model = GENERATION_MODEL_REGISTRY["comp_unet"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable_architectures import comp_unet_train

                comp_unet_train(cfg, dirs, model, ema, train_loader,
                                logger, device, writer, wandb_run)
            elif args.gen_model == "cascaded":
                model_params = dict(cfg.model.composable)
                model = GENERATION_MODEL_REGISTRY["cascaded"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable_architectures import cascaded_train

                cascaded_train(cfg, dirs, model, ema, train_loader,
                               logger, device, writer, wandb_run)
            elif args.gen_model == "guided":
                model_params = dict(cfg.model.composable)
                model = GENERATION_MODEL_REGISTRY["guided"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable_architectures import guided_train

                guided_train(cfg, dirs, model, ema, train_loader,
                             logger, device, writer, wandb_run)
            elif args.gen_model == "moe":
                model_params = dict(cfg.model.composable)
                model = GENERATION_MODEL_REGISTRY["moe"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable_architectures import moe_train
                moe_train(cfg, dirs, model, ema, train_loader,
                          logger, device, writer, wandb_run)
            elif args.gen_model == "classifier_guided":
                model_params = dict(cfg.model.classifier_guided)
                model = GENERATION_MODEL_REGISTRY["classifier_guided"]["unet"](**model_params).to(device)
                ema = EMA(model, decay=cfg.diffusion.ema_decay)
                from src.train.generate.composable_architectures.classifier_guided_train import train as classifier_guided_train

                classifier_guided_train(cfg, dirs, model, ema, train_loader,
                                        logger, device, writer, wandb_run)
    except Exception as e:
        log_exception(logger, exception=e)
        send_failure_email(run_id=cfg.run_id, reason=str(e), epoch=0)