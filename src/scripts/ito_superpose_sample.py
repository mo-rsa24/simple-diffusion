"""Example script to load two pretrained models and sample using Ito superposition."""
import torch
from pathlib import Path
import torchvision
from src.models.superposition.ito_operator import and_sampler
from src.models.vanilla.diffusion import p_sample_loop


def load_model(path: str):
    return torch.load(path, map_location="cpu")


def main(model_a_path: str, model_b_path: str, out_dir: str = "samples"):
    model_a = load_model(model_a_path)
    model_b = load_model(model_b_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imgs = and_sampler([model_a, model_b], p_sample_loop,
                       shape=(16, 3, 28, 28))
    for i, img in enumerate(imgs):
        img = ((img + 1) / 2).clamp(0, 1)
        torchvision.utils.save_image(img, Path(out_dir) / f"sample_{i}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a")
    parser.add_argument("--model_b")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    main(args.model_a, args.model_b, args.out)