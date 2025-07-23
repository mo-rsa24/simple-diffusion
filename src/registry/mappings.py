# mappings.py (new file or at top of run.py)
from src.dataset.MNIST import get_mnist_loaders
from src.dataset.utils import get_separate_loader, get_composable_loaders, get_composable_separate_loader
from src.models.classifier.digit_classifier import DigitClassifier
from src.models.classifier.digit_color_bbox_classifier import DigitColorBBoxClassifier
from src.models.classifier.digit_color_classifier import DigitColorClassifier
from src.models.composable_architectures import CompositionalUNet, CascadedDiffusion, GuidedUNet, MixtureOfExperts, \
    ClassifierGuidedUNet
from src.models.composable_diffusion import ComposableDiffusionModel
from src.models.edm.EDM import EDMUNet
from src.models.ldm.autoencoder import AutoencoderKL, Decoder, Encoder
from src.models.slot_diffusion.SlotDiffusionUNet import SlotDiffusionUNet
from src.models.vae.disentangled_vae import DisentangledVAE
from src.models.vanilla.composable_unet import ComposableUnet
from src.models.vanilla.unet import Unet
from src.models.vpsde.ColoredMNISTScoreModel import ColoredMNISTScoreModel

COLOR_MAP = {
    0: (255, 0, 0),     # Red
    1: (0, 255, 0),     # Green
    2: (0, 0, 255),     # Blue
    3: (255, 255, 0),   # Yellow
    4: (255, 0, 255),   # Magenta
    5: (0, 255, 255),   # Cyan
    6: (255, 165, 0),   # Orange
    7: (128, 0, 128),   # Purple
    8: (0, 128, 128),   # Teal
    9: (128, 128, 0),   # Olive
}

BBOX_COLOR_MAP = {
    0: "red",
    1: "green",
    2: "blue",
    3: "orange",
    4: "purple",
    5: "brown",
    6: "pink",
    7: "cyan",
    8: "yellow",
    9: "magenta"
}

CLASSIFIER_MODEL_REGISTRY = {
    "MNIST": DigitClassifier,
    "MNIST_COLOR": DigitColorClassifier,
    "MNIST_BBOX": DigitColorBBoxClassifier,
}

GENERATION_MODEL_REGISTRY = {
    "vanilla": {
        "unet": Unet
    },
    "composable_vanilla": {
        "unet": ComposableUnet
    },
    "vae": {
        "vae": AutoencoderKL
    },
    "beta_vae": {
        "vae": DisentangledVAE
    },
    "ldm": {
        "unet": Unet,
    },
    "vpsde": {
        "scoreModel": ColoredMNISTScoreModel,
    },
    "composable_ldm": {
        "unet": ComposableUnet,
    },
    "slot": {
        "unet": SlotDiffusionUNet,
    },
    "edm":{
        "unet": EDMUNet
    },
    "composable": {
        "unet": ComposableDiffusionModel
    },
    "comp_unet": {
        "unet": CompositionalUNet
    },
    "cascaded": {
        "unet": CascadedDiffusion
    },
    "guided": {
        "unet": GuidedUNet
    },
    "moe": {
        "unet": MixtureOfExperts
    },
    "classifier_guided": {
        "unet": ClassifierGuidedUNet
    }
}

DATASET_LOADERS = {
    "MNIST": lambda cfg, **kwargs: get_mnist_loaders(cfg, number=kwargs.get("number")),
    "MNIST_COLOR": lambda cfg, **kwargs: get_separate_loader(cfg, number=kwargs.get("number"), task=kwargs.get("task", "classify"), digit_color_label=kwargs.get("digit_color_label"), bbox_color_label=kwargs.get("bbox_color_label")),
    "MNIST_BBOX": lambda cfg, **kwargs: get_separate_loader(cfg, dataset="MNIST_BBOX", number=kwargs.get("number"), task=kwargs.get("task", "classify")),
    "MNIST_COMPOSABLE": lambda cfg, **kwargs: get_composable_separate_loader(cfg, number=kwargs.get("number"), digit_color_label=kwargs.get("digit_color_label"), bbox_color_label=kwargs.get("bbox_color_label")),
}
