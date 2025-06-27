# mappings.py (new file or at top of run.py)
from src.dataset.MNIST import get_mnist_loaders
from src.dataset.utils import get_separate_loader
from src.models.classifier.digit_classifier import DigitClassifier
from src.models.classifier.digit_color_bbox_classifier import DigitColorBBoxClassifier
from src.models.classifier.digit_color_classifier import DigitColorClassifier
from src.models.ldm.autoencoder import SimpleConvEncoder, SimpleConvDecoder
from src.models.vanilla.unet import Unet

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
    "ldm": {
        "unet": Unet,
        "encoder": SimpleConvEncoder,
        "decoder": SimpleConvDecoder,
    }
}

DATASET_LOADERS = {
    "MNIST": lambda cfg, **kwargs: get_mnist_loaders(cfg, number=kwargs.get("number")),
    "MNIST_COLOR": lambda cfg, **kwargs: get_separate_loader(number=kwargs.get("number"), batch_size=kwargs.get("batch_size", 8), task=kwargs.get("task", "classify")),
    "MNIST_BBOX": lambda cfg, **kwargs: get_separate_loader(dataset="MNIST_BBOX", number=kwargs.get("number"), batch_size=kwargs.get("batch_size", 8), task=kwargs.get("task", "classify")),
}
