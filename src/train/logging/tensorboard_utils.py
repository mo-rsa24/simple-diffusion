# src/utils/tensorboard_utils.py

from torchvision.utils import make_grid

def log_images_tb(writer, real_batch, gen_batch, epoch):
    if writer is None:
        return
    grid_real = make_grid(real_batch, nrow=4, normalize=True)
    grid_gen = make_grid(gen_batch, nrow=4, normalize=True)
    writer.add_image("Grid/Real", grid_real, epoch)
    writer.add_image("Grid/Generated", grid_gen, epoch)
    # Optionally: add single sample images
    writer.add_image("Sample/Real", real_batch[0], epoch)
    writer.add_image("Sample/Generated", gen_batch[0], epoch)

def log_scalar_tb(writer, tag, value, step):
    if writer is not None:
        writer.add_scalar(tag, value, step)
