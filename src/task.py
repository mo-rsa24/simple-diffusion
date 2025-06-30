from src.train.classifier.trainMNIST import train as train_mnist
from src.train.classifier.trainMNIST import test as test_mnist
from src.train.generate.simple_diffusion.train import train


def classify_task(cfg, dirs, model, train_loader, val_loader, test_loader, device, logger, eval_interval):
    train_mnist(cfg, dirs, model, train_loader, val_loader, device, logger, eval_interval=eval_interval)
    test_mnist(model, test_loader)
    
def generate_task(cfg, dirs, model, ema, train_loader, logger, device, writer, wandb_run):
    train(cfg, dirs, model, ema, train_loader, logger, device, writer=writer, wandb_run=wandb_run)
