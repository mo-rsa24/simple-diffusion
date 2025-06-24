from typing import Dict
from datetime import timedelta
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from src.config.configs import Config
from src.evaluate import evaluate_mnist
from src.models.classifier.digit_classifier import SimpleCNN
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    log_json, log_training_end


def test(model: SimpleCNN, test_loader: DataLoader):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct} / {len(test_loader.dataset)}\n ({100. * correct / len(test_loader.dataset):.0f}%)) ")

def train(cfg: Config, dirs: Dict, model: SimpleCNN, train_loader: DataLoader, num_epochs: int, logger, device):
    learning_rate= 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=num_epochs,
        total_batches=len(train_loader)
    )
    global_step = 0
    start_epoch = 1
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        log_epoch_start(epoch - 1, logger)

        running_loss = 0.0
        for step, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            logits = model(data)
            loss = criterion(logits, targets)

            optimizer.zero_grad()  # Sets the gradients to zero for each batch so that it stores not store back prop calculation from previous props
            loss.backward()  # L= nonsense
            optimizer.step()  # theta_i = theta_i + alpha(some nonsense)

            running_loss += loss.item()
            global_step += 1
            if global_step % 1 == 0:
                log_batch(step, loss, learning_rate, logger, writer=None, wandb_tracker=None)
        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        log_epoch_summary(logger, epoch, num_epochs, avg_loss, epoch_time=epoch_time)
        log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)
        if epoch % 1 == 0:
            evaluate_mnist(cfg, dirs, model, data, targets, epoch=epoch, device=device)
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )