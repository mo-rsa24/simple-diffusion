from datetime import timedelta
from typing import Dict
import time
import torch
from torch import nn
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    log_json, log_training_end
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.config.configs import Config
from src.evaluate import evaluate_colored_mnist
from src.utils.visualization import visualize_predictions

def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    correct_digit = 0
    correct_color = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in test_loader:
            data = batch["image"].to(device)
            digit_target = batch["digit_label"].to(device)
            color_target = batch["color_label"].to(device)

            digit_out, color_out = model(data)
            test_loss += (loss_fn(digit_out, digit_target) + loss_fn(color_out, color_target)).item()

            pred_digit = digit_out.argmax(dim=1)
            pred_color = color_out.argmax(dim=1)

            correct_digit += pred_digit.eq(digit_target).sum().item()
            correct_color += pred_color.eq(color_target).sum().item()
            total += data.size(0)

    test_loss /= len(test_loader.dataset)
    acc_digit = 100. * correct_digit / total
    acc_color = 100. * correct_color / total
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Digit Accuracy: {correct_digit}/{total} ({acc_digit:.2f}%), "
          f"Color Accuracy: {correct_color}/{total} ({acc_color:.2f}%)")


def train(cfg: Config, dirs: Dict, model, train_loader, val_loader, device, logger, epochs=5, eval_interval=1):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=epochs,
        total_batches=len(train_loader)
    )
    global_step = 0
    start_epoch = 1
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        total_loss, correct_digit, correct_color, total = 0, 0, 0, 0
        log_epoch_start(epoch - 1, logger)

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            digit_labels = batch["digit_label"].to(device)
            color_labels = batch["color_label"].to(device)

            optimizer.zero_grad()
            digit_logits, color_logits = model(images)

            loss_digit = criterion(digit_logits, digit_labels)
            loss_color = criterion(color_logits, color_labels)
            loss = loss_digit + loss_color
            loss.backward()
            optimizer.step()

            # Accuracy
            pred_digit = digit_logits.argmax(1)
            pred_color = color_logits.argmax(1)
            correct_digit += (pred_digit == digit_labels).sum().item()
            correct_color += (pred_color == color_labels).sum().item()
            total += images.size(0)
            total_loss += loss.item()

            global_step += 1
            if global_step % 1 == 0:
                log_batch(step, loss, learning_rate, logger, writer=None, wandb_tracker=None)
            break
        avg_loss = total_loss / len(train_loader)
        avg_correct_digit = correct_digit / len(train_loader)
        avg_correct_color = correct_color / len(train_loader)
        epoch_time = time.time() - epoch_start
        log_epoch_summary(logger, epoch, epochs, avg_loss, epoch_time=epoch_time)
        log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, "
              f"Digit Acc={avg_correct_digit:.4f}, Color Acc={avg_correct_color:.4f}")
        if val_loader and (epoch + 1) % eval_interval == 0:
            evaluate_colored_mnist(model, val_loader, device, criterion)
            visualize_predictions(model, dirs, val_loader, device, epoch=epoch, n=4)
        break
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )