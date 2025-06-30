from datetime import timedelta
from typing import Dict
import time
import torch
from torch import nn

from src.models.classifier.base_classifier import BaseClassifier
from src.models.classifier.loss import compute_loss
from src.train.logging.training_logger_utils import log_training_start, log_epoch_start, log_batch, log_epoch_summary, \
    log_json, log_training_end
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.config.configs import Config
from src.evaluate import evaluate_multilabel
from src.utils.visualization import visualize_predictions

def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total = 0
    correct = {}       # For counting correct predictions per head
    label_counts = {}  # For per-head denominator

    with torch.no_grad():
        for batch in test_loader:
            data = batch["image"].to(device)
            # Find all label keys and move them to device
            label_keys = [k for k in batch if k.endswith("_label")]
            targets = {k: batch[k].to(device) for k in label_keys}

            logits_dict = model(data) if isinstance(model(data), dict) else dict(zip(
                [k.replace("_label", "_logits") for k in label_keys], model(data)
            ))

            # Compute loss and accuracy for each present head
            loss = 0
            for logit_key, logits in logits_dict.items():
                label_type = logit_key.replace("_logits", "")
                gt_key = f"{label_type}_label"
                if gt_key in targets:
                    loss += loss_fn(logits, targets[gt_key])
                    preds = logits.argmax(dim=1)
                    correct[logit_key] = correct.get(logit_key, 0) + (preds == targets[gt_key]).sum().item()
                    label_counts[logit_key] = label_counts.get(logit_key, 0) + data.size(0)
            total_loss += loss.item()
            total += data.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    accs = {k: 100. * correct[k] / label_counts[k] for k in correct}
    # Print flexible report
    print(f"\nTest set: Average loss: {avg_loss:.4f}")
    for logit_key in accs:
        label = logit_key.replace("_logits", "").capitalize()
        print(f"{label} Accuracy: {correct[logit_key]}/{label_counts[logit_key]} ({accs[logit_key]:.2f}%)")

    return avg_loss, accs



def train(
    cfg: Config,
    dirs: Dict,
    model: BaseClassifier,
    train_loader,
    val_loader,
    device,
    logger,
    eval_interval=1,
):
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer.type.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer.params)
    start_time = time.time()
    log_training_start(
        logger,
        model_name=cfg.model.type,
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        task=cfg.task,
        total_epochs=cfg.training.epochs,
        total_batches=len(train_loader),
    )
    global_step = 0
    start_epoch = 1
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()
        total_loss, correct_digit, correct_color, total = 0, 0, 0, 0
        log_epoch_start(epoch - 1, logger)
        acc_dict = {}
        n_samples = 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            for k in batch:
                if "_label" in k:
                    batch[k] = batch[k].to(device)
            logits_dict = model(images)
            loss = compute_loss(logits_dict, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_samples += images.size(0)

            # Compute accuracies
            for key in logits_dict:
                pred = logits_dict[key].argmax(1)
                label_key = key.replace("_logits", "_label")
                if label_key in batch:
                    acc = (pred == batch[label_key]).float().sum().item()
                    acc_dict[key] = acc_dict.get(key, 0) + acc

            global_step += 1
            if global_step % 1 == 0:
                log_batch(
                    step,
                    loss,
                    cfg.optimizer.params.get("lr", 1e-3),
                    logger,
                    writer=None,
                    wandb_tracker=None,
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")
        for k in acc_dict:
            print(f"  {k.replace('_logits', '')} acc = {acc_dict[k] / n_samples:.4f}")
        epoch_time = time.time() - epoch_start
        log_epoch_summary(logger, epoch, cfg.training.epochs, avg_loss, epoch_time=epoch_time)
        log_json(logger, "Epoch Summary", epoch=epoch, train_loss=avg_loss, duration=epoch_time)
        if val_loader and (epoch + 1) % eval_interval == 0:
            evaluate_multilabel(model, val_loader, device, criterion)
            visualize_predictions(model, dirs, val_loader, device, epoch=epoch, n=4)
    total_time = time.time() - start_time
    log_training_end(logger, total_time)
    alert_on_success(
        experiment_id=cfg.experiment_id,
        run_id=cfg.run_id,
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )