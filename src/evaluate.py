from pathlib import Path
from typing import Dict

import torch
from torchvision.utils import save_image
from src.config.configs import Config
from src.models.classifier.digit_classifier import SimpleCNN


def evaluate_mnist(cfg: Config, dirs: Dict,  model: SimpleCNN, data, target, epoch: int = None, device = 'cuda'):
    model.eval()
    real_batch = data[:cfg.sampling.batch_size].to(device=device)
    target_batch = target[:cfg.sampling.batch_size].to(device=device)

    samples_dir: Path = dirs.get("results_samples") / Path(f"epoch_{epoch}")
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, target) in enumerate(zip(real_batch, target_batch)):
        output = model(real_batch[i].unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True).item()
        save_image(img,  samples_dir / Path(f"target_{target}_prediction_{pred}_{i:03d}.png"))
    # tsne_plot(model, test_loader, n=2000)
    # visualize_feature_maps(model, data[0], selected_layers=['features.0'], save_dir="viz")


def evaluate_colored_mnist(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct_digit, correct_color, total = 0, 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            digit_labels = batch["digit_label"].to(device)
            color_labels = batch["color_label"].to(device)

            digit_logits, color_logits = model(images)

            loss_digit = loss_fn(digit_logits, digit_labels)
            loss_color = loss_fn(color_logits, color_labels)
            total_loss += (loss_digit + loss_color).item()

            pred_digit = digit_logits.argmax(1)
            pred_color = color_logits.argmax(1)
            correct_digit += (pred_digit == digit_labels).sum().item()
            correct_color += (pred_color == color_labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / len(loader)
    acc_digit = correct_digit / total
    acc_color = correct_color / total

    print(f"[Eval] Loss: {avg_loss:.4f}, Digit Acc: {acc_digit:.4f}, Color Acc: {acc_color:.4f}")
    return avg_loss, acc_digit, acc_color

def evaluate_multilabel(model, loader, device, loss_fn):
    import torch
    model.eval()
    total_loss = 0
    total = 0
    # Dynamically track correct predictions for each head
    correct = {}
    label_counts = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            # Find all label keys
            label_keys = [k for k in batch if k.endswith("_label")]
            # Move labels to device
            labels = {k: batch[k].to(device) for k in label_keys}

            logits_dict = model(images)

            # Calculate loss for all present keys
            loss = 0
            for logit_key, logits in logits_dict.items():
                label_type = logit_key.replace("_logits", "")
                gt_key = f"{label_type}_label"
                if gt_key in labels:
                    loss += loss_fn(logits, labels[gt_key])
                    # Accuracy
                    preds = logits.argmax(1)
                    corr = (preds == labels[gt_key]).sum().item()
                    correct[logit_key] = correct.get(logit_key, 0) + corr
                    label_counts[logit_key] = label_counts.get(logit_key, 0) + images.size(0)

            total_loss += loss.item()
            total += images.size(0)

    avg_loss = total_loss / len(loader)
    # Build flexible acc report
    accs = {k: (correct[k] / label_counts[k]) for k in correct}
    acc_str = ", ".join([f"{k.replace('_logits', '')} Acc: {accs[k]:.4f}" for k in accs])
    print(f"[Eval] Loss: {avg_loss:.4f}, {acc_str}")
    return avg_loss, accs
