from torch import nn


def compute_loss(logits_dict, batch, keys=None):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    if keys is None:
        # Auto-discover keys
        keys = [k.replace("_logits", "") for k in logits_dict]
    for key in keys:
        label_key = f"{key}_label"
        logits_key = f"{key}_logits"
        if logits_key in logits_dict and label_key in batch:
            loss += criterion(logits_dict[logits_key], batch[label_key])
    return loss

