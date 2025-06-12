import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = model
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {name: param.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
