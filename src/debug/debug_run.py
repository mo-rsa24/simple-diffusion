#!/usr/bin/env python
import argparse
import sys
import yaml
import torch

def parse_args():
    p = argparse.ArgumentParser(
        description="Debug runner: sanity-check training pipeline structure with mocks"
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print loaded config and exit"
    )
    return p.parse_args()

# --- STUBS & MOCKS ---
class MockUnet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # a dummy scalar parameter so we can observe grad flow
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t):
        # forward must accept (x, t); just return same shape as x
        out = x + 0.0 * self.dummy_param
        return out

class DummyOptimizer:
    def __init__(self, params, **kwargs):
        self._params = list(params)
    def zero_grad(self):
        print("[OPT] zero_grad()")
        for p in self._params:
            p.grad = None
    def step(self):
        print("[OPT] step()")

def p_losses(model, x_start, t, loss_type="l1", timesteps=None):
    # Always return a scalar tensor requiring grad
    loss = torch.tensor(0.5, requires_grad=True)
    print(f"[LOSS] computed dummy loss={loss.item()}")
    return loss

def sample(model, image_size, batch_size, channels, timesteps=None):
    shape = (batch_size, channels, image_size, image_size)
    out = torch.randn(shape)
    print(f"[SAMPLE] returning mock samples of shape {tuple(out.shape)}")
    return out

def num_to_groups(num, batch_size):
    # split batch_size into num nearly-equal groups
    base = batch_size // num
    rem = batch_size % num
    sizes = [base + (1 if i < rem else 0) for i in range(num)]
    print(f"[GROUPS] splitting batch_size={batch_size} into {sizes}")
    return sizes

def save_image(tensor, path, nrow=None):
    print(f"[SAVE_IMAGE] would save tensor shape {tuple(tensor.shape)} to {path}")

class DummyDataLoader:
    def __init__(self, batches, batch_shape):
        self.batches = batches
        self.batch_shape = batch_shape
    def __iter__(self):
        for i in range(self.batches):
            batch = torch.randn(self.batch_shape)
            print(f"[DATALOADER] yielding batch {i} with shape {tuple(batch.shape)}")
            yield batch
    def __len__(self):
        return self.batches

# --- MAIN FLOW ---
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.dry_run:
        print("--- CONFIG ---")
        print(yaml.dump(cfg, default_flow_style=False))
        sys.exit(0)

    # pull out key params (with defaults)
    ds_cfg = cfg.get("dataset", {})
    batch_size = ds_cfg.get("batch_size", 16)
    channels   = ds_cfg.get("channel", 1)
    image_size = ds_cfg.get("image_size", 28)

    training_cfg = cfg.get("training", {})
    save_and_sample_every = training_cfg.get("save_and_sample_every", 10)
    epochs = training_cfg.get("epochs", 1)

    diffusion_cfg = cfg.get("diffusion", {})
    timesteps = diffusion_cfg.get("timesteps", 50)
    loss_type = diffusion_cfg.get("loss_type", "l1")

    # MOCKED COMPONENTS
    device = torch.device("cpu")
    model = MockUnet(**cfg.get("model", {}).get("params", {})).to(device)
    optimizer = DummyOptimizer(model.parameters(), **cfg.get("optimizer", {}).get("params", {}))

    # dummy loaders: 2 batches each
    train_loader = DummyDataLoader(batches=2, batch_shape=(batch_size, channels, image_size, image_size))
    test_loader  = DummyDataLoader(batches=1, batch_shape=(batch_size, channels, image_size, image_size))

    # TRAINING LOOP (single epoch & batch, thanks to breaks)
    for epoch in range(epochs):
        print(f"=== Epoch {epoch+1}/{epochs} ===")
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            # sample t per example
            t = torch.randint(0, timesteps, (batch.size(0),), dtype=torch.long)
            print(f"[STEP {step}] batch.shape={tuple(batch.shape)}, t.shape={tuple(t.shape)}")

            # FORWARD + LOSS
            out = model(batch, t)
            print(f"[MODEL] output shape={tuple(out.shape)}")
            loss = p_losses(model, batch, t, loss_type=loss_type, timesteps=timesteps)

            # BACKWARD + OPTIMIZER
            loss.backward()
            # instrument gradient on our dummy param
            print(f"[GRAD] dummy_param.grad={model.dummy_param.grad}")
            optimizer.step()

            # SAMPLE & SAVE
            if step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                groups = num_to_groups(4, batch_size)
                imgs = [ sample(model, image_size, n, channels, timesteps) for n in groups ]
                all_imgs = torch.cat(imgs, dim=0)
                print(f"[CONCAT] all_imgs.shape={tuple(all_imgs.shape)}")
                save_image(all_imgs, f"debug-sample-{milestone}.png", nrow=4)

            # only run first batch
            break
        # only run first epoch
        break

    # FINAL SAMPLING
    final_samples = sample(model, image_size, batch_size=8, channels=channels, timesteps=timesteps)
    print(f"[FINAL SAMPLE] shape={tuple(final_samples.shape)}")

if __name__ == "__main__":
    main()
