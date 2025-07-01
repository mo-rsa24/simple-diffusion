# Composable Diffusion Architectures

This note summarises several state-of-the-art approaches to compositional image generation beyond slot attention.  All models can be tested on the `ComposableColoredMNISTWithBBox` dataset.

Models are listed from the most recent first.

---

## 1. Compositional Diffusion Models *(Kim et al., 2024)*

**Core Idea**

Factorise the noise prediction network into separate **factor nets** (one per attribute) and fuse their scores with a learnable gating module.

**Blueprint**

- `FactorNet[i]` predicts a score for its concept (digit, colour, box).
- `GatingNetwork` merges all factor scores at each timestep.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.compositional_unet import CompositionalUNet
model = CompositionalUNet(dim=64, factors=3, beta_schedule="linear")
```

`forward(x, t, conditions)` expects a list of factor specific context tensors and returns the fused score.

**Training Hook-Ins**

- Optionally initialise each `FactorNet` from a pretrained checkpoint.
- The training loop can compute losses for each factor then backpropagate through the gating network.
- During sampling you can log the individual factor outputs before gating to inspect what each slot is doing.

---

## 2. Cascaded Conditional Diffusion *(Hoogeboom et al., 2023)*

**Core Idea**

Generate images in stages, where each stage conditions on the output of the previous one.  Each attribute is handled sequentially.

**Blueprint**

1. Stage 1 generates the digit shape.
2. Stage 2 adds the colour conditioned on Stage 1.
3. Stage 3 draws the bounding box on the intermediate image.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.cascaded_diffusion import CascadedDiffusion
cascade = CascadedDiffusion([shape_model, colour_model, box_model])
```

`forward(x, t, factors)` runs the list of stage models in order.

**Training Hook-Ins**

- Each stage can be trained independently and loaded from its own checkpoint.
- To compose new combinations simply chain the pretrained stages.
- Logging utilities can visualise the intermediate outputs after each stage.

---

## 3. Classifier‑Free Guidance for Compositional Control *(Ho & Salimans, 2022)*

**Core Idea**

Train a single model where any subset of the conditioning inputs may be dropped.  At inference interpolate between conditional and unconditional predictions for each factor.

**Blueprint**

- One `GuidedUNet` receives the image, timestep and a concatenation of all conditioning vectors.
- A binary mask specifies which conditions to drop on each training step.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.guided_unet import GuidedUNet
model = GuidedUNet(dim=64, slots=3)
```

`forward(x, t, cond_all, cond_mask)` returns a noise estimate depending on which conditions are active.

**Training Hook-Ins**

- Randomly zero out each slot of `cond_all` with 10–20% probability when calling the model.
- At sampling time use different `cond_mask` values to realise AND or OR style compositions.
- Log the guidance weights and resulting images for analysis.

---

## 4. Mixture‑of‑Experts Diffusion *(Zhu et al., 2023)*

**Core Idea**

Combine several expert UNets, each specialising in one concept.  A mixing network predicts per‑expert weights conditioned on the current image.

**Blueprint**

- `ExpertUNet[i]` models one attribute.
- `MixerNet` outputs the coefficients used to weight expert scores.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.mixture_of_experts import MixtureOfExperts
moe = MixtureOfExperts([shape_expert, colour_expert, box_expert])
```

`forward(x, t, masks)` stacks the expert predictions and returns their weighted sum.

**Training Hook-Ins**

- Each expert may be pretrained separately.  The mixer is then tuned on composite data.
- During sampling log the mixing coefficients to see which expert dominates.

---

## 5. Compositional SDE with Itô Superposition *(Sung et al., 2022)*

**Core Idea**

Superpose score functions from independently trained models by summing them and using Itô’s lemma for the reverse SDE.

**Blueprint**

- Load any number of pretrained score networks.
- Combine them using `ItoSuperposition` to obtain a single score function.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.ito_superposition import superposed_sample
image = superposed_sample([shape_net, colour_net], sampler=p_sample_loop, shape=(1,3,28,28))
```

**Training Hook-Ins**

- Use existing checkpoints and pass them into `superposed_sample` together with your diffusion sampler.
- Log both the individual slot predictions and the final generated images.

---

## 6. Classifier-Guided Diffusion

**Core Idea**

Combine a UNet with an auxiliary classifier. During training, minimize both the diffusion loss and a classification loss so the model learns representations aligned with digit labels.

**PyTorch Skeleton**

```python
from src.models.composable_architectures.classifier_guided_unet import ClassifierGuidedUNet
model = ClassifierGuidedUNet(dim=64, num_classes=10)
```

`forward(x, t, labels)` returns `(noise_pred, logits)` while training.

**Training Hook-Ins**

- Compute cross-entropy between logits and provided digit labels.
- Combine with MSE diffusion loss.
- Use logged samples to inspect classification accuracy and generative quality.

---