# Colored MNIST + BBoxes: Superposition Study

This document sketches how to train slot‑factorized diffusion models on
colored digits with bounding boxes and how to combine two trained models
using the Itô density estimator.

## Training

* Use `src/dataset/ComposableColoredMNISTWithBBox` for data loading.
* Train `src/models/composable_diffusion/ComposableDiffusionModel`
  with `src/train/generate/composable/composable_train.py`.
* Each slot predicts noise for digit shape, digit colour and box colour
  independently. The `sample_slots` method merges them for generation.

## Itô Superposition

The module `src/models/superposition/ito_operator.py` implements a tiny
`ItoSuperposition` class. Provide it with a list of pretrained score
networks and pass the resulting score function to a diffusion sampler,
for example `p_sample_loop` from `src/models/vanilla/diffusion.py`.

```python
from src.models.superposition.ito_operator import and_sampler
from src.models.vanilla.diffusion import p_sample_loop

result = and_sampler([model_a, model_b], p_sample_loop,
                     shape=(4, 3, 28, 28))
```

## Evaluation snippets

See `src/evaluation/composite_metrics.py` for helpers that compute
occlusion overlap, bounding box IoU and simple colour‑consistency
metrics. These utilities can be used to analyse the superposed samples
and compare them against naïve overlays.
