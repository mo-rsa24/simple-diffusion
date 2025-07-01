# Composable Diffusion Toy Experiments

This document lists example studies to run with the compositional MNIST setup.

## Ablation Ideas

- **Slot attention vs. simple sum**: compare models with attention-based merging against naive addition of slot outputs.
- **Joint vs. separate training**: train all slots together or one after another and compose at inference.
- **Number of slots**: remove the bounding box slot and evaluate quality of digit + color.

## Metrics

- **LPIPS** between real and generated images for unseen attribute combos.
- **Slot classification accuracy** using pretrained classifiers on digit identity and colors.
- **FID** computed on composed images vs. original dataset samples.