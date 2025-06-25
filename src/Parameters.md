`run_experiment.sh` to fully support all CLI combinations (including `-t`, `-c`, and `-n`), following the design in your `run.py` (which expects all those flags):

---

## **2. Explanation**

* All combinations (`-t`, `-c`, `-n`, etc.) are **optional** but passed to `run.py` **if present**.
* The script will **not fail** if you omit (for example) `-c` or `-n` for pure MNIST runs.
* Supports both long and short CLI flags (`--experiment`/`-e`, etc.).
* Ensures full compatibility with [the CLI API of `run.py`](#).

---

## **3. README Example**

Below is a **structured README** that clearly describes each combination and usage:

---

````markdown
# Colored MNIST Experiment Runner

## Overview

This script (`run_experiment.sh`) lets you train classifiers or generate images on MNIST and colored/bounding-box variants using state-of-the-art diffusion models and classifiers.

---

## Prerequisites

- Python 3.x
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
````

* Download or place your data in `./data/`

---

## Usage

To launch an experiment, run:

```bash
./src/scripts/run_experiment.sh --experiment <EXP_ID> --run <RUN_ID> --dataset <DATASET> [--task <TASK>] [--color <COLOR>] [--number <DIGIT>] [--use_wandb] [--use_tensorboard]
```

Or using short flags:

```bash
./run_experiment.sh -e <EXP_ID> -r <RUN_ID> -d <DATASET> [-t <TASK>] [-c <COLOR>] [-n <DIGIT>] [--use_wandb] [--use_tensorboard]
```

---

## Examples

### 1. Classify Background Colored MNIST

```bash
./run_experiment.sh -e mnist_bg_color_1 -r 99 -d MNIST_COLOR -t classify -c bg
```

### 2. Generate Foreground Colored MNIST (Target digit 3)

```bash
./run_experiment.sh -e mnist_fg_color_target_digit_1 -r 99 -d MNIST_COLOR -t generate -c fg -n 3
```

### 3. Classify Standard MNIST Digits

```bash
./run_experiment.sh -e mnist_digit_1 -r 99 -d MNIST -t classify
```

### 4. Generate Foreground Color Bbox MNIST (Target digit 8)

```bash
./run_experiment.sh -e mnist_fg_color_bbox_1 -r 99 -d MNIST_BBOX -t generate -c fg -n 8
```

---
✅ Supported Experiment Combinations
#### Classify background-colored MNIST
```bash
./run_experiment.sh -e mnist_bg_color_1 -r 99 -d MNIST_COLOR -t classify -c bg
```
#### Generate background-colored MNIST
```bash
./run_experiment.sh -e mnist_bg_color_1 -r 99 -d MNIST_COLOR -t generate -c bg
```
#### Classify background-colored MNIST with bounding boxes
```bash
./run_experiment.sh -e mnist_bg_color_bbox_1 -r 99 -d MNIST_BBOX  -t classify -c bg
```
#### Generate background-colored MNIST with bounding boxes
```bash
./run_experiment.sh -e mnist_bg_color_bbox_1_1  -r 99 -d MNIST_BBOX  -t generate -c bg
```
#### Classify standard MNIST digits
```bash
./run_experiment.sh -e mnist_digit_1 -r 99 -d MNIST -t classify
```
#### Generate standard MNIST digits
```bash
./run_experiment.sh -e mnist_digit_1 -r 99 -d MNIST -t generate
```
#### Generate foreground-colored MNIST (target digit 3)
```bash
./run_experiment.sh -e mnist_fg_color_target_digit_1 -r 99 -d MNIST_COLOR -t generate -c fg -n 3
```
#### Generate foreground-colored MNIST (target digit 8)
```bash
./run_experiment.sh -e mnist_fg_color_target_digit_1 -r 99 -d MNIST_COLOR -t generate -c fg -n 8
```
#### Generate foreground-colored MNIST (no target digit)
```bash
./run_experiment.sh -e mnist_fg_color_1 -r 99 -d MNIST_COLOR -t generate -c fg
```
#### Classify foreground-colored MNIST
```bash
./run_experiment.sh -e mnist_fg_color_1 -r 99 -d MNIST_COLOR -t classify -c fg
```
#### Classify foreground-colored MNIST with bounding boxes
```bash
./run_experiment.sh -e mnist_fg_color_bbox_1    -r 99 -d MNIST_BBOX  -t classify -c fg
```
#### Generate foreground-colored MNIST with bounding boxes
```bash
./run_experiment.sh -e mnist_fg_color_bbox_1    -r 99 -d MNIST_BBOX  -t generate -c fg
```
#### Generate foreground-colored MNIST with bounding boxes (target digit 3)
```bash
./run_experiment.sh -e mnist_fg_color_bbox_1    -r 99 -d MNIST_BBOX  -t generate -c fg -n 3
```
#### Generate foreground-colored MNIST with bounding boxes (target digit 8)
```bash
./run_experiment.sh -e mnist_fg_color_bbox_1    -r 99 -d MNIST_BBOX  -t generate -c fg -n 8
```
### 💾 Output
All outputs (images, logs, checkpoints, sample visualizations) are saved in:

Sample visualizations (e.g., comparison grids), loss curves, and generated images are inside subfolders such as results_samples/, logs/, or as configured in your run.py.


```bash
outputs/<experiment_id>/<run_id>/
```

---
### ⚙️ Notes & Tips
**Debugging:** The script launches with debugpy for remote debugging. Remove or comment out the debugpy line in run_experiment.sh if you want a standard run.

**WandB/TensorBoard:** Add --use_wandb and/or --use_tensorboard to enable experiment tracking/logging.

**CLI Flexibility:** You can mix short (-e) and long (--experiment) flags.

**Dataset Download:** The script will auto-download MNIST if not present. For custom colored or bbox variants, ensure preprocessing scripts (if any) have already run.

**Experiment Tracking:** Always use unique combinations of -e and -r for each independent experiment to avoid overwriting outputs.

---
### 🧩 How It Works
The bash script parses all flags and passes them directly to run.py, which:

Loads the appropriate config and data loaders.

Selects the correct dataset variant and preprocessing based on your arguments.

Runs either classification training/evaluation or generative modeling, as specified. 

All loader/task/variant/target logic is centralized in run.py for maintainability.

---
### 📞 Questions or Issues?
If you encounter unexpected behavior, check that your run.py and loader logic are up-to-date and support all CLI flags.

For further assistance, file an issue or contact the project maintainer.

---
## Flags and Arguments

| Flag               | Meaning                                             | Example values            |
| ------------------ | --------------------------------------------------- | ------------------------- |
| -e, --experiment   | Experiment ID                                       | mnist\_fg\_color\_bbox\_1 |
| -r, --run          | Run ID                                              | 99                        |
| -d, --dataset      | Dataset type (`MNIST`, `MNIST_COLOR`, `MNIST_BBOX`) | MNIST\_COLOR              |
| -t, --task         | Task type (`classify` or `generate`)                | generate                  |
| -c, --color        | Color variant (`fg` or `bg`)                        | fg                        |
| -n, --number       | Target digit for class-conditional tasks            | 3, 8                      |
| --use\_wandb       | Enable Weights & Biases logging                     |                           |
| --use\_tensorboard | Enable TensorBoard logging                          |                           |

---

## Outputs

* Results (images, checkpoints, logs) are saved in:

  ```
  outputs/<experiment_id>/<run_id>/
  ```

## Notes

* The `--color` and `--number` flags are only relevant for colored or bbox datasets.
* You can run with or without debugpy by commenting/uncommenting the debugpy line in the script.
* For purely standard MNIST, omit `--color` and `--number`.

---

```

---

**If you want the full copy-paste for your updated script and markdown, let me know!**
```
