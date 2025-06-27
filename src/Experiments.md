## 🛠️ Local Testing vs Cluster Launch

- **`./run_experiment.sh`**  
  Use this for **testing and debugging** locally or interactively on the login node.  
  It is NOT intended for launching jobs on the SLURM cluster.

- **`./launch_experiment.sh`**  
  Use this to **submit experiments to the cluster** via SLURM.  
  It takes exactly the same arguments and will queue your experiment as a SLURM job.

---

## 🖥️ Monitoring and Debugging SLURM Jobs

### 🔎 **Check Job Status**

- **See all your jobs in the queue:**  
  ```bash
  squeue -u $USER


* **List job history (recent and past jobs):**

  ```bash
  sacct -u $USER --format=JobID,JobName,Partition,State,ExitCode,Elapsed
  ```

* **Show details for a specific job (replace 12345):**

  ```bash
  sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
  ```

* **Live job output tail (replace jobid with SLURM job number):**

  ```bash
  tail -f slurm_logs/<jobname>_<jobid>.out
  tail -f slurm_logs/<jobname>_<jobid>.err
  ```

### 🚦 **Diagnosing Failed Jobs**

1. **Look for failed jobs:**

   ```bash
   sacct -u $USER --state=FAILED --format=JobID,JobName,State,ExitCode
   ```

2. **Check the `.err` file for error messages:**

   ```
   less slurm_logs/<jobname>_<jobid>.err
   ```

3. **Check the `.out` file for info/debug prints:**

   ```
   less slurm_logs/<jobname>_<jobid>.out
   ```

4. **Check the exit code and SLURM state for clues:**

   * `OUT_OF_MEMORY` → Try reducing batch size or requesting more memory.
   * `CANCELLED` → Job was cancelled manually or due to timeout.
   * `NODE_FAIL` → Compute node error, try resubmitting.

5. **Interactive troubleshooting:**

   * Use `scontrol show job <jobid>` for full job info.
   * Search logs for `Traceback`, `Error`, or Python exceptions.

### 📊 **Other Job Monitoring Tools**

* **Job progress:**
  If your scripts print progress updates, monitor with:

  ```bash
  tail -f slurm_logs/<jobname>_<jobid>.out
  ```
* **Cluster-wide usage (admin):**

  ```bash
  sinfo
  squeue
  ```
* **Resource usage after job finishes:**

  ```bash
  seff <jobid>
  ```

  (`seff` shows efficiency, memory, and time utilization.)

### 🗃️ **Exploring Results and Outputs**

* Use **MobaXterm’s SFTP GUI** (or your preferred file explorer)
  to browse `slurm_logs/` and your experiment output folders.
  This is often the fastest way to inspect multiple results, images, or logs visually.

---

## 🧪 Experiment Command Table

| Category                              | Description                                             | Command                                                                                       |
|----------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Background Color MNIST**             | Classify colored MNIST (background color)               | `./run_experiment.sh -e mnist_bg_color_1 -r 99 -d MNIST_COLOR -t classify -c bg`              |
|                                        | Generate colored MNIST (background color)               | `./run_experiment.sh -e mnist_bg_color_1 -r 99 -d MNIST_COLOR -t generate -c bg`              |
| **Background Color + BBox MNIST**      | Classify colored MNIST with bounding boxes (bg color)   | `./run_experiment.sh -e mnist_bg_color_bbox_1 -r 99 -d MNIST_BBOX -t classify -c bg`          |
|                                        | Generate colored MNIST with bounding boxes (bg color)   | `./run_experiment.sh -e mnist_bg_color_bbox_1_1 -r 99 -d MNIST_BBOX -t generate -c bg`        |
| **Standard MNIST**                     | Classify standard MNIST digits                          | `./run_experiment.sh -e mnist_digit_1 -r 99 -d MNIST -t classify`                             |
|                                        | Generate standard MNIST digits                          | `./run_experiment.sh -e mnist_digit_1 -r 99 -d MNIST -t generate`                             |
| **Foreground Color MNIST**             | Generate colored MNIST (fg color, target digit 3)       | `./run_experiment.sh -e mnist_fg_color_target_digit_1 -r 99 -d MNIST_COLOR -t generate -c fg -n 3`  |
|                                        | Generate colored MNIST (fg color, target digit 8)       | `./run_experiment.sh -e mnist_fg_color_target_digit_1 -r 99 -d MNIST_COLOR -t generate -c fg -n 8`  |
|                                        | Generate colored MNIST (fg color, no target digit)      | `./run_experiment.sh -e mnist_fg_color_1_1 -r 99 -d MNIST_COLOR -t generate -c fg`             |
|                                        | Classify colored MNIST (fg color)                       | `./run_experiment.sh -e mnist_fg_color_1 -r 99 -d MNIST_COLOR -t classify -c fg`               |
| **Foreground Color + BBox MNIST**      | Classify colored MNIST with bounding boxes (fg color)   | `./run_experiment.sh -e mnist_fg_color_bbox_1 -r 99 -d MNIST_BBOX -t classify -c fg`           |
|                                        | Generate colored MNIST with bounding boxes (fg color)   | `./run_experiment.sh -e mnist_fg_color_bbox_1 -r 99 -d MNIST_BBOX -t generate -c fg`           |
|                                        | Generate colored MNIST with bounding boxes (fg, n=3)    | `./run_experiment.sh -e mnist_fg_color_bbox_1 -r 99 -d MNIST_BBOX -t generate -c fg -n 3`      |
|                                        | Generate colored MNIST with bounding boxes (fg, n=8)    | `./run_experiment.sh -e mnist_fg_color_bbox_1 -r 99 -d MNIST_BBOX -t generate -c fg -n 8`      |

---

**Legend:**
- `-e`: experiment ID
- `-r`: run ID
- `-d`: dataset (`MNIST`, `MNIST_COLOR`, `MNIST_BBOX`)
- `-t`: task (`classify`, `generate`)
- `-c`: color (`fg` = foreground, `bg` = background)
- `-n`: digit (for class-conditional generation)

## 📝 **Recommended SLURM Job Debugging Workflow**

1. **Launch your job:**

   ```bash
   ./launch_experiment.sh -e ... -r ... -d ... [other args]
   ```
2. **Monitor the queue and job status:**

   ```bash
   squeue -u $USER
   sacct -u $USER --format=JobID,JobName,Partition,State,ExitCode
   ```
3. **When finished (or failed):**

   * Check `.out` and `.err` logs for your job in `slurm_logs/`
   * If failed, inspect logs and error codes
   * Use `seff <jobid>` for resource usage
4. **Review results and images via MobaXterm SFTP GUI** for easy browsing.
5. **(Optional) Receive job results/failure by email** with the SLURM mail options above.

---

**By following this workflow, you can quickly diagnose issues, rerun experiments, and ensure reproducibility of your computational research!**

```
