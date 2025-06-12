# src/monitoring/environment_checker.py

import torch
import subprocess

def get_gpu_memory_stats():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"]
        ).decode("utf-8").strip().split('\n')
        stats = [tuple(map(int, line.split(','))) for line in output]
        return [{"used": used, "total": total} for used, total in stats]
    except Exception:
        return [{"used": 0, "total": 0}]

def detect_loss_anomaly(loss, threshold=2.0):
    if not torch.isfinite(loss):
        return True, "NaN or Inf detected in loss"
    if loss.item() > threshold:
        return True, f"Loss exploded: {loss.item():.4f}"
    return False, ""
