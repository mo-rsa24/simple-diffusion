# src/utils/env.py
import os
import socket
import random
import torch
import numpy as np

def is_cluster():
    hostname = socket.gethostname()
    return "mscluster" in hostname or "wits" in hostname or os.environ.get("IS_CLUSTER") == "1"

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

