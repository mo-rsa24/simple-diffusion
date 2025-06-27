import torch
from torch.profiler import profile, ProfilerActivity
from src.models.vanilla.unet import Unet
import logging
from datetime import datetime

# === Logging Setup ===
log_filename = f"profile_unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

def log_section(title):
    border = "=" * (len(title) + 4)
    logging.info(f"\n{border}\n= {title} =\n{border}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=1,
    use_attention_at=[1]
).to(device)

dummy_x = torch.randn(1, 1, 256, 256).to(device)
dummy_t = torch.randint(0, 300, (1,)).to(device)

# --- Profiling Section ---
log_section("Starting PyTorch Profiler on U-Net Forward Pass")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    with torch.no_grad():
        model(dummy_x, dummy_t)
log_section("Profiler Run Completed")

# --- Print & Log Results ---
summary = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20)
log_section("CUDA Memory Usage Summary (Top 20 Ops)")
logging.info("\n" + summary)

# Save detailed profiling info for deep analysis
with open(f"profile_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
    f.write("Detailed PyTorch Profiler Output\n\n")
    f.write(summary)
    f.write("\n\n")
    f.write("Full Profiler Table:\n")
    f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100))

log_section("Profiler results have been saved to log and output files.")

# --- Human-Readable Guidance ---
logging.info("""
What do these columns mean?
- 'Self CUDA Mem': Memory allocated by this operation alone.
- 'CUDA Mem': Total memory held after this operation (including children).
- 'CPU time'/'CUDA time': Execution times.
- 'Number of Calls': How many times this op ran.

How to use this output:
- Look for ops with high 'Self CUDA Mem' to find memory bottlenecks.
- Compare results after architectural or batch size changes to measure improvements.
""")
