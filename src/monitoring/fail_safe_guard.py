# src/monitoring/fail_safe_guard.py

from src.monitoring.environment_checker import get_gpu_memory_stats, detect_loss_anomaly
from src.monitoring.alert_notifier import send_failure_email

def fail_safe_guard(epoch, loss, run_id, config, checkpoint_path, enable_alerts=True):
    error_detected, reason = detect_loss_anomaly(loss)
    if error_detected:
        log_message = f"❌ Epoch {epoch}: {reason}"

        # Save failure message locally
        with open(f"{checkpoint_path}/failure_reason.txt", "w") as f:
            f.write(log_message + "\n")

        # Optional alert
        if enable_alerts:
            send_failure_email(
                run_id=run_id,
                reason=reason,
                epoch=epoch,
                checkpoint_path=checkpoint_path
            )

        return True, log_message

    # GPU checks (optional)
    stats = get_gpu_memory_stats()
    for idx, gpu in enumerate(stats):
        if gpu["used"] / gpu["total"] > 0.95:
            return True, f"⚠️ GPU {idx} nearing full usage: {gpu['used']} / {gpu['total']} MB"

    return False, ""
