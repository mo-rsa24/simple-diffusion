# src/utils/logger.py

import logging, os, json
from datetime import datetime

def init_logger(log_dir, filename="training.log", log_to_stdout=False):
    os.makedirs(log_dir, exist_ok=True)

    # --- Clear previous logs ---
    for fname in ["training.log", "error_only.log", "training.jsonl"]:
        path = os.path.join(log_dir, fname)
        if os.path.exists(path):
            with open(path, 'w'): pass  # Truncate to empty file

    # --- Init logger ---
    full_path = os.path.join(log_dir, filename)
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Reset handlers (important!)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File logger
    fh = logging.FileHandler(full_path, mode='w')  # 'w' = overwrite
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Error log
    error_path = os.path.join(log_dir, "error_only.log")
    eh = logging.FileHandler(error_path, mode='w')
    eh.setLevel(logging.WARNING)
    eh.setFormatter(formatter)
    logger.addHandler(eh)

    # JSON structured log
    class JSONLinesHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None, delay=False):
            super().__init__(filename, mode, encoding, delay)

        def emit(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage()
            }
            if hasattr(record, "extra_data"):
                log_entry.update(record.extra_data)
            if self.stream is None:
                self.stream = self._open()
            self.stream.write(json.dumps(log_entry) + "\n")
            self.flush()

    json_handler = JSONLinesHandler(os.path.join(log_dir, "training.jsonl"), mode='w')
    logger.addHandler(json_handler)

    # Console logger
    if log_to_stdout:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info("🧠 Logger initialized (cleared previous logs)")
    return logger
