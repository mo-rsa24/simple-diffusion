import os


def get_log_paths(config: dict, base_dir="logs"):
    exp_id = config.get("experiment_id", "default")
    run_id = config.get("run_id", "default")
    vis_dir = os.path.join(base_dir, f"{exp_id}", f"{id}", "visuals")
    proj_dir = os.path.join(base_dir, f"{exp_id}", f"{id}", "projections")
    feature_dir = os.path.join(base_dir, f"{exp_id}", f"{run_id}", "features")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    return vis_dir, proj_dir, feature_dir
