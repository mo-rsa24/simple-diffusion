from src.debug.composable_debug import DebugPipeline

# path where MNIST data will be downloaded
DATA_ROOT = "../../data"

pipeline = DebugPipeline(root=DATA_ROOT, out_dir="debug_outputs", batch_size=4)
model = pipeline.sanity_check_overfit(epochs=5)
pipeline.single_timestep(model)
