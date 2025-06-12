# src/utils/projection_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import os

def plot_projection_tsne(features, labels, title, save_path, logger=None):
    reducer = TSNE(n_components=2)
    projection = reducer.fit_transform(features)
    _plot_projection_core(projection, labels, title, save_path, logger, "TSNE")

def plot_projection_umap(features, labels, title, save_path, logger=None):
    reducer = umap.UMAP(n_components=2)
    projection = reducer.fit_transform(features)
    _plot_projection_core(projection, labels, title, save_path, logger, "UMAP")

def _plot_projection_core(projection, labels, title, save_path, logger, method):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=labels, palette="tab10", s=20)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize='small')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    if logger:
        logger.info(f"📈 {method} projection saved at {save_path}")
