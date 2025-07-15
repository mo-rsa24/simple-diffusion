from torch.utils.data import Dataset, Subset


def tiny_subset(dataset: Dataset, num_items: int = 8) -> Subset:
    """Return a tiny subset of the dataset for quick overfitting checks."""
    indices = list(range(min(len(dataset), num_items)))
    return Subset(dataset, indices)