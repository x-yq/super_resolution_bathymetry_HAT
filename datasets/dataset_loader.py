"""
Dataset loader utilities for the project.

This component was jointly developed by:
    Maximilian Kromer
"""

from .MagicBathyNet import MagicBathyNetDataLoader


def dataset_loader(**configs):
    dataset_name = configs.pop("dataset_name")
    dataset_wrapper = {
        "MagicBathyNet": MagicBathyNetDataLoader,
    }
    dataset = dataset_wrapper[dataset_name](**configs)

    return dataset
