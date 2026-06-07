from datasets.dataset_loader import dataset_loader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import tifffile
import os
import numpy as np


def visualize_data(loader, show=True, save=False, output_dir="."):
    """
    Visualizes pairs of images and labels from the data loader.

    Args:
        loader (DataLoader): Dataloader.
        show (bool): If True, show the plots.
        save (bool): If True, save the plots to 'output_dir'.
        output_dir (str): Directory where plots will be saved.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, (id,_, img, label) in enumerate(loader):

        img = img[0].permute(1, 2, 0)
        label = label[0].permute(1, 2, 0)

        axs[i, 0].imshow(img, aspect="auto")
        axs[i, 0].axis("off")
        axs[i, 0].set_title("Sentinel-2")

        axs[i, 1].imshow(label, aspect="auto")
        axs[i, 1].axis("off")
        axs[i, 1].set_title("SPOT-6")

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_dir}/visualization.png")

    if show:
        plt.show()


# Usage
transform = transforms.Compose([
    transforms.ToTensor(),
])
# configs = {
#     "dataset_name": "SpecificImages",
#     "root_dir": "./datasets/data/",
#     "transform": transform,
#     "batch_size": 1,
#     "num_workers": 0,
#     "test_size": 0.2,
#     "val_size": 0.1
# }
configs = {
            "dataset_name": "MagicBathyNet",
            #"root_dir": "./datasets/data/",
            "root_dir": "/faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/datasets/data",
            "transform": transform,
            #"target_transform":target_transform,
            "batch_size": 1,
            "num_workers": 0,
            "test_size": 0.2,
            "val_size": 0.1,
            "bathymetry": True
        }
data_loader_manager = dataset_loader(**configs)
train_loader = data_loader_manager.get_dataloader("train")
val_loader = data_loader_manager.get_dataloader("val")
test_loader = data_loader_manager.get_dataloader("test")

visualize_data(test_loader, show=True, save=False, output_dir=".")
