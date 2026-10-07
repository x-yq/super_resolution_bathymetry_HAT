"""
HAT super-resolution inference on MagicBathyNetData.

Project:
    Super-Resolution of Ocean Imagery for Improving Bathymetry Prediction

Project authors:
    Yeqiao Xu

This script follows the project training pipeline:
    MagicBathyNetDataLoader
        -> location-specific normalization
        -> resize Sentinel-2 input to 64x64
        -> HAT
        -> 2x super-resolution to 128x128
        -> denormalization and GeoTIFF export

"""

import argparse
import os

import numpy as np
import torch
from torchvision.transforms import transforms as T

from datasets.dataset_loader import dataset_loader
from models.hat_arch import HAT
from PSNR_SSIM import PSNR_SSIM

try:
    from skimage.transform import resize
except ImportError:
    resize = None


class ResizeWithCV2:
    """Resize an image using the same interpolation behavior as train.py."""

    def __init__(self, target_size, interpolate_mode="bilinear"):
        self.target_size = target_size
        self.mode = interpolate_mode

    def __call__(self, img):
        if resize is None:
            raise ImportError("scikit-image is required for the project resize pipeline.")
        return resize(
            np.array(img),
            self.target_size,
            order=1,
            mode="reflect",
            anti_aliasing=False,
        )


def build_model(upscale: int) -> HAT:
    """Build the exact HAT configuration used by the project training code."""
    return HAT(
        upscale=upscale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6] * 12,
        embed_dim=180,
        num_heads=[6] * 12,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Support checkpoints saved through DataParallel.
    state_dict = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HAT satellite-image super-resolution on MagicBathyNet test data."
    )
    parser.add_argument(
        "--location",
        required=True,
        choices=["agia_napa", "puck_lagoon"],
        help="MagicBathyNet location to evaluate.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained HAT checkpoint.",
    )
    parser.add_argument(
        "--root-dir",
        default="./datasets/data/",
        help="Root directory of the MagicBathyNet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/sr_inference",
        help="Directory for super-resolved GeoTIFFs and metrics.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="HAT upscale factor. The project uses 2x SR.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Same input/target preprocessing used by train.py.
    transform = T.Compose([
        ResizeWithCV2((64, 64), "bilinear"),
        T.ToTensor(),
    ])

    target_transform = T.Compose([
        ResizeWithCV2((128, 128), "bilinear"),
        T.ToTensor(),
    ])

    # IMPORTANT:
    # bathymetry=False selects the satellite-image branch:
    #   data/<location>/img/s2
    #   data/<location>/img/spot6
    # and uses location-specific S2/SPOT-6 normalization parameters.
    data_loader_manager = dataset_loader(
        dataset_name="MagicBathyNet",
        root_dir=args.root_dir,
        transform=transform,
        target_transform=target_transform,
        batch_size=1,
        num_workers=0,
        test_size=0.2,
        val_size=0.1,
        locations=[args.location],
        bathymetry=False,
    )

    test_loader = data_loader_manager.get_dataloader("test")
    dataset = data_loader_manager.datasets["test"]

    model = build_model(args.upscale)
    load_weights(model, args.weights, device)
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    metric = PSNR_SSIM()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for img_paths, label_paths, inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = model(inputs)

            psnr = metric.calculate_psnr_pt(
                targets,
                outputs,
                crop_border=0,
                test_y_channel=False,
                img_size=(128, 128),
            )[0]
            ssim = metric.calculate_ssim_pt(
                targets,
                outputs,
                crop_border=0,
                test_y_channel=False,
                img_size=(128, 128),
            )[0]

            total_psnr += float(psnr)
            total_ssim += float(ssim)
            count += 1

            # Convert CHW -> HWC for MagicBathyNet.save_as_tiff().
            output_np = (
                outputs[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .clip(0, 1)
            )

            base_name = os.path.basename(img_paths[0])
            save_path = os.path.join(args.output_dir, base_name)

            # Reuse the dataset's denormalization and GeoTIFF metadata handling.
            dataset.save_as_tiff(
                output_np,
                img_paths[0],
                save_path,
            )

            print(
                f"{base_name}: "
                f"PSNR={float(psnr):.4f} dB, SSIM={float(ssim):.4f}"
            )

    if count == 0:
        raise RuntimeError("The test dataloader contains no samples.")

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Location: {args.location}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Samples: {count}\n")
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")

    print("\nEvaluation complete.")
    print(f"Location : {args.location}")
    print(f"Samples  : {count}")
    print(f"PSNR     : {avg_psnr:.4f} dB")
    print(f"SSIM     : {avg_ssim:.4f}")
    print(f"Results  : {args.output_dir}")


if __name__ == "__main__":
    main()
