"""
HFA-GAN Stage 1 Evaluation
============================
Evaluate a trained Stage 1 model on the test set.
Computes metrics and saves predictions as NIfTI files.

Usage:
    python evaluate.py --checkpoint ./experiments/stage1/checkpoints/best_G1.pth
    python evaluate.py --checkpoint ./experiments/stage1/checkpoints/best_G1.pth --save_outputs
"""

import os
import argparse
import torch
import numpy as np
from torch.cuda.amp import autocast

from config import Config
from dataset import build_dataloaders
from models.generator import DualAttentionResidualUNet3D
from models.losses import Stage1Loss
from utils import (
    compute_all_metrics,
    load_checkpoint,
    save_nifti,
    tensor_to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="HFA-GAN Stage 1 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (best_G1.pth)")
    parser.add_argument("--output_dir", type=str, default="./experiments/stage1/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_outputs", action="store_true",
                        help="Save predicted NIfTI files")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(config, checkpoint_path, output_dir, save_outputs=False, gpu_id=0):
    """Run evaluation on test set."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HFA-GAN Stage 1 Evaluation")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Build model
    model = DualAttentionResidualUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_features=config.base_features,
        norm_type=config.norm_type,
        use_attention_gates=config.use_attention_gates,
        use_self_attention=config.use_self_attention,
    ).to(device)

    # Load weights
    load_checkpoint(model, load_path=checkpoint_path)
    model.eval()

    # Build dataloader (only need test set)
    _, _, test_loader = build_dataloaders(config)
    print(f"Test subjects: {len(test_loader)}")

    # Build loss for computing loss values
    criterion = Stage1Loss(
        lambda_l1=config.lambda_l1,
        lambda_gdl=config.lambda_gdl,
        lambda_perceptual=config.lambda_perceptual,
        device=str(device),
    )

    # Evaluate
    all_metrics = []
    all_losses = []

    for i, batch in enumerate(test_loader):
        x = batch["lr"].to(device)
        y = batch["hr"].to(device)
        name = batch["name"][0]

        # Forward pass
        with autocast(dtype=torch.float16):
            y_pred = model(x)
            loss, loss_dict = criterion(y_pred.float(), y.float())

        # Compute metrics
        metrics = compute_all_metrics(y_pred.float(), y.float())
        metrics["name"] = name
        metrics["loss"] = loss.item()
        all_metrics.append(metrics)
        all_losses.append(loss_dict)

        print(f"  [{i+1}/{len(test_loader)}] {name}: "
              f"PSNR={metrics['psnr']:.2f}, "
              f"SSIM={metrics['ssim']:.4f}, "
              f"MAE={metrics['mae']:.4f}, "
              f"Loss={loss.item():.4f}")

        # Save outputs
        if save_outputs:
            base = name.replace(".nii.gz", "").replace(".nii", "")
            save_nifti(tensor_to_numpy(x), os.path.join(output_dir, f"{base}_input.nii.gz"))
            save_nifti(tensor_to_numpy(y_pred), os.path.join(output_dir, f"{base}_pred.nii.gz"))
            save_nifti(tensor_to_numpy(y), os.path.join(output_dir, f"{base}_target.nii.gz"))
            save_nifti(tensor_to_numpy(y_pred - x), os.path.join(output_dir, f"{base}_residual.nii.gz"))

    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"  AGGREGATE TEST RESULTS ({len(all_metrics)} subjects)")
    print(f"{'='*60}")

    psnr_vals = [m["psnr"] for m in all_metrics]
    ssim_vals = [m["ssim"] for m in all_metrics]
    mae_vals = [m["mae"] for m in all_metrics]

    print(f"  PSNR:  {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f}")
    print(f"  SSIM:  {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
    print(f"  MAE:   {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")

    # Save results to text file
    results_path = os.path.join(output_dir, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"HFA-GAN Stage 1 Test Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Subjects: {len(all_metrics)}\n\n")
        f.write(f"PSNR:  {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f}\n")
        f.write(f"SSIM:  {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}\n")
        f.write(f"MAE:   {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}\n\n")
        f.write(f"Per-subject results:\n")
        for m in all_metrics:
            f.write(f"  {m['name']}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}, MAE={m['mae']:.4f}\n")

    print(f"\nResults saved to: {results_path}")
    if save_outputs:
        print(f"NIfTI outputs saved to: {output_dir}")

    return all_metrics


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    config.gpu_id = args.gpu_id

    evaluate(config, args.checkpoint, args.output_dir, args.save_outputs, args.gpu_id)
