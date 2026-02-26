"""
HFA-GAN Utilities
==================
Helper functions for:
  - Metrics (PSNR, SSIM)
  - Checkpoint save/load
  - NIfTI saving
  - Logging
"""

import os
import json
import time
import numpy as np
import torch
import nibabel as nib
from datetime import datetime


# ==============================================================
# Metrics
# ==============================================================

def compute_psnr(pred, target, data_range=2.0):
    """
    Peak Signal-to-Noise Ratio.

    Higher = better. Measures pixel-level accuracy.
    For data in [-1, 1], the data_range is 2.0.

    Formula: PSNR = 10 * log10(data_range^2 / MSE)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim_3d(pred, target, window_size=7, data_range=2.0):
    """
    Structural Similarity Index (simplified 3D version).

    Higher = better. Measures structural similarity.
    Considers luminance, contrast, and structure.

    This is a patch-based computation averaged over the volume.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Use average pooling as the window function
    kernel_size = window_size
    pad = kernel_size // 2

    # Compute means
    mu_pred = F.avg_pool3d(pred, kernel_size, stride=1, padding=pad)
    mu_target = F.avg_pool3d(target, kernel_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    # Compute variances
    sigma_pred_sq = F.avg_pool3d(pred ** 2, kernel_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool3d(target ** 2, kernel_size, stride=1, padding=pad) - mu_target_sq
    sigma_cross = F.avg_pool3d(pred * target, kernel_size, stride=1, padding=pad) - mu_cross

    # SSIM formula
    numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean().item()


# Need F for SSIM
import torch.nn.functional as F


def compute_mae(pred, target):
    """Mean Absolute Error. Lower = better."""
    return torch.mean(torch.abs(pred - target)).item()


def compute_all_metrics(pred, target):
    """Compute all metrics at once. Returns dict."""
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim_3d(pred, target),
        "mae": compute_mae(pred, target),
    }


# ==============================================================
# Checkpoint Management
# ==============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                    loss_history, save_path):
    """Save training state to a checkpoint file."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "loss_history": loss_history,
    }
    torch.save(checkpoint, save_path)
    print(f"  [Checkpoint] Saved to {save_path}")


def load_checkpoint(model, optimizer=None, scheduler=None, load_path=None):
    """
    Load training state from a checkpoint.
    Returns the epoch number, best_metric, and loss_history.
    """
    if not load_path or not os.path.exists(load_path):
        print("[Checkpoint] No checkpoint found, starting fresh.")
        return 0, float("inf"), []

    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  [Checkpoint] Loaded model from epoch {checkpoint['epoch']}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("best_metric", float("inf")),
        checkpoint.get("loss_history", []),
    )


# ==============================================================
# NIfTI Saving
# ==============================================================

def save_nifti(volume, save_path, affine=None):
    """
    Save a 3D numpy array as a NIfTI file.

    Args:
        volume: numpy array [D, H, W]
        save_path: output .nii.gz path
        affine: 4x4 affine matrix (uses identity if None)
    """
    if affine is None:
        affine = np.eye(4)

    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, save_path)


def tensor_to_numpy(tensor):
    """
    Convert a torch tensor to numpy array.
    Handles GPU tensors and removes batch/channel dims.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 5:  # [B, C, D, H, W] -> [D, H, W]
            tensor = tensor[0, 0]
        elif tensor.ndim == 4:  # [C, D, H, W] -> [D, H, W]
            tensor = tensor[0]
        return tensor.numpy()
    return tensor


# ==============================================================
# Logging
# ==============================================================

class TrainLogger:
    """Simple logger that prints to console and saves to file."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        self.metrics_history = []

    def log(self, message):
        """Print and save a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def log_epoch(self, epoch, train_losses, val_metrics=None):
        """Log epoch summary."""
        entry = {
            "epoch": epoch,
            "train": train_losses,
            "val": val_metrics,
        }
        self.metrics_history.append(entry)

        # Save metrics to JSON
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def log_losses(self, epoch, batch, total_batches, loss_dict):
        """Log batch-level losses."""
        parts = [f"Epoch [{epoch}] Batch [{batch}/{total_batches}]"]
        for k, v in loss_dict.items():
            parts.append(f"{k}: {v:.4f}")
        self.log(" | ".join(parts))


# ==============================================================
# Timer
# ==============================================================

class Timer:
    """Simple timer for measuring epoch/training duration."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def elapsed_str(self):
        elapsed = self.elapsed()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"
