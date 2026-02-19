"""
HFA-GAN Stage 1 Training
==========================
Trains the 3D Dual-Attention Residual U-Net on paired LR-HR data.

What happens in one training step:
  1. Load a paired patch (low-res x, high-res y)
  2. Generator predicts: ŷ = x + G(x)
  3. Compute loss: L1 + GDL + Perceptual
  4. Backpropagate and update G's weights
  5. Repeat

Usage:
    python train.py
    python train.py --num_epochs 300 --lr 1e-4
    python train.py --resume_checkpoint ./experiments/stage1/checkpoints/epoch_100.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from config import Config
from dataset import build_dataloaders
from models.generator import DualAttentionResidualUNet3D
from models.losses import Stage1Loss
from utils import (
    compute_all_metrics,
    save_checkpoint,
    load_checkpoint,
    save_nifti,
    tensor_to_numpy,
    TrainLogger,
    Timer,
)


def parse_args():
    """Parse command line arguments (override config values)."""
    parser = argparse.ArgumentParser(description="HFA-GAN Stage 1 Training")

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lambda_l1", type=float, default=None)
    parser.add_argument("--lambda_gdl", type=float, default=None)
    parser.add_argument("--lambda_perceptual", type=float, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--no_perceptual", action="store_true",
                        help="Disable perceptual loss (faster training)")
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def apply_args_to_config(args, config):
    """Override config values with command-line arguments."""
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lambda_l1 is not None:
        config.lambda_l1 = args.lambda_l1
    if args.lambda_gdl is not None:
        config.lambda_gdl = args.lambda_gdl
    if args.lambda_perceptual is not None:
        config.lambda_perceptual = args.lambda_perceptual
    if args.resume_checkpoint is not None:
        config.resume_checkpoint = args.resume_checkpoint
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id
    if args.no_perceptual:
        config.lambda_perceptual = 0.0
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.log_dir = os.path.join(args.output_dir, "logs")
        config.sample_dir = os.path.join(args.output_dir, "samples")
    return config


# ==============================================================
# Training One Epoch
# ==============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    epoch, logger, use_amp=True):
    """
    Train for one epoch.

    For each batch:
      1. Load paired patch (x=low-res, y=high-res)
      2. Forward: ŷ = model(x) = x + G(x)
      3. Compute loss: L_S1 = λ₁*L1 + λ_GDL*GDL + λ_VGG*Perceptual
      4. Backward: compute gradients
      5. Update: optimizer step
    """
    model.train()
    epoch_losses = {"l1": 0, "gdl": 0, "perceptual": 0, "total": 0}
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Get data
        x = batch["lr"].to(device)   # Low-res input
        y = batch["hr"].to(device)   # High-res target

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (with mixed precision if enabled)
        if use_amp:
            with autocast(dtype=torch.float16):
                y_pred = model(x)            # ŷ = x + G(x)
                loss, loss_dict = criterion(y_pred, y)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss, loss_dict = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # Accumulate losses
        for k in epoch_losses:
            epoch_losses[k] += loss_dict.get(k, 0)

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            logger.log_losses(epoch, batch_idx + 1, num_batches, loss_dict)

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


# ==============================================================
# Validation
# ==============================================================

@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    """
    Run validation and compute metrics.

    Returns:
      val_losses: Average loss values
      val_metrics: Average PSNR, SSIM, MAE
    """
    model.eval()
    val_losses = {"l1": 0, "gdl": 0, "perceptual": 0, "total": 0}
    val_metrics = {"psnr": 0, "ssim": 0, "mae": 0}
    num_batches = len(dataloader)

    for batch in dataloader:
        x = batch["lr"].to(device)
        y = batch["hr"].to(device)

        if use_amp:
            with autocast(dtype=torch.float16):
                y_pred = model(x)
                loss, loss_dict = criterion(y_pred, y)
        else:
            y_pred = model(x)
            loss, loss_dict = criterion(y_pred, y)

        # Accumulate losses
        for k in val_losses:
            val_losses[k] += loss_dict.get(k, 0)

        # Compute metrics (on float32)
        y_pred_f32 = y_pred.float()
        y_f32 = y.float()
        metrics = compute_all_metrics(y_pred_f32, y_f32)
        for k in val_metrics:
            val_metrics[k] += metrics[k]

    # Average
    for k in val_losses:
        val_losses[k] /= max(num_batches, 1)
    for k in val_metrics:
        val_metrics[k] /= max(num_batches, 1)

    return val_losses, val_metrics


# ==============================================================
# Save Sample Predictions
# ==============================================================

@torch.no_grad()
def save_samples(model, dataloader, device, save_dir, epoch, num_samples=2):
    """Save a few sample predictions as NIfTI files for visual inspection."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        x = batch["lr"].to(device)
        y = batch["hr"].to(device)
        name = batch["name"][0]

        y_pred = model(x)

        # Save as NIfTI
        base = name.replace(".nii.gz", "").replace(".nii", "")
        save_nifti(
            tensor_to_numpy(x),
            os.path.join(save_dir, f"epoch{epoch}_{base}_input.nii.gz"),
        )
        save_nifti(
            tensor_to_numpy(y_pred),
            os.path.join(save_dir, f"epoch{epoch}_{base}_pred.nii.gz"),
        )
        save_nifti(
            tensor_to_numpy(y),
            os.path.join(save_dir, f"epoch{epoch}_{base}_target.nii.gz"),
        )
        # Save difference map (what the residual network predicted)
        diff = tensor_to_numpy(y_pred - x)
        save_nifti(
            diff,
            os.path.join(save_dir, f"epoch{epoch}_{base}_residual.nii.gz"),
        )


# ==============================================================
# Main Training Function
# ==============================================================

def train(config):
    """
    Full Stage 1 training pipeline.

    1. Build model, losses, optimizer
    2. Load data
    3. Train loop with validation
    4. Save best model as best_G1.pth (used to initialize Stage 2)
    """
    # Setup device
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  HFA-GAN Stage 1 Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Logger
    logger = TrainLogger(config.log_dir)
    logger.log(f"Config: {config}")

    # ---- Build Model ----
    model = DualAttentionResidualUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_features=config.base_features,
        norm_type=config.norm_type,
        use_attention_gates=config.use_attention_gates,
        use_self_attention=config.use_self_attention,
    ).to(device)

    param_count = model.count_parameters()
    logger.log(f"Generator parameters: {param_count:,}")

    # ---- Build Loss ----
    criterion = Stage1Loss(
        lambda_l1=config.lambda_l1,
        lambda_gdl=config.lambda_gdl,
        lambda_perceptual=config.lambda_perceptual,
        device=str(device),
        medicalnet_weights=config.medicalnet_weights,
    ).to(device)

    # ---- Build Optimizer ----
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # ---- Build Scheduler ----
    if config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    elif config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # ---- AMP Scaler ----
    scaler = GradScaler() if config.mixed_precision else None

    # ---- Load Data ----
    logger.log("Loading datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    logger.log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_loss = float("inf")
    loss_history = []

    if config.resume_checkpoint:
        start_epoch, best_val_loss, loss_history = load_checkpoint(
            model, optimizer, scheduler, config.resume_checkpoint
        )
        logger.log(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- Training Loop ----
    timer = Timer()
    logger.log(f"\nStarting training for {config.num_epochs} epochs...")
    logger.log(f"Loss weights: L1={config.lambda_l1}, GDL={config.lambda_gdl}, MedNet3D={config.lambda_perceptual}")

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        timer.start()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, logger, use_amp=config.mixed_precision
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.lr

        # Validate
        val_losses = None
        val_metrics = None
        if epoch % config.validate_every == 0 or epoch == config.num_epochs:
            val_losses, val_metrics = validate(
                model, val_loader, criterion, device,
                use_amp=config.mixed_precision
            )
            logger.log(
                f"Epoch [{epoch}/{config.num_epochs}] "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f} | "
                f"SSIM: {val_metrics['ssim']:.4f} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {timer.elapsed_str()}"
            )

            # Save best model
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss, loss_history,
                    os.path.join(config.checkpoint_dir, "best_G1.pth"),
                )
                logger.log(f"  *** New best model! Val loss: {best_val_loss:.4f} ***")
        else:
            logger.log(
                f"Epoch [{epoch}/{config.num_epochs}] "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {timer.elapsed_str()}"
            )

        # Save loss history
        loss_history.append({
            "epoch": epoch,
            "train": train_losses,
            "val": val_losses,
            "val_metrics": val_metrics,
        })
        logger.log_epoch(epoch, train_losses, val_metrics)

        # Save periodic checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, loss_history,
                os.path.join(config.checkpoint_dir, f"epoch_{epoch}.pth"),
            )

        # Save sample predictions
        if epoch % config.sample_every == 0:
            save_samples(model, val_loader, device, config.sample_dir, epoch)
            logger.log(f"  Saved sample predictions to {config.sample_dir}")

    # ---- Final Evaluation on Test Set ----
    logger.log(f"\n{'='*60}")
    logger.log("Final evaluation on test set...")

    # Load best model
    load_checkpoint(model, load_path=os.path.join(config.checkpoint_dir, "best_G1.pth"))

    test_losses, test_metrics = validate(
        model, test_loader, criterion, device, use_amp=config.mixed_precision
    )
    logger.log(f"TEST RESULTS:")
    logger.log(f"  Loss: {test_losses['total']:.4f}")
    logger.log(f"  PSNR: {test_metrics['psnr']:.2f}")
    logger.log(f"  SSIM: {test_metrics['ssim']:.4f}")
    logger.log(f"  MAE:  {test_metrics['mae']:.4f}")

    # Save test predictions
    save_samples(model, test_loader, device,
                 os.path.join(config.output_dir, "test_predictions"),
                 epoch="final", num_samples=len(test_loader))

    logger.log(f"\nTraining complete! Best model saved at:")
    logger.log(f"  {os.path.join(config.checkpoint_dir, 'best_G1.pth')}")
    logger.log(f"\nThis file is used to initialize G_refine in Stage 2.")

    return model


# ==============================================================
# Entry Point
# ==============================================================

if __name__ == "__main__":
    args = parse_args()
    config = Config()
    config = apply_args_to_config(args, config)

    # Create directories
    config.__post_init__()

    train(config)
