"""
HFA-GAN Stage 1 Configuration
==============================
All hyperparameters in one place. Change things here, not in the code.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ==================== PATHS ====================
    # Base data directory (contains train/, val/, test/ with high_field/ and low_field/ subfolders)
    data_root: str = "/mnt/Data/AKIB/CALSNIC2_T1W1_experiment/dataset_split/preprocessed/5_final"

    # Subfolder names inside each split
    hr_subfolder: str = "high_field"   # High-res target (T1w 1.0mm)
    lr_subfolder: str = "low_field"    # Low-res input  (T1w 0.8mm)

    # Output directories
    output_dir: str = "./experiments/stage1"
    checkpoint_dir: str = "./experiments/stage1/checkpoints"
    log_dir: str = "./experiments/stage1/logs"
    sample_dir: str = "./experiments/stage1/samples"

    # ==================== DATA ====================
    patch_size: Tuple[int, int, int] = (128, 128, 128)  # 3D patch dimensions
    num_workers: int = 4
    pin_memory: bool = True

    # Intensity normalization range (matches tanh output)
    intensity_range: Tuple[float, float] = (-1.0, 1.0)

    # ==================== MODEL ====================
    # Generator: 3D Dual-Attention Residual U-Net
    in_channels: int = 1          # T1 only for CALSNIC. Change to 3 for T1+T2+FLAIR
    out_channels: int = 1         # Same as input
    base_features: int = 32       # First encoder level features (32 -> 64 -> 128 -> 256 -> 512)
    num_levels: int = 4           # Encoder/decoder depth (4 levels + 1 bottleneck)
    use_attention_gates: bool = True    # Attention gates on skip connections
    use_self_attention: bool = True     # Self-attention at bottleneck and deepest levels
    use_residual_blocks: bool = True    # ResBlocks in each level
    norm_type: str = "instance"         # "instance" or "batch"
    activation: str = "leaky_relu"      # "leaky_relu" or "relu"
    leaky_slope: float = 0.01

    # ==================== LOSS WEIGHTS ====================
    lambda_l1: float = 1.0         # L1 pixel loss weight
    lambda_gdl: float = 0.5        # Gradient Difference Loss weight
    lambda_perceptual: float = 0.1  # Perceptual (MedicalNet 3D) loss weight

    # MedicalNet 3D ResNet-50 pretrained on 23 medical datasets
    # Download: https://github.com/Tencent/MedicalNet
    medicalnet_weights: str = "./pretrained/resnet_50_23dataset.pth"

    # ==================== TRAINING ====================
    batch_size: int = 1            # 3D volumes are large, usually batch=1
    num_epochs: int = 200
    lr: float = 2e-4               # Learning rate
    beta1: float = 0.5             # Adam beta1
    beta2: float = 0.999           # Adam beta2
    weight_decay: float = 1e-5

    # Learning rate scheduler
    lr_scheduler: str = "step"     # "step", "cosine", or "none"
    lr_step_size: int = 50         # For step scheduler: decay every N epochs
    lr_gamma: float = 0.5          # For step scheduler: multiply lr by this

    # Checkpointing
    save_every: int = 10           # Save checkpoint every N epochs
    validate_every: int = 5        # Run validation every N epochs
    sample_every: int = 10         # Save sample predictions every N epochs

    # Resume training
    resume_checkpoint: str = ""    # Path to checkpoint to resume from

    # ==================== HARDWARE ====================
    device: str = "cuda"           # "cuda" or "cpu"
    gpu_id: int = 0                # Which GPU to use
    mixed_precision: bool = True   # Use AMP for faster training

    def __post_init__(self):
        """Create output directories."""
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir, self.sample_dir]:
            os.makedirs(d, exist_ok=True)
