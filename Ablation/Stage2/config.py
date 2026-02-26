"""
Stage 2 Configuration — HFA-GAN CycleGAN Refinement.
All hyperparameters in one place.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Stage2Config:
    # Paths
    stage1_checkpoint: str = ""
    data_dir: str = ""
    output_dir: str = "outputs/stage2"
    
    # Data
    volume_size: int = 128
    in_channels: int = 1  # Single channel T1
    
    # Training
    num_epochs: int = 200
    batch_size: int = 1
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_decay_start: int = 100
    lr_decay_end: int = 200
    
    # Loss Weights (CRITICAL — these control the balance)
    lambda_cyc: float = 10.0    # Cycle consistency (standard CycleGAN = 10)
    lambda_id: float = 5.0      # Identity loss
    lambda_freq: float = 10.0   # Frequency consistency (OUR KEY CONTRIBUTION)
    
    # Frequency Loss
    freq_cutoff_ratio: float = 0.1  # Keep lowest 10% of frequencies
    
    # Generator architecture (same as Stage 1)
    g_init_features: int = 32
    g_encoder_channels: tuple = (64, 128, 256, 512)
    g_attention_levels: tuple = (2, 3)
    g_attention_bottleneck: bool = True
    
    # Discriminator
    d_base_features: int = 64
    d_n_layers: int = 3
    
    # Image buffer for discriminator stability
    buffer_size: int = 50
    
    # Logging
    log_interval: int = 10
    save_interval: int = 10
    val_interval: int = 5
    num_val_samples: int = 5
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.output_dir) / "checkpoints").mkdir(exist_ok=True)
        (Path(self.output_dir) / "samples").mkdir(exist_ok=True)
        (Path(self.output_dir) / "logs").mkdir(exist_ok=True)
