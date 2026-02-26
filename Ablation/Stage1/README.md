# HFA-GAN Stage 1: Structural Anchor

## What is this?

Stage 1 of the HFA-GAN (Hybrid Frequency-Attention GAN) pipeline. This trains a **3D Dual-Attention Residual U-Net** on paired low-res/high-res brain MRI data to learn the structural transformation.

After Stage 1 training, the saved weights (`best_G1.pth`) are used to initialize Stage 2 (CycleGAN refinement).

## Project Structure

```
hfa_gan/
├── config.py              # All hyperparameters in one place
├── dataset.py             # NIfTI data loading and patch extraction
├── train.py               # Main training loop
├── evaluate.py            # Standalone evaluation on test set
├── utils.py               # Metrics (PSNR/SSIM), checkpointing, logging
├── models/
│   ├── __init__.py
│   ├── generator.py       # 3D Dual-Attention Residual U-Net
│   └── losses.py          # L1 + GDL + Perceptual loss
└── README.md
```

## Quick Start

### 1. Edit paths in config.py

```python
data_dir_hr = "/mnt/Data/AKIB/CALSNIC2_T1W1_experiment/CALSNIC2_T1W1mm"
data_dir_lr = "/mnt/Data/AKIB/CALSNIC2_T1W1_experiment/CALSNIC2_T1W8mm"
split_dir   = "/mnt/Data/AKIB/CALSNIC2_T1W1_experiment/dataset_split"
```

### 2. Train

```bash
# Default settings
python train.py

# Custom settings
python train.py --num_epochs 300 --lr 1e-4 --lambda_gdl 1.0

# Without perceptual loss (faster, less VRAM)
python train.py --no_perceptual

# Resume from checkpoint
python train.py --resume_checkpoint ./experiments/stage1/checkpoints/epoch_100.pth
```

### 3. Evaluate

```bash
python evaluate.py --checkpoint ./experiments/stage1/checkpoints/best_G1.pth --save_outputs
```

## Architecture Summary

```
Input [1, 128³] ──→ Conv7×7 ──→ [32, 128³]
                                    │
              Encoder               │ skip (+ Attention Gate)
              ├─ Enc1: [64,  64³]   │
              ├─ Enc2: [128, 32³]   │
              ├─ Enc3: [256, 16³] + Self-Attn
              └─ Bottleneck: [512, 8³] + Self-Attn
                                    │
              Decoder               │
              ├─ Dec4: [256, 16³] + Self-Attn
              ├─ Dec3: [128, 32³]
              ├─ Dec2: [64,  64³]
              └─ Dec1: [32,  128³]
                                    │
                   Conv7×7 + Tanh ──→ Residual Δ [1, 128³]
                                    │
              Output = Input + Δ ──→ Enhanced [1, 128³]
```

## Loss Function

```
L_S1 = λ₁ × L1(y, ŷ)  +  λ_GDL × GDL(y, ŷ)  +  λ_VGG × Perceptual(y, ŷ)
```

| Loss | What it does | Default weight |
|------|-------------|----------------|
| L1 | Every voxel should be close to ground truth | 1.0 |
| GDL | Edges should be as sharp as ground truth | 0.5 |
| Perceptual | VGG features should match structurally | 0.1 |

## Output

After training completes:
- `experiments/stage1/checkpoints/best_G1.pth` — **Use this for Stage 2 initialization**
- `experiments/stage1/logs/` — Training logs and metrics JSON
- `experiments/stage1/samples/` — Sample predictions as NIfTI files

## Requirements

```
torch >= 2.0
torchvision
nibabel
numpy
```

## Dataset Split Format

The `dataset_split` directory should contain text files listing subject IDs:
```
train.txt    # One subject ID per line
val.txt
test.txt
```

If split files are not found, automatic 60/20/10/10 split is performed.
