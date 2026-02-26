# HFA-GAN Stage 2 — CycleGAN with Frequency Consistency

## Quick Start

```bash
# Step 1: Pre-compute Stage 1 predictions
python precompute_stage1.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --input_dir /path/to/data/train/64mT \
    --output_dir /path/to/data/train/stage1_pred

python precompute_stage1.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --input_dir /path/to/data/val/64mT \
    --output_dir /path/to/data/val/stage1_pred

# Step 2: Train Stage 2
python train.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --data_dir /path/to/data \
    --lambda_cyc 10.0 --lambda_id 5.0 --lambda_freq 10.0

# Step 3: Inference
python inference.py \
    --checkpoint outputs/stage2/checkpoints/best_model.pth \
    --input_dir /path/to/new_scans \
    --output_dir /path/to/enhanced
```

## Data Structure

```
data_dir/
├── train/
│   ├── 64mT/           ← 64mT NIfTI volumes
│   ├── 3T/             ← 3T volumes (unpaired OK)
│   └── stage1_pred/    ← Pre-computed Stage 1 outputs
└── val/
    ├── 64mT/           ← Paired validation
    ├── 3T/             ← Paired ground truth
    └── stage1_pred/    ← Stage 1 outputs for val
```

## Key Hyperparameters

| Param | Default | What it controls |
|-------|---------|-----------------|
| `lambda_cyc` | 10.0 | Cycle consistency. Higher = more structure preservation |
| `lambda_id` | 5.0 | Identity loss. Prevents unnecessary changes |
| `lambda_freq` | **10.0** | **Frequency consistency. Our key contribution** |
| `freq_cutoff` | 0.1 | Low-pass filter cutoff (0.1 = keep lowest 10%) |

## Healthy Training Signs

- G_cyc should be ~20-40% of G_total (NOT <3% like before)
- G_freq should be non-zero and contributing
- Val PSNR ≥ 22 (not crashing to 20)
- Val SSIM ≥ 0.75
- Texture ratios approaching 1.0

## Files

```
config.py              — Hyperparameters
networks.py            — Generator (3D U-Net) + Discriminator (PatchGAN)
losses.py              — LSGAN + Cycle + Identity + Frequency Consistency
dataset.py             — Unpaired train + Paired val datasets
utils.py               — Metrics, scheduling, logging
train.py               — Main training loop
precompute_stage1.py   — Generate S1 predictions before training
inference.py           — Run trained model on new data
```
