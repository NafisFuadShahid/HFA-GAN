"""
HFA-GAN Stage 1 Loss Functions
================================
Three losses combined:

1. L1 Loss         - Pixel-wise accuracy (every voxel close to ground truth)
2. GDL             - Gradient Difference Loss (edges must be sharp)
3. Perceptual Loss - MedicalNet 3D feature similarity (structures must look correct)

Combined: L_S1 = λ_L1 * L1 + λ_GDL * GDL + λ_perc * MedicalNet3D_Perceptual
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Loss 1: L1 Loss (Mean Absolute Error)
# ==============================================================

class L1Loss(nn.Module):
    """
    Simple pixel-wise L1 loss.

    For each voxel: |predicted - actual|
    Then average across all voxels.

    Why L1 and not L2 (MSE)?
      L2 squares errors, so it penalizes big errors way more than small ones.
      This makes the network "play it safe" and predict blurry averages.
      L1 treats all errors equally -> produces sharper results.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# ==============================================================
# Loss 2: Gradient Difference Loss (GDL)
# ==============================================================

class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss - makes edges sharp.

    How it works:
      1. Compute the "gradient" of both predicted and target images
         in x, y, and z directions.
         (Gradient = how much the intensity changes between neighboring voxels)
      2. Compare these gradients with L1 loss.

    Why? L1 loss treats all voxels equally. A flat region and an edge
    get the same weight. GDL specifically focuses on edges - the places
    where intensity changes rapidly. If the predicted edges don't match
    the real edges, GDL penalizes it heavily.

    This directly fights the blurriness that L1 loss alone produces.
    """

    def __init__(self):
        super().__init__()

    def _compute_gradient(self, volume):
        """
        Compute spatial gradients using finite differences.

        For each direction (x, y, z), the gradient at position i is:
          grad[i] = volume[i+1] - volume[i]

        This gives you an "edge map" in each direction.
        """
        # Gradient in depth (D) direction
        grad_d = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
        # Gradient in height (H) direction
        grad_h = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
        # Gradient in width (W) direction
        grad_w = volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]

        return grad_d, grad_h, grad_w

    def forward(self, pred, target):
        # Get gradients of predicted and target
        pred_d, pred_h, pred_w = self._compute_gradient(pred)
        tgt_d, tgt_h, tgt_w = self._compute_gradient(target)

        # L1 difference between gradients in all 3 directions
        loss_d = F.l1_loss(pred_d, tgt_d)
        loss_h = F.l1_loss(pred_h, tgt_h)
        loss_w = F.l1_loss(pred_w, tgt_w)

        return (loss_d + loss_h + loss_w) / 3.0


# ==============================================================
# Loss 3: Perceptual Loss (MedicalNet 3D ResNet-50)
# ==============================================================
# Replaces VGG-16 2D perceptual loss.
#
# Why MedicalNet instead of VGG?
#   - VGG is 2D: forces slicing 3D volumes → loses z-axis consistency
#   - VGG trained on ImageNet: features capture "fur", "bricks" not tissue
#   - MedicalNet is native 3D: single forward pass on (B,1,D,H,W)
#   - MedicalNet trained on 23 medical datasets (8 MRI incl. brain)
#   - Features capture anatomical boundaries (GM/WM/CSF), not objects
#
# Architecture: 3D ResNet-50 (Bottleneck) — frozen feature extractor
# Weights: pretrained/resnet_50_23dataset.pth
# Source: https://github.com/Tencent/MedicalNet
# ==============================================================


class _Bottleneck3D(nn.Module):
    """3D Bottleneck residual block (matches MedicalNet architecture)."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_ch)
        self.conv3 = nn.Conv3d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _MedicalNet3DResNet50(nn.Module):
    """3D ResNet-50 matching MedicalNet's architecture exactly.
    Used purely as a frozen feature extractor — no classification head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # [3, 4, 6, 3] bottleneck blocks — standard ResNet-50
        self.layer1 = self._make_layer(64,   64,  3)
        self.layer2 = self._make_layer(256,  128, 4, stride=2)
        self.layer3 = self._make_layer(512,  256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

    def _make_layer(self, in_ch, mid_ch, num_blocks, stride=1):
        out_ch = mid_ch * _Bottleneck3D.expansion
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [_Bottleneck3D(in_ch, mid_ch, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(_Bottleneck3D(out_ch, mid_ch))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        """Forward pass returning intermediate features from layer1-4.

        For a 96³ input:
          layer1 → (B,  256, 24, 24, 24)  — low-level: edges, tissue boundaries
          layer2 → (B,  512, 12, 12, 12)  — mid-level: local anatomy, contrast
          layer3 → (B, 1024,  6,  6,  6)  — high-level: regional structure
          layer4 → (B, 2048,  3,  3,  3)  — semantic: global anatomy
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class PerceptualLoss(nn.Module):
    """
    3D Perceptual Loss using MedicalNet ResNet-50.

    Pretrained on 23 medical segmentation datasets (8 MRI including
    brain, 15 CT — ~110K annotated volumes). Features are anatomically
    relevant for brain MRI (GM/WM/CSF boundaries, cortical folding).

    Key advantages over VGG-16:
      - Native 3D: single forward pass on (B, 1, D, H, W) — no slicing
      - No channel replication: accepts 1-channel MRI directly
      - Z-axis consistency: 3D convolutions preserve depth continuity
      - Medical features: trained on brain MRI, not ImageNet

    Loss = Σ w_l * L1(φ_l(pred), φ_l(target))
    where φ_l = frozen MedicalNet features at layer l.

    Layer weights [1.0, 1.0, 0.5, 0.25] downweight deeper layers
    where CT-specific features may reside (conservative for MRI-only).
    """

    def __init__(self, weights_path=None, device="cuda",
                 layer_weights=(1.0, 1.0, 0.5, 0.25)):
        super().__init__()
        self.net = _MedicalNet3DResNet50()
        self.layer_weights = list(layer_weights)

        # Load MedicalNet pretrained weights
        if weights_path is not None and os.path.isfile(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            # Clean key names: remove "module." prefix, skip segmentation head
            cleaned = {}
            for k, v in state.items():
                k_clean = k.replace("module.", "")
                if k_clean.startswith(("conv_seg", "fc")):
                    continue
                cleaned[k_clean] = v
            missing, unexpected = self.net.load_state_dict(cleaned, strict=False)
            print(f"[MedicalNet3D] Loaded {len(cleaned)} keys from {weights_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            msg = weights_path or "None"
            print(f"[MedicalNet3D] WARNING: No weights loaded (path={msg})")
            print(f"  Download from: https://github.com/Tencent/MedicalNet")
            print(f"  Expected at: ./pretrained/resnet_50_23dataset.pth")

        # Freeze all weights — feature extractor only, not trainable
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
        self.net.to(device)

    def forward(self, pred, target):
        """
        Compute 3D perceptual loss.

        Args:
            pred:   [B, 1, D, H, W] — generator output
            target: [B, 1, D, H, W] — ground truth

        Returns:
            Scalar perceptual loss (weighted L1 across 4 feature layers)
        """
        with torch.no_grad():
            tgt_feats = self.net.extract_features(target)
        pred_feats = self.net.extract_features(pred)

        loss = 0.0
        for w, pf, tf in zip(self.layer_weights, pred_feats, tgt_feats):
            loss = loss + w * F.l1_loss(pf, tf.detach())
        return loss

    def train(self, mode=True):
        """Override to keep MedicalNet always in eval mode."""
        super().train(mode)
        self.net.eval()
        return self


# ==============================================================
# Combined Stage 1 Loss
# ==============================================================

class Stage1Loss(nn.Module):
    """
    Combined loss for Stage 1 training.

    L_S1 = λ_L1 * L1 + λ_GDL * GDL + λ_perc * MedicalNet3D_Perceptual

    All three work together:
      - L1:         "make every voxel close to the real value"
      - GDL:        "make the edges as sharp as the real edges"
      - Perceptual: "make the 3D features look anatomically similar" (MedicalNet)
    """

    def __init__(self, lambda_l1=1.0, lambda_gdl=0.5, lambda_perceptual=0.1,
                 device="cuda", medicalnet_weights=None):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_gdl = lambda_gdl
        self.lambda_perceptual = lambda_perceptual

        self.l1_loss = L1Loss()
        self.gdl_loss = GradientDifferenceLoss()

        # MedicalNet 3D perceptual loss (set lambda=0 to disable)
        if lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(
                weights_path=medicalnet_weights, device=device
            )
        else:
            self.perceptual_loss = None

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, D, H, W] - Generator output (enhanced image)
            target: [B, C, D, H, W] - Ground truth high-res image

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary with individual loss values (for logging)
        """
        # L1 loss
        loss_l1 = self.l1_loss(pred, target)

        # Gradient Difference Loss
        loss_gdl = self.gdl_loss(pred, target)

        # Perceptual Loss
        if self.perceptual_loss is not None and self.lambda_perceptual > 0:
            loss_perceptual = self.perceptual_loss(pred, target)
        else:
            loss_perceptual = torch.tensor(0.0, device=pred.device)

        # Combine
        total = (
            self.lambda_l1 * loss_l1
            + self.lambda_gdl * loss_gdl
            + self.lambda_perceptual * loss_perceptual
        )

        loss_dict = {
            "l1": loss_l1.item(),
            "gdl": loss_gdl.item(),
            "perceptual": loss_perceptual.item(),
            "total": total.item(),
        }

        return total, loss_dict


# ==============================================================
# Quick test
# ==============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = Stage1Loss(
        lambda_l1=1.0,
        lambda_gdl=0.5,
        lambda_perceptual=0.1,
        device=str(device),
    )

    pred = torch.randn(1, 1, 64, 64, 64).to(device)
    target = torch.randn(1, 1, 64, 64, 64).to(device)

    total, losses = criterion(pred, target)
    print(f"Total loss: {total.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
