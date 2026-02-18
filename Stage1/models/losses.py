"""
HFA-GAN Stage 1 Loss Functions
================================
Three losses combined:

1. L1 Loss         - Pixel-wise accuracy (every voxel close to ground truth)
2. GDL             - Gradient Difference Loss (edges must be sharp)
3. Perceptual Loss - VGG feature similarity (structures must look correct)

Combined: L_S1 = λ_L1 * L1 + λ_GDL * GDL + λ_VGG * Perceptual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


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
# Loss 3: Perceptual Loss (VGG-16)
# ==============================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained VGG-16.

    The idea: instead of comparing pixels, compare how images "look"
    at a higher level. VGG-16 was trained on millions of images to
    recognize objects. Its internal features capture edges, textures,
    and shapes - things that matter perceptually.

    How it works:
      1. Pass both predicted and target through VGG-16
      2. Extract features from intermediate layers
      3. Compare features with L2 loss

    The 3D problem:
      VGG-16 only works on 2D images. Our data is 3D.
      Solution: extract 2D slices from all 3 anatomical planes
      (axial, sagittal, coronal), compute perceptual loss on each,
      and average.

    Note: VGG expects 3-channel (RGB) input. For single-channel MRI,
    we repeat the channel 3 times.
    """

    def __init__(self, feature_layers=None, device="cuda"):
        super().__init__()

        # Load pre-trained VGG-16 (frozen - we don't train it)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()

        # Which layers to extract features from
        # These correspond to different levels of abstraction:
        #   Layer 4  = early features (edges, corners)
        #   Layer 9  = mid features (textures, patterns)
        #   Layer 16 = deep features (shapes, parts)
        if feature_layers is None:
            feature_layers = [4, 9, 16]
        self.feature_layers = feature_layers

        # Split VGG into sub-networks at each extraction point
        self.blocks = nn.ModuleList()
        prev = 0
        for layer_idx in feature_layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer_idx]))
            prev = layer_idx

        # Freeze all VGG weights
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

        # VGG expects images normalized with ImageNet stats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_for_vgg(self, x):
        """
        Convert from our [-1, 1] range to VGG's expected range.
        Also convert 1-channel to 3-channel.
        """
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0

        # 1-channel -> 3-channel (repeat grayscale)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Apply ImageNet normalization
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def _extract_features(self, x):
        """Extract VGG features at specified layers."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _perceptual_2d(self, pred_2d, target_2d):
        """Compute perceptual loss on a batch of 2D slices."""
        pred_norm = self._normalize_for_vgg(pred_2d)
        tgt_norm = self._normalize_for_vgg(target_2d)

        pred_feats = self._extract_features(pred_norm)
        tgt_feats = self._extract_features(tgt_norm)

        loss = 0.0
        for pf, tf in zip(pred_feats, tgt_feats):
            loss += F.mse_loss(pf, tf)

        return loss / len(pred_feats)

    def forward(self, pred, target, num_slices=8):
        """
        Compute perceptual loss on 3D volumes by sampling 2D slices
        from axial, sagittal, and coronal planes.

        Args:
            pred: [B, C, D, H, W] predicted volume
            target: [B, C, D, H, W] ground truth volume
            num_slices: how many slices to sample per plane

        Returns:
            Average perceptual loss across all planes and slices.
        """
        B, C, D, H, W = pred.shape
        total_loss = 0.0
        count = 0

        # Sample evenly-spaced slice indices
        for plane in ["axial", "sagittal", "coronal"]:
            if plane == "axial":
                max_idx = D
                get_slice = lambda vol, i: vol[:, :, i, :, :]  # [B, C, H, W]
            elif plane == "sagittal":
                max_idx = W
                get_slice = lambda vol, i: vol[:, :, :, :, i]  # [B, C, D, H]
            elif plane == "coronal":
                max_idx = H
                get_slice = lambda vol, i: vol[:, :, :, i, :]  # [B, C, D, W]

            # Sample slice indices (evenly spaced, skip edges)
            margin = max_idx // 8
            indices = torch.linspace(margin, max_idx - margin - 1, num_slices).long()

            for idx in indices:
                pred_slice = get_slice(pred, idx)     # [B, C, ?, ?]
                tgt_slice = get_slice(target, idx)     # [B, C, ?, ?]

                # Resize to VGG's expected input size (224x224)
                pred_resized = F.interpolate(pred_slice, size=(224, 224), mode="bilinear", align_corners=False)
                tgt_resized = F.interpolate(tgt_slice, size=(224, 224), mode="bilinear", align_corners=False)

                total_loss += self._perceptual_2d(pred_resized, tgt_resized)
                count += 1

        return total_loss / max(count, 1)


# ==============================================================
# Combined Stage 1 Loss
# ==============================================================

class Stage1Loss(nn.Module):
    """
    Combined loss for Stage 1 training.

    L_S1 = λ_L1 * L1 + λ_GDL * GDL + λ_VGG * Perceptual

    All three work together:
      - L1:         "make every voxel close to the real value"
      - GDL:        "make the edges as sharp as the real edges"
      - Perceptual: "make the features look structurally similar"
    """

    def __init__(self, lambda_l1=1.0, lambda_gdl=0.5, lambda_perceptual=0.1,
                 device="cuda"):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_gdl = lambda_gdl
        self.lambda_perceptual = lambda_perceptual

        self.l1_loss = L1Loss()
        self.gdl_loss = GradientDifferenceLoss()

        # Perceptual loss is optional (set lambda=0 to disable)
        if lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
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
