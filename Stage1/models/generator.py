"""
HFA-GAN Stage 1 Generator
==========================
3D Dual-Attention Residual U-Net

Architecture:
  - Encoder: 4 levels of downsampling (Conv3d stride 2)
  - Bottleneck: ResBlock + Self-Attention at 8^3
  - Decoder: 4 levels of upsampling (ConvTranspose3d stride 2)
  - Skip connections with Attention Gates
  - Self-Attention at 16^3 and 8^3 levels
  - Global residual: output = input + predicted_residual

Input:  [B, C, 128, 128, 128]  (C=1 for T1 only, C=3 for T1+T2+FLAIR)
Output: [B, C, 128, 128, 128]  (the enhanced image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Building Blocks
# ==============================================================

class ResBlock3D(nn.Module):
    """
    Residual Block: two convolutions with a skip connection.

    input -> Conv -> Norm -> ReLU -> Conv -> Norm -> (+) -> ReLU -> output
      |                                                ^
      +------------------------------------------------+
    """

    def __init__(self, channels, norm_type="instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d

        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            Norm(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            Norm(channels),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class AttentionGate3D(nn.Module):
    """
    Attention Gate for skip connections.

    Takes:
      - g: gating signal from decoder (coarse, has context)
      - x: feature map from encoder (detailed, has noise)

    Returns:
      - x * alpha  (filtered features, only important stuff passes through)

    How it works:
      1. Project both g and x to same channel dim with 1x1x1 conv
      2. Add them + ReLU
      3. Another 1x1x1 conv to get single-channel attention map
      4. Sigmoid to get values between 0 and 1
      5. Multiply with original x
    """

    def __init__(self, gate_channels, feature_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = feature_channels // 2

        # Project gate signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )

        # Project feature map (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv3d(feature_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )

        # Combine to single-channel attention map
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g and x might have different spatial sizes if upsampling isn't exact
        # Interpolate g to match x's spatial size
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)

        g_proj = self.W_g(g)         # [B, inter, D, H, W]
        x_proj = self.W_x(x)         # [B, inter, D, H, W]
        combined = self.relu(g_proj + x_proj)  # [B, inter, D, H, W]
        alpha = self.psi(combined)    # [B, 1, D, H, W]  values in [0, 1]
        return x * alpha              # Filter encoder features


class SelfAttention3D(nn.Module):
    """
    Self-Attention: every voxel attends to every other voxel.

    Only used at small spatial resolutions (8^3, 16^3) because
    the attention matrix is N x N where N = D*H*W.

    Uses the standard Q, K, V mechanism:
      - Q (query): what am I looking for?
      - K (key): what do I contain?
      - V (value): what information should I send?
      - attention = softmax(Q^T * K / sqrt(d)) * V
      - output = gamma * attention + input  (learnable residual)
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        inter_channels = channels // 8  # Reduce dims for efficiency

        self.query = nn.Conv3d(channels, inter_channels, kernel_size=1)
        self.key = nn.Conv3d(channels, inter_channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)

        # Learnable scaling factor, starts at 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W  # Total spatial positions

        # Project to Q, K, V
        q = self.query(x).view(B, -1, N)   # [B, C', N]
        k = self.key(x).view(B, -1, N)     # [B, C', N]
        v = self.value(x).view(B, -1, N)   # [B, C, N]

        # Attention scores: [B, N, N]
        attn = torch.bmm(q.permute(0, 2, 1), k)  # [B, N, N]
        scale = q.shape[1] ** 0.5
        attn = F.softmax(attn / scale, dim=-1)

        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, C, D, H, W)

        # Residual with learnable gamma
        return self.gamma * out + x


# ==============================================================
# Encoder and Decoder Blocks
# ==============================================================

class EncoderBlock(nn.Module):
    """
    One level of the encoder:
      Conv3d(stride=2) to downsample -> Norm -> LeakyReLU -> ResBlock
      Optionally adds self-attention.
    """

    def __init__(self, in_ch, out_ch, norm_type="instance", use_self_attn=False):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d

        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.resblock = ResBlock3D(out_ch, norm_type)
        self.self_attn = SelfAttention3D(out_ch) if use_self_attn else None

    def forward(self, x):
        x = self.down(x)
        x = self.resblock(x)
        if self.self_attn is not None:
            x = self.self_attn(x)
        return x


class DecoderBlock(nn.Module):
    """
    One level of the decoder:
      ConvTranspose3d(stride=2) to upsample -> Norm -> ReLU
      Then concatenate with (attention-gated) skip connection
      Then Conv3d to reduce channels -> Norm -> ReLU
      Optionally adds self-attention.
    """

    def __init__(self, in_ch, skip_ch, out_ch, norm_type="instance",
                 use_attn_gate=True, use_self_attn=False):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d

        # Upsample
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            Norm(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Attention gate for skip connection
        self.attn_gate = AttentionGate3D(out_ch, skip_ch) if use_attn_gate else None

        # After concatenation: out_ch (from upsample) + skip_ch (from encoder)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            Norm(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.resblock = ResBlock3D(out_ch, norm_type)
        self.self_attn = SelfAttention3D(out_ch) if use_self_attn else None

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch from rounding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Apply attention gate to skip connection
        if self.attn_gate is not None:
            skip = self.attn_gate(g=x, x=skip)

        # Concatenate and reduce
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.resblock(x)

        if self.self_attn is not None:
            x = self.self_attn(x)
        return x


# ==============================================================
# The Full Generator
# ==============================================================

class DualAttentionResidualUNet3D(nn.Module):
    """
    3D Dual-Attention Residual U-Net

    "Dual Attention" means:
      1. Attention Gates on ALL skip connections (encoder -> decoder)
      2. Self-Attention at the bottleneck and deep levels (8^3, 16^3)

    "Residual" means:
      - ResBlocks inside each level (local residual)
      - Global residual: final output = input + predicted_delta

    Architecture (for 128^3 input, base_features=32):
      Initial:  [C_in, 128^3] -> Conv7x7 -> [32, 128^3]

      Enc1:     [32, 128^3]  -> stride-2 conv -> [64, 64^3]
      Enc2:     [64, 64^3]   -> stride-2 conv -> [128, 32^3]
      Enc3:     [128, 32^3]  -> stride-2 conv -> [256, 16^3]  + self-attention
      Bottleneck: [256, 16^3] -> stride-2 conv -> [512, 8^3]  + self-attention

      Dec4:     [512, 8^3]  + skip(256) -> [256, 16^3]  + self-attention
      Dec3:     [256, 16^3] + skip(128) -> [128, 32^3]
      Dec2:     [128, 32^3] + skip(64)  -> [64, 64^3]
      Dec1:     [64, 64^3]  + skip(32)  -> [32, 128^3]

      Final:    [32, 128^3] -> Conv7x7 -> [C_out, 128^3] -> Tanh
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_features=32,
        norm_type="instance",
        use_attention_gates=True,
        use_self_attention=True,
    ):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d
        bf = base_features  # shorthand

        # ---- Initial convolution (large kernel for wide receptive field) ----
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, bf, kernel_size=7, stride=1, padding=3, bias=False),
            Norm(bf),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ---- Encoder ----
        # Level 1: 128^3 -> 64^3,  bf -> bf*2
        self.enc1 = EncoderBlock(bf, bf * 2, norm_type, use_self_attn=False)
        # Level 2: 64^3 -> 32^3,   bf*2 -> bf*4
        self.enc2 = EncoderBlock(bf * 2, bf * 4, norm_type, use_self_attn=False)
        # Level 3: 32^3 -> 16^3,   bf*4 -> bf*8
        self.enc3 = EncoderBlock(bf * 4, bf * 8, norm_type,
                                 use_self_attn=use_self_attention)
        # Bottleneck: 16^3 -> 8^3, bf*8 -> bf*16
        self.bottleneck = EncoderBlock(bf * 8, bf * 16, norm_type,
                                       use_self_attn=use_self_attention)

        # ---- Decoder ----
        # Level 4: 8^3 -> 16^3,  bf*16 + skip(bf*8) -> bf*8
        self.dec4 = DecoderBlock(bf * 16, bf * 8, bf * 8, norm_type,
                                 use_attn_gate=use_attention_gates,
                                 use_self_attn=use_self_attention)
        # Level 3: 16^3 -> 32^3, bf*8 + skip(bf*4) -> bf*4
        self.dec3 = DecoderBlock(bf * 8, bf * 4, bf * 4, norm_type,
                                 use_attn_gate=use_attention_gates,
                                 use_self_attn=False)
        # Level 2: 32^3 -> 64^3, bf*4 + skip(bf*2) -> bf*2
        self.dec2 = DecoderBlock(bf * 4, bf * 2, bf * 2, norm_type,
                                 use_attn_gate=use_attention_gates,
                                 use_self_attn=False)
        # Level 1: 64^3 -> 128^3, bf*2 + skip(bf) -> bf
        self.dec1 = DecoderBlock(bf * 2, bf, bf, norm_type,
                                 use_attn_gate=use_attention_gates,
                                 use_self_attn=False)

        # ---- Final convolution ----
        self.final = nn.Sequential(
            nn.Conv3d(bf, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass with GLOBAL RESIDUAL LEARNING.

        The network predicts delta (the difference between LR and HR).
        Final output = input + delta

        Args:
            x: [B, C, D, H, W] - Low-res input normalized to [-1, 1]

        Returns:
            enhanced: [B, C, D, H, W] - Enhanced output in [-1, 1]
        """
        # Save input for global residual
        identity = x

        # Initial conv
        s0 = self.initial(x)     # [B, 32, 128, 128, 128]

        # Encoder path (save features for skip connections)
        s1 = self.enc1(s0)       # [B, 64,  64,  64,  64]
        s2 = self.enc2(s1)       # [B, 128, 32,  32,  32]
        s3 = self.enc3(s2)       # [B, 256, 16,  16,  16]
        bn = self.bottleneck(s3) # [B, 512,  8,   8,   8]

        # Decoder path (with attention-gated skip connections)
        d4 = self.dec4(bn, s3)   # [B, 256, 16,  16,  16]
        d3 = self.dec3(d4, s2)   # [B, 128, 32,  32,  32]
        d2 = self.dec2(d3, s1)   # [B, 64,  64,  64,  64]
        d1 = self.dec1(d2, s0)   # [B, 32, 128, 128, 128]

        # Final conv -> residual (delta)
        delta = self.final(d1)   # [B, C, 128, 128, 128]

        # GLOBAL RESIDUAL: output = input + learned delta
        enhanced = identity + delta

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, -1.0, 1.0)

        return enhanced

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================
# Quick test
# ==============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualAttentionResidualUNet3D(
        in_channels=1,
        out_channels=1,
        base_features=32,
        use_attention_gates=True,
        use_self_attention=True,
    ).to(device)

    print(f"Total parameters: {model.count_parameters():,}")

    # Test with a dummy input
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
