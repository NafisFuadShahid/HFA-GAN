"""
Stage 2 Networks — HFA-GAN.
Generator: 3D Dual-Attention Residual U-Net (same as Stage 1).
Discriminator: 3D PatchGAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== Building Blocks ======================

class ResidualBlock(nn.Module):
    def __init__(self, channels, leaky_slope=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(leaky_slope, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SelfAttention3D(nn.Module):
    """Self-attention for long-range dependencies. Used at deep encoder levels."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        inner = max(channels // reduction, 1)
        self.query = nn.Conv3d(channels, inner, 1)
        self.key = nn.Conv3d(channels, inner, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        q = self.query(x).view(B, -1, N)
        k = self.key(x).view(B, -1, N)
        v = self.value(x).view(B, -1, N)
        attn = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, D, H, W)
        return self.gamma * out + x


class DownBlock(nn.Module):
    """Encoder: downsample + residual + optional attention."""
    def __init__(self, in_ch, out_ch, leaky_slope=0.2, use_attention=False):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.res = ResidualBlock(out_ch, leaky_slope)
        self.attn = SelfAttention3D(out_ch) if use_attention else None

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        if self.attn is not None:
            x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Decoder: upsample + concat skip + conv + residual."""
    def __init__(self, in_ch, skip_ch, out_ch, leaky_slope=0.2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.res = ResidualBlock(out_ch, leaky_slope)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.res(x)
        return x


# ====================== Generator ======================

class Generator3D(nn.Module):
    """
    3D Dual-Attention Residual U-Net.
    
    Channels: 1 -> 32 -> 64 -> 128 -> 256 -> 512 (bottleneck)
    Spatial:  128 -> 64 -> 32 -> 16 -> 8 (bottleneck)
    Global residual: output = input + G(input)
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        init_features=32,
        encoder_channels=(64, 128, 256, 512),
        attention_levels=(2, 3),
        attention_bottleneck=True,
        leaky_slope=0.2,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_features, 7, padding=3, bias=False),
            nn.InstanceNorm3d(init_features, affine=True),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # Encoder
        enc_in = [init_features] + list(encoder_channels[:-1])
        self.encoders = nn.ModuleList([
            DownBlock(enc_in[i], encoder_channels[i], leaky_slope,
                      use_attention=(i in attention_levels))
            for i in range(len(encoder_channels))
        ])

        # Bottleneck
        bneck_ch = encoder_channels[-1]
        self.bottleneck = ResidualBlock(bneck_ch, leaky_slope)
        self.bottleneck_attn = SelfAttention3D(bneck_ch) if attention_bottleneck else None

        # Decoder — mirror of encoder
        # Decoder level 0 (deepest): input=512(bottleneck), skip=512(enc3), out=256
        # Decoder level 1: input=256, skip=256(enc2), out=128
        # Decoder level 2: input=128, skip=128(enc1), out=64
        # Decoder level 3: input=64, skip=64(enc0), out=32
        enc_ch = list(encoder_channels)  # [64, 128, 256, 512]
        self.decoders = nn.ModuleList()
        for i in range(len(encoder_channels)):
            rev_i = len(encoder_channels) - 1 - i  # 3, 2, 1, 0
            d_in = bneck_ch if i == 0 else enc_ch[rev_i + 1]
            d_skip = enc_ch[rev_i]
            d_out = enc_ch[rev_i - 1] if rev_i > 0 else init_features
            self.decoders.append(UpBlock(d_in, d_skip, d_out, leaky_slope))

        # Output head
        self.head = nn.Sequential(
            nn.Conv3d(init_features, out_channels, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        s0 = self.stem(x)
        
        skips = []
        h = s0
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        h = self.bottleneck(h)
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)

        for i, dec in enumerate(self.decoders):
            skip_idx = len(skips) - 1 - i
            h = dec(h, skips[skip_idx])

        residual = self.head(h)
        return x + residual  # Global residual connection


# ====================== Discriminator ======================

class PatchGAN3D(nn.Module):
    """
    3D PatchGAN Discriminator.
    Outputs a grid of real/fake predictions (not a single value).
    No sigmoid — uses LSGAN (MSE) loss.
    """
    def __init__(self, in_channels=1, base_features=64, n_layers=3, leaky_slope=0.2):
        super().__init__()
        layers = []

        # First layer: no normalization
        layers += [nn.Conv3d(in_channels, base_features, 4, stride=2, padding=1),
                   nn.LeakyReLU(leaky_slope, inplace=True)]

        # Middle layers
        ch_in = base_features
        for i in range(1, n_layers):
            ch_out = min(ch_in * 2, 512)
            layers += [nn.Conv3d(ch_in, ch_out, 4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm3d(ch_out, affine=True),
                       nn.LeakyReLU(leaky_slope, inplace=True)]
            ch_in = ch_out

        # Penultimate: stride 1
        ch_out = min(ch_in * 2, 512)
        layers += [nn.Conv3d(ch_in, ch_out, 4, stride=1, padding=1, bias=False),
                   nn.InstanceNorm3d(ch_out, affine=True),
                   nn.LeakyReLU(leaky_slope, inplace=True)]

        # Output: 1 channel per patch
        layers += [nn.Conv3d(ch_out, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ====================== Utilities ======================

class ImageBuffer:
    """History buffer for discriminator stability (Shrivastava et al., 2017)."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def query(self, images):
        if self.max_size == 0:
            return images
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img.clone().detach())
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.max_size, (1,)).item()
                    old = self.buffer[idx].clone()
                    self.buffer[idx] = img.clone().detach()
                    result.append(old)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


def load_stage1_weights(generator, checkpoint_path, device="cpu"):
    """Load Stage 1 weights into G_XtoY. Handles various checkpoint formats."""
    print(f"Loading Stage 1 weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "generator_state_dict", "state_dict", "g_state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean keys: remove 'module.' (DataParallel) and 'generator.' (wrapper) prefixes
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        if key.startswith("generator."):
            key = key[len("generator."):]
        cleaned[key] = v

    missing, unexpected = generator.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning — missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Warning — unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("  All weights loaded successfully!")
    return generator
