"""
Stage 2 Losses — HFA-GAN.
- LSGAN adversarial loss
- Cycle consistency loss
- Identity loss  
- Frequency Consistency Loss (THE KEY CONTRIBUTION)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSGANLoss(nn.Module):
    """Least-Squares GAN loss. Real=1, Fake=0."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss_fn(pred, target)


class FrequencyConsistencyLoss(nn.Module):
    """
    THE KEY CONTRIBUTION — Frequency Consistency Loss.
    
    Constrains Stage 2 output to preserve the low-frequency anatomical
    structure from Stage 1, while allowing high-frequency texture changes.
    
    ℒ_freq = || LowPass(FFT(Ŷ_stage2)) - LowPass(FFT(ŷ_stage1)) ||₁
    
    Low frequencies = brain shape, ventricle position, tissue boundaries
    High frequencies = texture, edges, noise
    
    Stage 2 is FREE to change high frequencies (add sharpness).
    Stage 2 is LOCKED from changing low frequencies (preserve anatomy).
    """
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self._mask_cache = {}

    def _get_low_pass_mask(self, shape, device):
        cache_key = (shape, str(device))
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        D, H, W = shape
        d_freq = torch.fft.fftfreq(D, device=device)
        h_freq = torch.fft.fftfreq(H, device=device)
        w_freq = torch.fft.fftfreq(W, device=device)
        fd, fh, fw = torch.meshgrid(d_freq, h_freq, w_freq, indexing='ij')
        freq_dist = torch.sqrt(fd**2 + fh**2 + fw**2)
        freq_dist_norm = freq_dist / (freq_dist.max() + 1e-8)

        # Smooth sigmoid transition (avoids ringing from hard cutoff)
        mask = torch.sigmoid(20.0 * (self.cutoff_ratio - freq_dist_norm))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, stage2_output, stage1_output):
        spatial = stage2_output.shape[2:]
        fft_s2 = torch.fft.fftn(stage2_output, dim=(-3, -2, -1))
        fft_s1 = torch.fft.fftn(stage1_output, dim=(-3, -2, -1))
        mask = self._get_low_pass_mask(spatial, stage2_output.device)
        return F.l1_loss(torch.abs(fft_s2 * mask), torch.abs(fft_s1 * mask))


class Stage2LossComputer:
    """Computes all Stage 2 losses and returns individual values for logging."""
    
    def __init__(self, lambda_cyc=10.0, lambda_id=5.0, lambda_freq=10.0, freq_cutoff_ratio=0.1):
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        self.lambda_freq = lambda_freq
        self.gan_loss = LSGANLoss()
        self.l1 = nn.L1Loss()
        self.freq_loss = FrequencyConsistencyLoss(cutoff_ratio=freq_cutoff_ratio)

    def compute_generator_loss(
        self, d_y_fake, d_x_fake,
        cycle_x, real_x, cycle_y, real_y,
        idt_y, idt_x,
        stage2_output, stage1_output,
    ):
        # Adversarial: fool both discriminators
        loss_adv = self.gan_loss(d_y_fake, True) + self.gan_loss(d_x_fake, True)
        # Cycle: X -> Y -> X ≈ X, Y -> X -> Y ≈ Y
        loss_cyc = self.l1(cycle_x, real_x) + self.l1(cycle_y, real_y)
        # Identity: G_XtoY(Y) ≈ Y, G_YtoX(X) ≈ X
        loss_idt = self.l1(idt_y, real_y) + self.l1(idt_x, real_x)
        # Frequency: preserve low-freq anatomy from Stage 1
        loss_freq = self.freq_loss(stage2_output, stage1_output.detach())

        total = (loss_adv 
                 + self.lambda_cyc * loss_cyc 
                 + self.lambda_id * loss_idt 
                 + self.lambda_freq * loss_freq)

        return total, {
            "G_total": total.item(), "G_adv": loss_adv.item(),
            "G_cyc": loss_cyc.item(), "G_idt": loss_idt.item(),
            "G_freq": loss_freq.item(),
        }

    def compute_discriminator_loss(self, d_real, d_fake):
        loss = 0.5 * (self.gan_loss(d_real, True) + self.gan_loss(d_fake, False))
        return loss, {"D_total": loss.item()}
