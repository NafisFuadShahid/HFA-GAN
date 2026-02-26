"""
Stage 2 Utilities — Metrics, LR scheduling, logging, checkpointing.
"""
import torch
import torch.nn.functional as F
import numpy as np
import csv
import nibabel as nib
from pathlib import Path


def compute_psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10: return 50.0
    return 10.0 * np.log10(data_range**2 / mse)

def compute_ssim_3d(pred, target):
    p = pred.squeeze().cpu().numpy()
    t = target.squeeze().cpu().numpy()
    try:
        from skimage.metrics import structural_similarity
        D = p.shape[0]
        start, end = int(D*0.1), int(D*0.9)
        vals = [structural_similarity(p[d], t[d], data_range=2.0,
                win_size=min(7, min(p[d].shape)-1)) for d in range(start, end)]
        return float(np.mean(vals))
    except ImportError:
        C1, C2 = (0.01*2)**2, (0.03*2)**2
        mu_x, mu_y = p.mean(), t.mean()
        sig_xy = ((p-mu_x)*(t-mu_y)).mean()
        return float((2*mu_x*mu_y+C1)*(2*sig_xy+C2)/((mu_x**2+mu_y**2+C1)*(p.var()+t.var()+C2)))

def compute_mae(pred, target):
    return F.l1_loss(pred, target).item()

def compute_texture_metrics(pred, target):
    p = pred.squeeze().detach().cpu().numpy()
    t = target.squeeze().detach().cpu().numpy()
    m = {}
    gp = np.sqrt(sum(np.gradient(p, axis=i)**2 for i in range(3)))
    gt = np.sqrt(sum(np.gradient(t, axis=i)**2 for i in range(3)))
    m["grad_mag_ratio"] = float(gp.mean() / (gt.mean()+1e-8))
    from scipy.ndimage import laplace
    m["lap_var_ratio"] = float(laplace(p).var() / (laplace(t).var()+1e-8))
    fp, ft = np.fft.fftn(p), np.fft.fftn(t)
    D,H,W = p.shape
    d,h,w = np.meshgrid(np.fft.fftfreq(D), np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    hf = np.sqrt(d**2+h**2+w**2) > 0.1
    m["hf_energy_ratio"] = float(np.abs(fp[hf]).sum() / (np.abs(ft[hf]).sum()+1e-8))
    return m


class LinearDecayLR:
    """Constant LR then linear decay to 0. Standard for CycleGAN."""
    def __init__(self, optimizer, total_epochs, decay_start):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.decay_start = decay_start
        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.decay_start:
            factor = 1.0
        else:
            factor = max(0.0, (self.total_epochs - epoch) / (self.total_epochs - self.decay_start))
        for pg, lr0 in zip(self.optimizer.param_groups, self.initial_lrs):
            pg["lr"] = lr0 * factor


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        with open(filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames).writeheader()

    def log(self, row):
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, self.fieldnames).writerow(row)


def save_volume_nifti(tensor, filepath, affine=None):
    data = tensor.squeeze().detach().cpu().numpy()
    nib.save(nib.Nifti1Image(data, affine or np.eye(4)), filepath)


def save_checkpoint(epoch, g_xy, g_yx, d_y, d_x, opt_g, opt_d, metrics, filepath):
    torch.save({
        "epoch": epoch,
        "g_xy_state_dict": g_xy.state_dict(),
        "g_yx_state_dict": g_yx.state_dict(),
        "d_y_state_dict": d_y.state_dict(),
        "d_x_state_dict": d_x.state_dict(),
        "opt_g_state_dict": opt_g.state_dict(),
        "opt_d_state_dict": opt_d.state_dict(),
        "metrics": metrics,
    }, filepath)
