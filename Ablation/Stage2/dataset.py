"""
Stage 2 Datasets — Unpaired training + Paired validation.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import numpy as np


class Stage2UnpairedDataset(Dataset):
    """
    Unpaired dataset for CycleGAN training.
    
    Directory structure:
      data_dir/train/64mT/         ← 64mT volumes
      data_dir/train/3T/           ← 3T volumes (different subjects OK)
      data_dir/train/stage1_pred/  ← Pre-computed Stage 1 outputs (same names as 64mT/)
    """
    def __init__(self, data_dir, split="train", volume_size=128):
        self.volume_size = volume_size
        base = Path(data_dir) / split
        
        self.x_files = sorted(list((base / "64mT").glob("*.nii*")))
        self.y_files = sorted(list((base / "3T").glob("*.nii*")))
        
        s1_dir = base / "stage1_pred"
        self.s1_files = sorted(list(s1_dir.glob("*.nii*"))) if s1_dir.exists() else None
        
        assert len(self.x_files) > 0, f"No 64mT files in {base / '64mT'}"
        assert len(self.y_files) > 0, f"No 3T files in {base / '3T'}"
        if self.s1_files:
            assert len(self.s1_files) == len(self.x_files), "Stage1 preds must match 64mT count"
        
        print(f"Dataset ({split}): {len(self.x_files)} x 64mT, {len(self.y_files)} x 3T" +
              (f", {len(self.s1_files)} x S1 preds" if self.s1_files else ""))

    def _load(self, path):
        data = nib.load(str(path)).get_fdata().astype(np.float32)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        data = data * 2.0 - 1.0
        data = self._pad_crop(data)
        return torch.from_numpy(data).unsqueeze(0)

    def _pad_crop(self, data):
        t = self.volume_size
        result = np.full((t, t, t), -1.0, dtype=np.float32)
        d, h, w = data.shape[:3]
        sd, sh, sw = max(0,(d-t)//2), max(0,(h-t)//2), max(0,(w-t)//2)
        dd, dh, dw = max(0,(t-d)//2), max(0,(t-h)//2), max(0,(t-w)//2)
        cd, ch, cw = min(d,t), min(h,t), min(w,t)
        result[dd:dd+cd, dh:dh+ch, dw:dw+cw] = data[sd:sd+cd, sh:sh+ch, sw:sw+cw]
        return result

    def __len__(self):
        return max(len(self.x_files), len(self.y_files))

    def __getitem__(self, idx):
        x = self._load(self.x_files[idx % len(self.x_files)])
        y = self._load(self.y_files[idx % len(self.y_files)])
        out = {"x": x, "y": y}
        if self.s1_files:
            out["s1_pred"] = self._load(self.s1_files[idx % len(self.x_files)])
        return out


class Stage2PairedValDataset(Dataset):
    """Paired val dataset for PSNR/SSIM evaluation."""
    def __init__(self, data_dir, split="val", volume_size=128):
        self.volume_size = volume_size
        base = Path(data_dir) / split
        self.x_files = sorted(list((base / "64mT").glob("*.nii*")))
        self.y_files = sorted(list((base / "3T").glob("*.nii*")))
        s1_dir = base / "stage1_pred"
        self.s1_files = sorted(list(s1_dir.glob("*.nii*"))) if s1_dir.exists() else None
        assert len(self.x_files) == len(self.y_files), "Paired data must match"

    def _load(self, path):
        data = nib.load(str(path)).get_fdata().astype(np.float32)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        data = data * 2.0 - 1.0
        t = self.volume_size
        result = np.full((t,t,t), -1.0, dtype=np.float32)
        d,h,w = data.shape[:3]
        sd,sh,sw = max(0,(d-t)//2),max(0,(h-t)//2),max(0,(w-t)//2)
        dd,dh,dw = max(0,(t-d)//2),max(0,(t-h)//2),max(0,(t-w)//2)
        cd,ch,cw = min(d,t),min(h,t),min(w,t)
        result[dd:dd+cd,dh:dh+ch,dw:dw+cw] = data[sd:sd+cd,sh:sh+ch,sw:sw+cw]
        return torch.from_numpy(result).unsqueeze(0)

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        out = {"x": self._load(self.x_files[idx]), "y": self._load(self.y_files[idx]),
               "filename": self.x_files[idx].name}
        if self.s1_files:
            out["s1_pred"] = self._load(self.s1_files[idx])
        return out
