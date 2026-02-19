"""
HFA-GAN Dataset Loader
======================
Loads paired low-res / high-res NIfTI volumes and extracts 3D patches.

Expected directory structure:
  data_dir_lr/  ->  subject001.nii.gz, subject002.nii.gz, ...
  data_dir_hr/  ->  subject001.nii.gz, subject002.nii.gz, ...

The filenames must match between lr and hr directories.
"""

import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


class PairedNIfTIDataset(Dataset):
    """
    Dataset for paired low-res and high-res NIfTI brain MRI volumes.

    For each pair, extracts random 3D patches during training
    or returns the full volume during validation/testing.
    """

    def __init__(
        self,
        lr_paths: list,
        hr_paths: list,
        patch_size: tuple = (128, 128, 128),
        intensity_range: tuple = (-1.0, 1.0),
        patches_per_volume: int = 4,
        is_training: bool = True,
        min_brain_fraction: float = 0.1,
    ):
        """
        Args:
            lr_paths: List of paths to low-res NIfTI files.
            hr_paths: List of paths to matching high-res NIfTI files.
            patch_size: (D, H, W) size of 3D patches to extract.
            intensity_range: Target intensity range for normalization.
            patches_per_volume: How many random patches to extract per volume per epoch.
            is_training: If True, extract random patches. If False, return full volume.
            min_brain_fraction: Minimum fraction of non-zero voxels in a patch (skip empty patches).
        """
        assert len(lr_paths) == len(hr_paths), \
            f"Mismatch: {len(lr_paths)} LR files vs {len(hr_paths)} HR files"

        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.patch_size = patch_size
        self.intensity_range = intensity_range
        self.patches_per_volume = patches_per_volume
        self.is_training = is_training
        self.min_brain_fraction = min_brain_fraction

        print(f"[Dataset] Loaded {len(self.lr_paths)} paired subjects | "
              f"mode={'train' if is_training else 'eval'} | "
              f"patch_size={patch_size}")

    def __len__(self):
        if self.is_training:
            return len(self.lr_paths) * self.patches_per_volume
        else:
            return len(self.lr_paths)

    def _load_nifti(self, path):
        """Load a NIfTI file and return numpy array as float32."""
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return data

    def _normalize(self, volume):
        """
        Z-score normalize, then rescale to target intensity range.
        Ignores background (zero) voxels.
        """
        mask = volume > 0
        if mask.sum() == 0:
            return volume

        # Z-score on brain voxels only
        brain_mean = volume[mask].mean()
        brain_std = volume[mask].std() + 1e-8
        volume[mask] = (volume[mask] - brain_mean) / brain_std

        # Clip outliers (beyond 3 std)
        volume = np.clip(volume, -3.0, 3.0)

        # Rescale to target range
        lo, hi = self.intensity_range
        volume = (volume - (-3.0)) / (3.0 - (-3.0))  # now [0, 1]
        volume = volume * (hi - lo) + lo               # now [lo, hi]

        return volume

    def _extract_random_patch(self, lr_vol, hr_vol):
        """
        Extract a random patch from the same location in both volumes.
        Retries if the patch is mostly empty (background).
        """
        d, h, w = lr_vol.shape
        pd, ph, pw = self.patch_size

        # Make sure volume is large enough
        assert d >= pd and h >= ph and w >= pw, \
            f"Volume {lr_vol.shape} is smaller than patch {self.patch_size}"

        max_attempts = 20
        for _ in range(max_attempts):
            # Random start position
            sd = random.randint(0, d - pd)
            sh = random.randint(0, h - ph)
            sw = random.randint(0, w - pw)

            lr_patch = lr_vol[sd:sd+pd, sh:sh+ph, sw:sw+pw]
            hr_patch = hr_vol[sd:sd+pd, sh:sh+ph, sw:sw+pw]

            # Check if patch has enough brain content
            brain_fraction = (lr_patch > self.intensity_range[0] + 0.01).mean()
            if brain_fraction >= self.min_brain_fraction:
                return lr_patch, hr_patch

        # If all attempts fail, just return the last patch
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        if self.is_training:
            vol_idx = idx // self.patches_per_volume
        else:
            vol_idx = idx

        # Load volumes
        lr_vol = self._load_nifti(self.lr_paths[vol_idx])
        hr_vol = self._load_nifti(self.hr_paths[vol_idx])

        # Normalize
        lr_vol = self._normalize(lr_vol)
        hr_vol = self._normalize(hr_vol)

        if self.is_training:
            # Extract random patch
            lr_patch, hr_patch = self._extract_random_patch(lr_vol, hr_vol)
        else:
            # For validation: center crop or pad to patch size
            lr_patch, hr_patch = self._center_crop_or_pad(lr_vol, hr_vol)

        # Convert to torch tensors: add channel dimension [1, D, H, W]
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)

        return {
            "lr": lr_tensor,       # Low-res input (x)
            "hr": hr_tensor,       # High-res target (y)
            "name": os.path.basename(self.lr_paths[vol_idx]),
        }

    def _center_crop_or_pad(self, lr_vol, hr_vol):
        """
        For validation: center crop to patch_size if volume is larger,
        or pad with zeros if smaller.
        """
        pd, ph, pw = self.patch_size
        result_lr = np.zeros(self.patch_size, dtype=np.float32) + self.intensity_range[0]
        result_hr = np.zeros(self.patch_size, dtype=np.float32) + self.intensity_range[0]

        d, h, w = lr_vol.shape

        # Calculate crop/pad offsets
        sd = max(0, (d - pd) // 2)
        sh = max(0, (h - ph) // 2)
        sw = max(0, (w - pw) // 2)

        # Target offsets (for padding case)
        td = max(0, (pd - d) // 2)
        th = max(0, (ph - h) // 2)
        tw = max(0, (pw - w) // 2)

        # Copy region
        copy_d = min(d, pd)
        copy_h = min(h, ph)
        copy_w = min(w, pw)

        result_lr[td:td+copy_d, th:th+copy_h, tw:tw+copy_w] = \
            lr_vol[sd:sd+copy_d, sh:sh+copy_h, sw:sw+copy_w]
        result_hr[td:td+copy_d, th:th+copy_h, tw:tw+copy_w] = \
            hr_vol[sd:sd+copy_d, sh:sh+copy_h, sw:sw+copy_w]

        return result_lr, result_hr


def _extract_subject_key(filename):
    """
    Extract a subject key from a filename for matching LR<->HR pairs.

    Filenames differ only in the resolution tag:
      LR: CALSNIC2_EDM_C001_T1w08_V1.nii.gz
      HR: CALSNIC2_EDM_C001_T1w10_V1.nii.gz

    We strip the resolution part (T1w08 / T1w10) to get a common key:
      -> CALSNIC2_EDM_C001__V1
    """
    import re
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    # Remove T1wXX pattern (e.g. T1w08, T1w10) to create a resolution-agnostic key
    key = re.sub(r"T1w\d+", "", name)
    return key


def get_file_pairs(lr_dir, hr_dir):
    """
    Match LR and HR files by subject key (handles different filenames).

    LR files contain 'T1w08', HR files contain 'T1w10'. We match them
    by stripping the resolution tag and comparing the remaining subject key.

    Args:
        lr_dir: Directory containing low-res NIfTI files.
        hr_dir: Directory containing high-res NIfTI files.

    Returns:
        lr_paths, hr_paths: Matched lists of file paths (sorted by subject key).
    """
    # Find all NIfTI files
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.nii.gz")))
    if not lr_files:
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.nii")))

    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.nii.gz")))
    if not hr_files:
        hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.nii")))

    # Build lookup: subject_key -> filepath
    lr_map = {_extract_subject_key(os.path.basename(f)): f for f in lr_files}
    hr_map = {_extract_subject_key(os.path.basename(f)): f for f in hr_files}

    # Match by common keys
    common_keys = sorted(set(lr_map.keys()) & set(hr_map.keys()))

    lr_paths = [lr_map[k] for k in common_keys]
    hr_paths = [hr_map[k] for k in common_keys]

    unmatched_lr = set(lr_map.keys()) - set(hr_map.keys())
    unmatched_hr = set(hr_map.keys()) - set(lr_map.keys())
    if unmatched_lr:
        print(f"[Warning] {len(unmatched_lr)} LR files have no matching HR file")
    if unmatched_hr:
        print(f"[Warning] {len(unmatched_hr)} HR files have no matching LR file")

    print(f"[Data] Found {len(lr_paths)} matched pairs in {lr_dir}")
    return lr_paths, hr_paths


def build_dataloaders(config):
    """
    Build train, validation, and test dataloaders from config.

    Expects pre-split directory structure:
      data_root/
        train/  high_field/  low_field/
        val/    high_field/  low_field/
        test/   high_field/  low_field/

    Returns:
        train_loader, val_loader, test_loader
    """
    # Build paths for each split
    splits = {}
    for split_name in ["train", "val", "test"]:
        lr_dir = os.path.join(config.data_root, split_name, config.lr_subfolder)
        hr_dir = os.path.join(config.data_root, split_name, config.hr_subfolder)

        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")
        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")

        lr_paths, hr_paths = get_file_pairs(lr_dir, hr_dir)
        splits[split_name] = (lr_paths, hr_paths)

    train_lr, train_hr = splits["train"]
    val_lr, val_hr = splits["val"]
    test_lr, test_hr = splits["test"]

    # Create datasets
    train_dataset = PairedNIfTIDataset(
        train_lr, train_hr,
        patch_size=config.patch_size,
        intensity_range=config.intensity_range,
        patches_per_volume=4,
        is_training=True,
    )

    val_dataset = PairedNIfTIDataset(
        val_lr, val_hr,
        patch_size=config.patch_size,
        intensity_range=config.intensity_range,
        is_training=False,
    )

    test_dataset = PairedNIfTIDataset(
        test_lr, test_hr,
        patch_size=config.patch_size,
        intensity_range=config.intensity_range,
        is_training=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader
