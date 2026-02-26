"""
HFA-GAN Inference — Apply trained Stage 2 model to new 64mT scans.

Usage:
    python inference.py --checkpoint best_model.pth --input_dir new_scans/ --output_dir enhanced/
"""
import argparse, torch, nibabel as nib, numpy as np
from pathlib import Path
from tqdm import tqdm
from networks import Generator3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--in_channels", type=int, default=1)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator3D(in_channels=a.in_channels, out_channels=a.in_channels).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    g.load_state_dict(ckpt["g_xy_state_dict"])
    g.eval()
    print(f"Loaded epoch {ckpt.get('epoch','?')}")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(Path(a.input_dir).glob("*.nii*")))

    for f in tqdm(files, desc="Enhancing"):
        nii = nib.load(str(f))
        data = nii.get_fdata().astype(np.float32)
        dmin, dmax = data.min(), data.max()
        if dmax-dmin > 1e-8: data = (data-dmin)/(dmax-dmin)
        data = data * 2.0 - 1.0
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): enhanced = g(t)
        e = (enhanced.squeeze().cpu().numpy() + 1.0) / 2.0 * (dmax-dmin) + dmin
        nib.save(nib.Nifti1Image(e, nii.affine, nii.header), str(out/f.name))

    print(f"Enhanced {len(files)} volumes → {out}")

if __name__ == "__main__": main()
