"""
Pre-compute Stage 1 predictions (needed for Frequency Consistency Loss).

Usage:
    python precompute_stage1.py --stage1_checkpoint stage1.pth --input_dir data/train/64mT --output_dir data/train/stage1_pred
"""
import argparse, torch, nibabel as nib, numpy as np
from pathlib import Path
from tqdm import tqdm
from networks import Generator3D, load_stage1_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--in_channels", type=int, default=1)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator3D(in_channels=a.in_channels, out_channels=a.in_channels).to(device)
    g = load_stage1_weights(g, a.stage1_checkpoint, device)
    g.eval()

    inp = Path(a.input_dir)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(inp.glob("*.nii.gz")) + list(inp.glob("*.nii")))
    print(f"Processing {len(files)} volumes...")

    for f in tqdm(files):
        nii = nib.load(str(f))
        data = nii.get_fdata().astype(np.float32)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8: data = (data - dmin) / (dmax - dmin)
        data = data * 2.0 - 1.0
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): pred = g(t)
        nib.save(nib.Nifti1Image(pred.squeeze().cpu().numpy(), nii.affine, nii.header), str(out/f.name))

    print(f"Done. Saved to {out}")

if __name__ == "__main__": main()
