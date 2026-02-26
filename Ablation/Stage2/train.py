"""
HFA-GAN Stage 2 Training — CycleGAN with Frequency Consistency.

Usage:
    python train.py --stage1_checkpoint path/to/stage1.pth --data_dir path/to/data
"""
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from config import Stage2Config
from networks import Generator3D, PatchGAN3D, ImageBuffer, load_stage1_weights
from losses import Stage2LossComputer
from dataset import Stage2UnpairedDataset, Stage2PairedValDataset
from utils import (compute_psnr, compute_ssim_3d, compute_mae, compute_texture_metrics,
                   LinearDecayLR, CSVLogger, save_checkpoint, save_volume_nifti)


def set_requires_grad(nets, requires_grad):
    if not isinstance(nets, list): nets = [nets]
    for net in nets:
        for p in net.parameters(): p.requires_grad = requires_grad


@torch.no_grad()
def validate(g_xy, val_loader, device, epoch, output_dir, num_save=3):
    g_xy.eval()
    psnr_l, ssim_l, mae_l = [], [], []
    tex_l = {}

    for i, batch in enumerate(val_loader):
        x, y = batch["x"].to(device), batch["y"].to(device)
        y_fake = g_xy(x)
        psnr_l.append(compute_psnr(y_fake, y))
        ssim_l.append(compute_ssim_3d(y_fake, y))
        mae_l.append(compute_mae(y_fake, y))
        try:
            for k, v in compute_texture_metrics(y_fake, y).items():
                tex_l.setdefault(k, []).append(v)
        except: pass
        if i < num_save:
            d = Path(output_dir)/"samples"/f"epoch_{epoch:04d}"
            d.mkdir(parents=True, exist_ok=True)
            fn = batch.get("filename", [f"s{i}"])[0].replace(".nii.gz","").replace(".nii","")
            save_volume_nifti(y_fake, str(d/f"{fn}_s2out.nii.gz"))
            save_volume_nifti(y, str(d/f"{fn}_gt.nii.gz"))

    m = {"val_psnr": sum(psnr_l)/len(psnr_l), "val_ssim": sum(ssim_l)/len(ssim_l),
         "val_mae": sum(mae_l)/len(mae_l)}
    for k, v in tex_l.items(): m[f"val_{k}"] = sum(v)/len(v)
    g_xy.train()
    return m


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)

    # ====== Networks ======
    gen_kw = dict(in_channels=cfg.in_channels, out_channels=cfg.in_channels,
                  init_features=cfg.g_init_features, encoder_channels=cfg.g_encoder_channels,
                  attention_levels=cfg.g_attention_levels, attention_bottleneck=cfg.g_attention_bottleneck)

    g_xy = Generator3D(**gen_kw).to(device)
    if cfg.stage1_checkpoint:
        g_xy = load_stage1_weights(g_xy, cfg.stage1_checkpoint, device)
    else:
        print("WARNING: No Stage 1 checkpoint! G_XtoY starts random.")

    g_yx = Generator3D(**gen_kw).to(device)
    d_y = PatchGAN3D(cfg.in_channels, cfg.d_base_features, cfg.d_n_layers).to(device)
    d_x = PatchGAN3D(cfg.in_channels, cfg.d_base_features, cfg.d_n_layers).to(device)

    cp = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"G_XtoY: {cp(g_xy):,}  G_YtoX: {cp(g_yx):,}  D_Y: {cp(d_y):,}  D_X: {cp(d_x):,}")

    # ====== Optimizers ======
    opt_g = torch.optim.Adam(list(g_xy.parameters()) + list(g_yx.parameters()),
                             lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(list(d_y.parameters()) + list(d_x.parameters()),
                             lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    sched_g = LinearDecayLR(opt_g, cfg.num_epochs, cfg.lr_decay_start)
    sched_d = LinearDecayLR(opt_d, cfg.num_epochs, cfg.lr_decay_start)

    # ====== Losses ======
    loss_comp = Stage2LossComputer(cfg.lambda_cyc, cfg.lambda_id, cfg.lambda_freq, cfg.freq_cutoff_ratio)
    print(f"Lambdas — cyc:{cfg.lambda_cyc} id:{cfg.lambda_id} freq:{cfg.lambda_freq} cutoff:{cfg.freq_cutoff_ratio}")

    # ====== Buffers ======
    buf_y = ImageBuffer(cfg.buffer_size)
    buf_x = ImageBuffer(cfg.buffer_size)

    # ====== Data ======
    train_ds = Stage2UnpairedDataset(cfg.data_dir, "train", cfg.volume_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_ds = Stage2PairedValDataset(cfg.data_dir, "val", cfg.volume_size)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # ====== Logging ======
    train_log = CSVLogger(str(Path(cfg.output_dir)/"logs"/"train_log.csv"),
        ["epoch","step","G_total","G_adv","G_cyc","G_idt","G_freq","D_Y","D_X","lr"])
    val_log = CSVLogger(str(Path(cfg.output_dir)/"logs"/"val_log.csv"),
        ["epoch","val_psnr","val_ssim","val_mae","val_grad_mag_ratio","val_lap_var_ratio","val_hf_energy_ratio"])

    # ====== Training ======
    print(f"\n{'='*60}\nStarting training: {cfg.num_epochs} epochs\n{'='*60}\n")
    best_psnr = 0.0
    step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        g_xy.train(); g_yx.train(); d_y.train(); d_x.train()
        ep_loss = {k: 0.0 for k in ["G_total","G_adv","G_cyc","G_idt","G_freq","D_Y","D_X"]}
        n_batch = 0
        t0 = time.time()

        for batch in train_dl:
            real_x = batch["x"].to(device)
            real_y = batch["y"].to(device)
            s1_pred = batch["s1_pred"].to(device) if "s1_pred" in batch else None

            # ---- Generator ----
            fake_y = g_xy(real_x)       # G_XtoY(X) = Ŷ
            fake_x = g_yx(real_y)       # G_YtoX(Y) = X̃
            cycle_x = g_yx(fake_y)      # G_YtoX(Ŷ) ≈ X
            cycle_y = g_xy(fake_x)      # G_XtoY(X̃) ≈ Y
            idt_y = g_xy(real_y)        # G_XtoY(Y) ≈ Y
            idt_x = g_yx(real_x)        # G_YtoX(X) ≈ X

            set_requires_grad([d_y, d_x], False)
            opt_g.zero_grad()

            freq_s1 = s1_pred if s1_pred is not None else fake_y.detach()

            g_loss, g_dict = loss_comp.compute_generator_loss(
                d_y(fake_y), d_x(fake_x),
                cycle_x, real_x, cycle_y, real_y,
                idt_y, idt_x,
                fake_y, freq_s1)
            g_loss.backward()
            opt_g.step()

            # ---- Discriminator ----
            set_requires_grad([d_y, d_x], True)
            opt_d.zero_grad()

            dy_loss, dy_d = loss_comp.compute_discriminator_loss(d_y(real_y), d_y(buf_y.query(fake_y.detach())))
            dx_loss, dx_d = loss_comp.compute_discriminator_loss(d_x(real_x), d_x(buf_x.query(fake_x.detach())))
            (dy_loss + dx_loss).backward()
            opt_d.step()

            # ---- Accumulate ----
            for k in ["G_total","G_adv","G_cyc","G_idt","G_freq"]:
                ep_loss[k] += g_dict[k]
            ep_loss["D_Y"] += dy_d["D_total"]
            ep_loss["D_X"] += dx_d["D_total"]
            n_batch += 1; step += 1

            if step % cfg.log_interval == 0:
                train_log.log({"epoch":epoch, "step":step,
                    "G_total":f"{g_dict['G_total']:.4f}", "G_adv":f"{g_dict['G_adv']:.4f}",
                    "G_cyc":f"{g_dict['G_cyc']:.4f}", "G_idt":f"{g_dict['G_idt']:.4f}",
                    "G_freq":f"{g_dict['G_freq']:.4f}",
                    "D_Y":f"{dy_d['D_total']:.4f}", "D_X":f"{dx_d['D_total']:.4f}",
                    "lr":f"{opt_g.param_groups[0]['lr']:.6f}"})

        sched_g.step(epoch); sched_d.step(epoch)
        a = {k: v/max(n_batch,1) for k,v in ep_loss.items()}
        print(f"Ep {epoch:3d}/{cfg.num_epochs} ({time.time()-t0:.0f}s) | "
              f"G:{a['G_total']:.4f} [adv:{a['G_adv']:.4f} cyc:{a['G_cyc']:.4f} "
              f"idt:{a['G_idt']:.4f} freq:{a['G_freq']:.4f}] | "
              f"D_Y:{a['D_Y']:.4f} D_X:{a['D_X']:.4f} | lr:{opt_g.param_groups[0]['lr']:.6f}")

        # ---- Validation ----
        if epoch % cfg.val_interval == 0 or epoch == 1:
            vm = validate(g_xy, val_dl, device, epoch, cfg.output_dir, cfg.num_val_samples)
            print(f"  Val PSNR:{vm['val_psnr']:.2f} SSIM:{vm['val_ssim']:.4f} MAE:{vm['val_mae']:.4f}")
            for k in ["val_grad_mag_ratio","val_lap_var_ratio","val_hf_energy_ratio"]:
                if k in vm: print(f"       {k}: {vm[k]:.4f}")
            val_log.log({
                "epoch":epoch, "val_psnr":f"{vm['val_psnr']:.4f}", "val_ssim":f"{vm['val_ssim']:.4f}",
                "val_mae":f"{vm['val_mae']:.4f}",
                "val_grad_mag_ratio":f"{vm.get('val_grad_mag_ratio',0):.4f}",
                "val_lap_var_ratio":f"{vm.get('val_lap_var_ratio',0):.4f}",
                "val_hf_energy_ratio":f"{vm.get('val_hf_energy_ratio',0):.4f}"})
            if vm["val_psnr"] > best_psnr:
                best_psnr = vm["val_psnr"]
                save_checkpoint(epoch, g_xy, g_yx, d_y, d_x, opt_g, opt_d, vm,
                                str(Path(cfg.output_dir)/"checkpoints"/"best_model.pth"))
                print(f"  ★ New best PSNR: {best_psnr:.2f}")

        if epoch % cfg.save_interval == 0:
            save_checkpoint(epoch, g_xy, g_yx, d_y, d_x, opt_g, opt_d, {"epoch":epoch},
                            str(Path(cfg.output_dir)/"checkpoints"/f"epoch_{epoch:04d}.pth"))

    print(f"\n=== Done. Best PSNR: {best_psnr:.2f} ===")


def main():
    p = argparse.ArgumentParser("HFA-GAN Stage 2")
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/stage2")
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=5.0)
    p.add_argument("--lambda_freq", type=float, default=10.0)
    p.add_argument("--freq_cutoff", type=float, default=0.1)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = Stage2Config(
        stage1_checkpoint=a.stage1_checkpoint, data_dir=a.data_dir, output_dir=a.output_dir,
        num_epochs=a.num_epochs, batch_size=a.batch_size, lr_g=a.lr_g, lr_d=a.lr_d,
        lambda_cyc=a.lambda_cyc, lambda_id=a.lambda_id, lambda_freq=a.lambda_freq,
        freq_cutoff_ratio=a.freq_cutoff, in_channels=a.in_channels, seed=a.seed)

    print("="*60 + "\nHFA-GAN Stage 2 — CycleGAN + Frequency Consistency\n" + "="*60)
    for k,v in vars(cfg).items(): print(f"  {k}: {v}")
    print("="*60)
    train(cfg)


if __name__ == "__main__":
    main()
