#utils/inference.py
import importlib
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from models.tsr_cnn import TSRNet
from models.autoencoder import Autoencoder, load_checkpoint
from models.unet_watermark import UNetWatermark
from models.latent_decoder import LatentDecoder
from utils.screen_shooting import ScreenShooting 
import utils.metrics as metrics
importlib.reload(metrics)
from utils.metrics import psnr, ssim_batch, lpips_batch


@torch.no_grad()
def evaluate_unet_watermark(test_loader, device,
                            unet_ckpt="results/checkpoints/unet/unet_best.pth",
                            latent_decoder_ckpt="results/checkpoints/unet/latent_decoder_best.pth",
                            autoencoder_ckpt="results/checkpoints/autoencoder/autoencoder_best.pth",
                            tsr_ckpt="results/checkpoints/tsr/tsr_best_model.pth",
                            latent_dim=512,
                            save_dir="results/eval_samples",
                            shooting_prob=0.8):

    os.makedirs(save_dir, exist_ok=True)

    tsr_model = TSRNet(num_classes=43).to(device)
    ckpt = torch.load(tsr_ckpt, map_location=device)
    tsr_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    tsr_model.eval()

    ae = Autoencoder(latent_dim=latent_dim).to(device)
    load_checkpoint(ae, None, autoencoder_ckpt, map_location=device)
    ae.eval()

    unet = UNetWatermark(latent_dim=latent_dim).to(device)
    unet_ckpt_data = torch.load(unet_ckpt, map_location=device)
    unet.load_state_dict(unet_ckpt_data["model_state"])
    unet.eval()

    latent_decoder = LatentDecoder(latent_dim=latent_dim).to(device)
    latent_ckpt_data = torch.load(latent_decoder_ckpt, map_location=device)
    latent_decoder.load_state_dict(latent_ckpt_data["model_state"])
    latent_decoder.eval()

    screen = ScreenShooting(apply_prob=shooting_prob).to(device)

    correct_orig = correct_wm = correct_dist = correct_recon = 0
    total = count = 0

 
    psnr_iw_total = ssim_iw_total = lpips_iw_total = 0.0
    psnr_rec_total = ssim_rec_total = lpips_rec_total = 0.0


    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        total += labels.size(0)

        outputs_orig = tsr_model(imgs)
        _, preds_orig = torch.max(outputs_orig, 1)
        correct_orig += (preds_orig == labels).sum().item()

        z = ae.encoder(imgs)

        Iw = unet(imgs, z)

        outputs_wm = tsr_model(Iw)
        _, preds_wm = torch.max(outputs_wm, 1)
        correct_wm += (preds_wm == labels).sum().item()

        Iw_shot = screen(Iw)
        outputs_dist = tsr_model(Iw_shot)
        _, preds_dist = torch.max(outputs_dist, 1)
        correct_dist += (preds_dist == labels).sum().item()

        z_pred = latent_decoder(Iw_shot)
        recon = ae.decoder(z_pred)
        recon = torch.clamp(recon, -1, 1)

        outputs_recon = tsr_model(recon)
        _, preds_recon = torch.max(outputs_recon, 1)
        correct_recon += (preds_recon == labels).sum().item()

        psnr_iw_total += psnr(imgs, Iw)
        ssim_iw_total += ssim_batch(imgs, Iw)
        lpips_iw_total += lpips_batch(imgs, Iw, device)

        psnr_rec_total += psnr(imgs, recon)
        ssim_rec_total += ssim_batch(imgs, recon)
        lpips_rec_total += lpips_batch(imgs, recon, device)

        count += 1

    acc_orig = 100 * correct_orig / total
    acc_wm = 100 * correct_wm / total
    acc_dist = 100 * correct_dist / total
    acc_recon = 100 * correct_recon / total

    avg_psnr_iw = psnr_iw_total / count
    avg_ssim_iw = ssim_iw_total / count
    avg_lpips_iw = lpips_iw_total / count

    avg_psnr_rec = psnr_rec_total / count
    avg_ssim_rec = ssim_rec_total / count
    avg_lpips_rec = lpips_rec_total / count

    print("\n TSR Classification Accuracy:")
    print(f"   Original Images:      {acc_orig:.2f}%")
    print(f"   Watermarked Images:   {acc_wm:.2f}%")
    print(f"   Distorted Images:     {acc_dist:.2f}%")
    print(f"   Reconstructed Images: {acc_recon:.2f}%")

    print("\n Image Quality Metrics:")
    print(f"Between Original ↔ Watermarked:")
    print(f"   PSNR:  {avg_psnr_iw:.2f} dB")
    print(f"   SSIM:  {avg_ssim_iw:.4f}")
    print(f"   LPIPS: {avg_lpips_iw:.4f}")

    print(f"\nBetween Original ↔ Reconstructed:")
    print(f"   PSNR:  {avg_psnr_rec:.2f} dB")
    print(f"   SSIM:  {avg_ssim_rec:.4f}")
    print(f"   LPIPS: {avg_lpips_rec:.4f}")

    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(device)[:8]

    z = ae.encoder(imgs)

    Iw = unet(imgs, z)
    Iw_shot = screen(Iw)
    z_pred = latent_decoder(Iw_shot)
    recon = ae.decoder(z_pred)

    imgs_vis = (imgs + 1) / 2
    Iw_vis = (Iw + 1) / 2
    Iw_shot_vis = (Iw_shot + 1) / 2
    recon_vis = (recon + 1) / 2

    grid = torch.cat([imgs_vis, Iw_vis, Iw_shot_vis, recon_vis], dim=0)
    save_path = os.path.join(save_dir, "original_watermarked_distorted_reconstructed.png")
    save_image(grid, save_path, nrow=8)
    print(f"\n Saved comparison grid at: {save_path}")

    grid_img = make_grid(grid, nrow=8)
    plt.figure(figsize=(16, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("Top: Original | 2nd: Watermarked | 3rd: Distorted | Bottom: Reconstructed", fontsize=16)
    plt.show()
