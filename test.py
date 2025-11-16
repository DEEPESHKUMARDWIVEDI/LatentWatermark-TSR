import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.gtsrb_dataset import GTSRBDataset
from models.tsr_cnn import TSRNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm 
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Datasets & loaders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


train_dataset = GTSRBDataset(csv_file="./data/Train.csv", root_dir="./data/", transform=transform_train)
test_dataset  = GTSRBDataset(csv_file="./data/Test.csv",  root_dir="./data/", transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from models.tsr_cnn import TSRNet
from models.autoencoder import Autoencoder, load_checkpoint
from models.unet_watermark import UNetWatermark
from models.latent_decoder import LatentDecoder
from utils.screen_shooting import ScreenShooting 
import importlib
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
                            shooting_prob=1):
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------
    # 1. Load Models
    # -------------------------------
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

    # -------------------------------
    # 2. Evaluation Metrics
    # -------------------------------
    correct, total = 0, 0
    psnr_total, ssim_total, lpips_total, count = 0.0, 0.0, 0.0, 0
 

    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)

        # Encode latent
        z, _ = ae(imgs)

        # Watermark embedding
        Iw = unet(imgs, z)

        # Apply screen shooting distortion

        Iw_shot = screen(Iw)

        # Decode latent and reconstruct
        z_pred = latent_decoder(Iw_shot)
        recon = ae.decoder(z_pred)
        recon = torch.clamp(recon, -1, 1)

        # TSR classification accuracy
        outputs = tsr_model(recon)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # ---- Compute perceptual metrics ----
        psnr_total += psnr(imgs, recon)
        ssim_total += ssim_batch(imgs, recon)
        lpips_total += lpips_batch(imgs, recon, device)
        count += 1

    # Average metrics
    acc = 100 * correct / total
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    avg_lpips = lpips_total / count

    print(f"\n✅ TSR Accuracy: {acc:.2f}%")
    print(f"🔹 PSNR: {avg_psnr:.2f} dB")
    print(f"🔹 SSIM: {avg_ssim:.4f}")
    print(f"🔹 LPIPS: {avg_lpips:.4f}")

    # for imgs, labels in tqdm(test_loader, desc="Evaluating"):
    #     imgs, labels = imgs.to(device), labels.to(device)

    #     # Encode latent
    #     z, _ = ae(imgs)

    #     # Watermark embedding
    #     Iw = unet(imgs, z)

    #     # ✅ Apply screen shooting effect (robustness training)
    #     Iw_shot = screen(Iw)

    #         # Predict latent from the distorted image
    #     z_pred = latent_decoder(Iw_shot)

    #     # Reconstruct image from predicted latent
    #     recon = ae.decoder(z_pred)
    #     recon = torch.clamp(recon, -1, 1)

    #     # TSR accuracy on reconstructed images
    #     outputs = tsr_model(recon)
    #     _, preds = torch.max(outputs, 1)
    #     correct += (preds == labels).sum().item()
    #     total += labels.size(0)

    #     psnr_total += psnr(imgs.detach(), recon.detach())
    #     count += 1

    # acc = 100 * correct / total
    # avg_psnr = psnr_total / count
    # print(f"\nTSR Accuracy on reconstructed images: {acc:.2f}%")
    # print(f"Average PSNR: {avg_psnr:.2f} dB")

    # -------------------------------
    # 3. Visualization (one batch)
    # -------------------------------
    # imgs, _ = next(iter(test_loader))
    # imgs = imgs.to(device)[:8]
    # z, _ = ae(imgs)
    # Iw = unet(imgs, z)
    # z_pred = latent_decoder(Iw)
    # recon = ae.decoder(z_pred)

    # # Bring to [0,1] range for visualization
    # imgs_vis = (imgs + 1) / 2
    # Iw_vis = (Iw + 1) / 2
    # recon_vis = (recon + 1) / 2

    # grid = torch.cat([imgs_vis, Iw_vis, recon_vis], dim=0)
    # save_path = os.path.join(save_dir, "original_watermarked_reconstructed.png")
    # save_image(grid, save_path, nrow=8)
    # print(f"Saved comparison grid at: {save_path}")

    # # Optional: also show inline if running in notebook
    # grid_img = make_grid(grid, nrow=8)
    # plt.figure(figsize=(16, 6))
    # plt.imshow(grid_img.permute(1, 2, 0).cpu())
    # plt.axis("off")
    # plt.title("Top: Original | Middle: Watermarked | Bottom: Reconstructed")
    # plt.show()



    # -------------------------------
    # 3. Visualization (one batch)
    # -------------------------------
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(device)[:8]

    # Encode latent
    z, _ = ae.en(imgs)

    # Generate watermarked image
    Iw = unet(imgs, z)

    # Apply screen shooting distortion
    Iw_shot = screen(Iw)
    print("Distortion difference:", torch.mean((Iw - Iw_shot) ** 2).item())

    # Decode latent and reconstruct
    z_pred = latent_decoder(Iw_shot)
    recon = ae.decoder(z_pred)

    z_random = torch.randn_like(z_pred)
    recon_random = ae .decoder(z_random)
    print("Random reconstruction difference", torch.mean((recon - recon_random)**2).item())


    #test
    latent_mse = torch.mean((z - z_pred) ** 2).item()
    print("Latent MSE:",latent_mse)

    # Bring to [0,1] range for visualization
    imgs_vis = (imgs + 1) / 2
    Iw_vis = (Iw + 1) / 2
    Iw_shot_vis = (Iw_shot + 1) / 2
    recon_vis = (recon + 1) / 2

    # Concatenate in order: Original → Watermarked → Distorted → Reconstructed
    grid = torch.cat([imgs_vis, Iw_vis, Iw_shot_vis, recon_vis], dim=0)
    #grid = torch.cat([imgs_vis, Iw_vis,  recon_vis], dim=0)

    save_path = os.path.join(save_dir, "original_watermarked_distorted_reconstructed.png")
    save_image(grid, save_path, nrow=8)
    print(f"✅ Saved comparison grid at: {save_path}")

    # Optional: also show inline (if in Jupyter or Colab)
    grid_img = make_grid(grid, nrow=8)
    plt.figure(figsize=(16, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("Top: Original | 2nd: Watermarked | 3rd: Distorted | Bottom: Reconstructed")
    plt.show()


evaluate_unet_watermark(test_loader, device)
