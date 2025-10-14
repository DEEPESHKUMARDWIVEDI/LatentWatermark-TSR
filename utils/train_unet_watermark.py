# utils/train_unet_watermark.py
import os
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from models.unet_watermark import UNetWatermark
from models.autoencoder import Autoencoder, load_checkpoint
from models.latent_decoder import LatentDecoder  # ✅ newly imported
from utils.perceptual import VGGPerceptualLoss


def train_unet_watermark(train_loader, device,
                         autoencoder_ckpt="results/checkpoints/autoencoder/autoencoder_best.pth",
                         latent_dim=128,
                         num_epochs=10,
                         lr=1e-4,
                         save_watermarked_dir="results/watermarked",
                         ckpt_dir="results/checkpoints/unet"):

    os.makedirs(save_watermarked_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load pretrained autoencoder and freeze
    ae = Autoencoder(latent_dim=latent_dim).to(device)
    load_checkpoint(ae, None, autoencoder_ckpt, map_location=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Initialize UNet and LatentDecoder
    unet = UNetWatermark(latent_dim=latent_dim).to(device)
    latent_decoder = LatentDecoder(latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    optimizer1 = optim.Adam(latent_decoder.parameters(), lr=lr)

    # Perceptual loss
    vgg_loss = VGGPerceptualLoss(device=device)

    best_loss = float("inf")
    for epoch in range(num_epochs):
        unet.train()
        latent_decoder.train()
        running_loss = 0.0
        total = 0

        for imgs, _ in tqdm(train_loader, desc=f"Unet Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            with torch.no_grad():
                z, _ = ae(imgs)

            Iw = unet(imgs, z)
            z_pred = latent_decoder(Iw)

            # Combined loss: perceptual + latent consistency
            loss = vgg_loss(imgs, Iw) + nn.MSELoss()(z_pred, z)

            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer1.step()

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        avg_loss = running_loss / total
        print(f"[Epoch {epoch+1}] Perceptual Loss: {avg_loss:.6f}")

        # Save visual samples
        unet.eval()
        with torch.no_grad():
            sample_imgs, _ = next(iter(train_loader))
            sample_imgs = sample_imgs.to(device)[:16]
            sample_z, _ = ae(sample_imgs)
            sample_Iw = unet(sample_imgs, sample_z)
            save_image((sample_Iw + 1.0)/2.0,
                       os.path.join(save_watermarked_dir, f"epoch{epoch+1}_watermarked.png"),
                       nrow=4)
            torch.save(sample_z.cpu(),
                       os.path.join(save_watermarked_dir, f"epoch{epoch+1}_latents.pt"))

        # Save checkpoints
        ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": unet.state_dict(),
            "loss": avg_loss
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, "unet_best.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state": unet.state_dict(),
                "loss": avg_loss
            }, best_path)
            print(" Saved best UNet checkpoint.")

    return unet, best_loss
