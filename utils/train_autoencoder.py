# utils/train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.autoencoder import Autoencoder, save_checkpoint
import torch.nn.functional as F
import importlib
import utils.perceptual as perceptual
importlib.reload(perceptual)
from utils.perceptual import VGGPerceptualLoss 
from utils import metrics

def train_autoencoder(train_loader, test_loader, device,
                      latent_dim=128, num_epochs=10, batch_size=64,
                      lr=1e-3, ckpt_dir="results/checkpoints/autoencoder"):
    model = Autoencoder(latent_dim=latent_dim).to(device)
    mse_loss = nn.MSELoss()
    vgg_loss = VGGPerceptualLoss(device=device).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_test_loss = float("inf")
    best_ckpt = None

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            z, x_rec = model(imgs)

            # --- Loss Components ---
            loss_recon = mse_loss(x_rec, imgs)
            loss_percep = vgg_loss(x_rec, imgs).mean()
            # cos_loss = 1 - F.cosine_similarity(z, z).mean()  # if you later add latent decoder, use z_pred

            # --- Weighted Total Loss ---
            # loss = 0.6 * loss_recon + 0.3 * loss_percep + 0.1 * cos_loss
            loss = 0.6 * loss_recon + 0.4 * loss_percep 
            


            # --- Backprop ---
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        train_loss = running_loss / total
        train_losses.append(train_loss)

        # Evaluation loop
        model.eval()
        test_loss_total = 0.0
        test_total = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                z, x_rec = model(imgs)
                loss_recon = mse_loss(x_rec, imgs)
                loss_percep = vgg_loss(x_rec, imgs).mean()
                # cos_loss = 1 - F.cosine_similarity(z, z).mean()
                # loss = 0.6 * loss_recon + 0.3 * loss_percep + 0.1 * cos_loss
                loss = 0.6 * loss_recon + 0.4 * loss_percep 
  
                test_loss_total += loss.item() * imgs.size(0)
                test_total += imgs.size(0)
        test_loss = test_loss_total / test_total
        test_losses.append(test_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            ckpt_name = os.path.join(ckpt_dir, "autoencoder_best.pth")
            save_checkpoint(model, optimizer, epoch+1, ckpt_name)
            best_ckpt = ckpt_name
            print(f"Saved best checkpoint: {ckpt_name}")

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.title("Autoencoder Training with Perceptual + Cosine Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "autoencoder_training_curve.png"))
    plt.show()

    print(f"\nTraining finished  | Best test loss: {best_test_loss:.6f}")
    print(f"Best checkpoint saved at: {best_ckpt}")
    return model, best_test_loss, best_ckpt
