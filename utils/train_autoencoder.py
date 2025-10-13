# utils/train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.autoencoder import Autoencoder, save_checkpoint, load_checkpoint

def train_autoencoder(train_loader, test_loader, device,
                      latent_dim=128, num_epochs=10, batch_size=64,
                      lr=1e-3, ckpt_dir="results/checkpoints/autoencoder"):
    """
    Train the autoencoder and save best checkpoint.
    Returns: trained model, best_test_loss
    """
    model = Autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_test_loss = float("inf")
    best_ckpt = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for imgs, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            z, x_rec = model(imgs)
            loss = criterion(x_rec, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = running_loss / total

        # Evaluation
        model.eval()
        test_loss_total = 0.0
        test_total = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                z, x_rec = model(imgs)
                loss = criterion(x_rec, imgs)
                test_loss_total += loss.item() * imgs.size(0)
                test_total += imgs.size(0)
        test_loss = test_loss_total / test_total

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        # Save checkpoint if best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            ckpt_name = os.path.join(ckpt_dir, "autoencoder_best.pth")
            save_checkpoint(model, optimizer, epoch+1, ckpt_name)
            best_ckpt = ckpt_name
            print(f"✅ Saved best checkpoint: {ckpt_name}")

    print("Training finished. Best test loss:", best_test_loss)
    return model, best_test_loss, best_ckpt

