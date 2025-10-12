import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )
        self.fc = nn.Linear(4 * 4 * 128, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        z = self.fc(h)
        return z


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4 * 4 * 128)
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 16x16 -> 32x32
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        x_rec = self.deconv(h)
        return x_rec


# -----------------------------
# Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec



def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")



# -----------------------------
# Quick Test
# -----------------------------
if __name__ == "__main__":
    model = Autoencoder(latent_dim=128)
    x = torch.randn(2, 3, 32, 32)
    z, x_rec = model(x)
    print(f"Latent z: {z.shape}, Reconstructed x: {x_rec.shape}")
