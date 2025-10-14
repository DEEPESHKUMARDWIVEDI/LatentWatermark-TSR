# models/latent_decoder.py
import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 8x8 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )
        self.fc = nn.Linear(4 * 4 * 256, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        z_pred = self.fc(h)
        return z_pred
