# models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(8 * 8 * 128, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        zhat = self.fc(h)
        return zhat


if __name__ == "__main__":
    m = LatentDecoder(128)
    x = torch.randn(2, 3, 64, 64)
    z = m(x)
    print("z'", z.shape)
