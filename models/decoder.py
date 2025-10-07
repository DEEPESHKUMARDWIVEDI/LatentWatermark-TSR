import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Decoder to recover latent vector from watermarked image
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(LatentDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*4*4, latent_dim)  # assuming 32x32 input

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        z_hat = self.fc(x)
        return z_hat
