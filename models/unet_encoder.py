import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    """
    U-Net based encoder to embed latent vector into image
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(UNetEncoder, self).__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.out_conv = nn.Conv2d(64, in_channels, 1)
        self.latent_fc = nn.Linear(latent_dim, 256*4*4)  # Assuming 32x32 input

    def forward(self, x, z):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Inject latent vector
        z_proj = self.latent_fc(z).view(-1, 256, 4, 4)
        e3 = e3 + z_proj  # Broadcast addition

        # Decode
        d2 = self.up2(e3) + e2
        d1 = self.up1(d2) + e1
        out = torch.sigmoid(self.out_conv(d1))
        return out
