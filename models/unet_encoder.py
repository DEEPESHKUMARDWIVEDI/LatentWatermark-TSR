# models/unet_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.enc2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.enc3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bottleneck_conv = nn.Conv2d(128 + 16, 128, 3, 1, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out_conv = nn.Conv2d(32, 3, 3, 1, 1)
        self.latent_proj = nn.Linear(latent_dim, 16 * 8)

    def forward(self, x, z):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))

        lp = self.latent_proj(z)
        lp = lp.view(-1, 1, 16, 8)
        lp = F.interpolate(lp, size=(16, 16), mode="bilinear", align_corners=False)

        bottleneck = torch.cat([e3, lp.repeat(1, 16, 1, 1)[:, :128, :, :]], dim=1)
        b = F.relu(self.bottleneck_conv(bottleneck))

        d2 = F.relu(self.dec2(b))
        d1 = F.relu(self.dec1(d2))
        out = torch.tanh(self.out_conv(d1))
        Iw = torch.clamp(x + 0.05 * out, -1.0, 1.0)
        return Iw


if __name__ == "__main__":
    m = SimpleUNet(128)
    x = torch.randn(2, 3, 64, 64)
    z = torch.randn(2, 128)
    Iw = m(x, z)
    print("Iw", Iw.shape)
