# models/unet_watermark.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWatermark(nn.Module):
    """
    U-Net-style watermark embedder.
    Forward: Iw = forward(I, z) where
      I : [B,3,H,W], values in [-1,1]
      z : [B, latent_dim]
    Returns Iw same size as I.
    """
    def __init__(self, latent_dim=128, base_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )                                                   # H x W
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )                                                   # H/2 x W/2
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )                                                   # H/4 x W/4

        # latent projection -> spatial map
        self.latent_proj = nn.Linear(latent_dim, base_channels*4)

        # bottleneck fusion conv
        self.bottleneck_conv = nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1)

        # Decoder
        self.dec2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1) # H/2
        self.dec1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)   # H
        self.out_conv = nn.Conv2d(base_channels, 3, 3, padding=1)

        # output residual multiplier (controls watermark strength)
        self.res_scale = 0.05

    def forward(self, x, z):
        """
        x: [B,3,H,W] in [-1,1]
        z: [B,latent_dim]
        """
        B, C, H, W = x.shape
        e1 = self.enc1(x)      # B, base, H, W
        e2 = self.enc2(e1)     # B, 2base, H/2, W/2
        e3 = self.enc3(e2)     # B, 4base, H/4, W/4

        # latent -> map
        lp = self.latent_proj(z)                # B, 4base
        lp = lp.view(B, lp.size(1), 1, 1)       # B, 4base,1,1
        lp = F.interpolate(lp, size=(e3.size(2), e3.size(3)), mode='bilinear', align_corners=False)  # B,4base,h,w

        # concat along channels
        b = torch.cat([e3, lp], dim=1)          # B, 8base, h, w
        b = F.relu(self.bottleneck_conv(b))     # B,4base,h,w

        d2 = F.relu(self.dec2(b))               # B,2base,H/2,W/2
        d1 = F.relu(self.dec1(d2))              # B,base,H,W
        out = torch.tanh(self.out_conv(d1))     # residual in [-1,1]

        Iw = torch.clamp(x + self.res_scale * out, -1.0, 1.0)
        return Iw


if __name__ == "__main__":
    # quick shape test
    x = torch.randn(4,3,32,32)
    z = torch.randn(4,128)
    m = UNetWatermark(latent_dim=128)
    y = m(x, z)
    print("Iw", y.shape)
