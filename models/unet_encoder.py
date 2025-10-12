# models/unet_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWatermark(nn.Module):
    """
    U-Net-based watermark embedding network with flexible input size.
    Inputs:
        x: original image  [B, 3, H, W]
        z: latent vector   [B, latent_dim]
    Output:
        Iw: watermarked image [B, 3, H, W]
    """
    def __init__(self, latent_dim=128):
        super().__init__()

        # --- Encoder ---
        self.enc1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # --- Bottleneck ---
        self.bottleneck_conv = nn.Conv2d(128 + 128, 128, 3, stride=1, padding=1)

        # --- Decoder ---
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(32, 3, 3, stride=1, padding=1)

        # --- Latent projection ---
        self.latent_proj = nn.Linear(latent_dim, 128) 
        


    def forward(self, x, z):
        print("Forward pass of UNetWatermark")
        B, C, H, W = x.shape
        print("Input x:", x.shape)

        # --- Encoder ---
        e1 = F.relu(self.enc1(x))       # [B,32,H,W]
        e2 = F.relu(self.enc2(e1))      # [B,64,H/2,W/2]
        e3 = F.relu(self.enc3(e2))      # [B,128,H/4,W/4]
        print("e3:", e3.shape)
      # Latent projection
        lp = self.latent_proj(z)        # [B,128]
        lp = lp.view(B, 128, 1, 1)
        print("e3:", e3.shape)
        print("lp before interpolate:", lp.shape)
        lp = F.interpolate(lp, size=e3.shape[2:], mode='bilinear', align_corners=False)
        print("lp after interpolate:", lp.shape)
        # Bottleneck fusion
        b = torch.cat([e3, lp], dim=1)  # should now match perfectly
        b = F.relu(self.bottleneck_conv(b))


        # --- Decoder ---
        d2 = F.relu(self.dec2(b))       # upsample
        d1 = F.relu(self.dec1(d2))      # upsample
        out = torch.tanh(self.out_conv(d1))

        # --- Add watermark ---
        Iw = torch.clamp(x + 0.05 * out, -1.0, 1.0)
        return Iw



# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    m = UNetWatermark(latent_dim=128)
    x = torch.randn(2, 3, 32, 32)
    z = torch.randn(2, 128)
    Iw = m(x, z)
    print("32x32 Iw shape:", Iw.shape)

