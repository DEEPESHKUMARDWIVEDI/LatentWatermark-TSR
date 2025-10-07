import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple BiSeNet-style backbone for segmentation
class BiSeNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BiSeNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class BiSeNetDetector(nn.Module):
    """
    Detects watermark regions and outputs corrected image
    """
    def __init__(self, in_channels=3, n_classes=2):
        super(BiSeNetDetector, self).__init__()
        self.spatial_path = nn.Sequential(
            BiSeNetBlock(in_channels, 64),
            BiSeNetBlock(64, 64)
        )
        self.context_path = nn.Sequential(
            BiSeNetBlock(in_channels, 128),
            BiSeNetBlock(128, 128)
        )
        self.combine = nn.Sequential(
            BiSeNetBlock(192, 64),
            nn.Conv2d(64, n_classes, 1)
        )

    def forward(self, x):
        sp = self.spatial_path(x)
        cp = F.interpolate(self.context_path(x), size=sp.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([sp, cp], dim=1)
        out = self.combine(combined)
        return out  # [B, n_classes, H, W]
