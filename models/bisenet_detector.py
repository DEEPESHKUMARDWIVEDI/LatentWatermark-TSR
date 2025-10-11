# models/bisenet_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        h = self.backbone(x)
        out = self.decoder(h)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    net = SmallSeg(2)
    x = torch.randn(2, 3, 64, 64)
    logits = net(x)
    print(logits.shape)
