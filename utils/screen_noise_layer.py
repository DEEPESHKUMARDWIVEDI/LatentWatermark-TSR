import torch
import torch.nn as nn
import torch.nn.functional as F

class ScreenNoiseLayer(nn.Module):
    """
    Simulate real-world distortions from screen shooting.
    Can include blurring, noise, and color distortions.
    """
    def __init__(self, blur_prob=0.5, noise_std=0.02):
        super(ScreenNoiseLayer, self).__init__()
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            # Random Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)

            # Random blur with probability
            if torch.rand(1) < self.blur_prob:
                kernel = torch.ones((1,1,3,3), device=x.device) / 9.0
                x = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return x
