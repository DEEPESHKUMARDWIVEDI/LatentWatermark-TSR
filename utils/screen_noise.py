# utils/screen_noise.py
import torch
import torch.nn as nn

class ScreenNoiseLayer(nn.Module):
    """
    Adds Gaussian noise to simulate screen or transmission distortion.
    """
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return torch.clamp(x + noise, -1, 1)
        else:
            return x
