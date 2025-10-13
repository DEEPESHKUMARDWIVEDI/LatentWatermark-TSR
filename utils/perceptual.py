# utils/perceptual.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models

# ImageNet normalization (VGG expects inputs normalized)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features (relu1_1..relu4_1).
    Expects inputs in range [-1,1]. Internally converts to [0,1] and applies ImageNet normalization and resize to 224.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        # take first layers upto relu4_1 (inclusive) ~ index 21
        self.slice_ids = [2, 7, 12, 21]  # relu1_1, relu2_1, relu3_1, relu4_1
        self.slices = nn.ModuleList()
        prev = 0
        for sid in self.slice_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:sid+1]))
            prev = sid+1
        for p in self.parameters():
            p.requires_grad = False
        self.device = device
        self.mean = IMAGENET_MEAN.to(device).view(1,3,1,1)
        self.std  = IMAGENET_STD.to(device).view(1,3,1,1)

    def forward(self, x, y):
        """
        x, y : tensors in [-1,1], shape [B,3,H,W]
        returns sum of L2 differences across selected feature slices
        """
        # convert to [0,1]
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        # resize to 224 for VGG
        x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        y = nn.functional.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)

        # normalize to imagenet
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        xi = x
        yi = y
        for s in self.slices:
            xi = s(xi)
            yi = s(yi)
            loss = loss + torch.mean((xi - yi)**2)
        return loss
