import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def psnr(img1, img2):
    """
    Compute PSNR between two images.
    Args:
        img1, img2: torch tensors [C,H,W] or [B,C,H,W], range 0-1
    """
    if img1.ndim == 4:
        return np.mean([psnr_metric(i1.cpu().numpy(), i2.cpu().numpy()) 
                        for i1, i2 in zip(img1, img2)])
    else:
        return psnr_metric(img1.cpu().numpy(), img2.cpu().numpy())

def ssim(img1, img2):
    """
    Compute SSIM between two images.
    Args:
        img1, img2: torch tensors [C,H,W] or [B,C,H,W], range 0-1
    """
    if img1.ndim == 4:
        return np.mean([ssim_metric(i1.cpu().numpy().transpose(1,2,0), 
                                    i2.cpu().numpy().transpose(1,2,0), 
                                    multichannel=True) 
                        for i1, i2 in zip(img1, img2)])
    else:
        return ssim_metric(img1.cpu().numpy().transpose(1,2,0), 
                           img2.cpu().numpy().transpose(1,2,0), 
                           multichannel=True)

def latent_similarity(z1, z2, method='mse'):
    """
    Compare latent vectors.
    Args:
        z1, z2: torch tensors [B, latent_dim]
        method: 'mse' or 'cosine'
    """
    if method == 'mse':
        return F.mse_loss(z1, z2)
    elif method == 'cosine':
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        return 1 - (z1_norm * z2_norm).sum(dim=1).mean()
    else:
        raise ValueError("method should be 'mse' or 'cosine'")
