# utils/metrics.py
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10

def psnr(img1, img2):
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10(1.0 / mse)

def ssim_batch(x, y):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    def safe_ssim(a, b):
        # Move channel to last dimension if needed
        if a.shape[0] in [1, 3]:  # C,H,W -> H,W,C
            a = np.moveaxis(a, 0, -1)
            b = np.moveaxis(b, 0, -1)
        # compute win_size: must be odd and <= min(H,W)
        win_size = min(a.shape[0], a.shape[1], 7)
        if win_size % 2 == 0:
            win_size -= 1
        return ssim(a, b, channel_axis=-1, win_size=win_size)

    if x.ndim == 4:  # batch of images
        res = []
        for i in range(x.shape[0]):
            res.append(safe_ssim(x[i], y[i]))
        return float(np.mean(res))
    else:
        return float(safe_ssim(x, y))



def latent_similarity(z, z_hat, metric='mse'):
    """
    Computes similarity between latent vectors.
    Args:
        z: original latent vector (tensor)
        z_hat: recovered latent vector (tensor)
        metric: 'mse' or 'cosine'
    Returns:
        similarity score (float)
    """
    if metric == 'mse':
        return F.mse_loss(z_hat, z, reduction='mean').item()
    elif metric == 'cosine':
        z_norm = F.normalize(z, dim=1)
        z_hat_norm = F.normalize(z_hat, dim=1)
        return (1 - (z_norm * z_hat_norm).sum(dim=1).mean()).item()
    else:
        raise ValueError("metric must be 'mse' or 'cosine'")
