# utils/metrics.py
import torch
import numpy as np
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

    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            a = np.moveaxis(x[i], 0, -1)
            b = np.moveaxis(y[i], 0, -1)
            res.append(ssim(a, b, multichannel=True))
        return float(np.mean(res))
    else:
        a = np.moveaxis(x, 0, -1)
        b = np.moveaxis(y, 0, -1)
        return float(ssim(a, b, multichannel=True))
