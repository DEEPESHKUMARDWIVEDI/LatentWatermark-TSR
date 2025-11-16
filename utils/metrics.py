# utils/metrics.py
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10
import lpips

_lpips_model = None

def get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='vgg').to(device)
        _lpips_model.eval()
    return _lpips_model


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
        try:
            
            if a.shape[0] in [1, 3]: 
                a = np.moveaxis(a, 0, -1)
                b = np.moveaxis(b, 0, -1)

            if a.min() < 0 or b.min() < 0:
                a = (a + 1.0) / 2.0
                b = (b + 1.0) / 2.0

            win_size = min(a.shape[0], a.shape[1], 7)
            if win_size % 2 == 0:
                win_size -= 1

            val = ssim(a, b, channel_axis=-1, win_size=win_size, data_range=1.0)
            return float(val)
        except Exception as e:
            print(f"[Warning] SSIM failed for one image: {e}")
            return 0.0  

    if x.ndim == 4:  
        vals = [safe_ssim(x[i], y[i]) for i in range(x.shape[0])]
        return float(np.mean(vals))
    else:
        return safe_ssim(x, y)

def lpips_batch(x, y, device):
    model = get_lpips_model(device)
    with torch.no_grad():
        score = model(x, y)
    return score.mean().item()
