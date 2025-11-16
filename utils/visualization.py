# utils/visualization.py
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.utils import make_grid, save_image
from utils import metrics 

def save_sample_with_metrics(model, dataloader, device, save_path="results/samples/reconstruction_metrics.png", num_images=8):
  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    mse_list, psnr_list, ssim_list, lpips_list =  [], [], [], []

    with torch.no_grad():
        imgs, _ = next(iter(dataloader))
        imgs = imgs[:num_images].to(device)
        z, recons = model(imgs)

    mse_val = ((recons - imgs) ** 2).mean().item()
    psnr_val = metrics.psnr(imgs, recons)
    ssim_val = metrics.ssim_batch(imgs, recons)
    lpips_val = metrics.lpips_batch(imgs, recons, device)

    mse_list.append(mse_val)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    lpips_list.append(lpips_val)

    print(f" Metrics on first {num_images} images:")
    print(f"Mean MSE: {np.mean(mse_list):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_list):.4f}")

    imgs_vis = (imgs + 1) / 2  
    recons_vis = (recons + 1) / 2
    grid = torch.cat([imgs_vis, recons_vis], dim=0)
    save_image(grid, save_path, nrow=num_images)
    print(f"Saved reconstruction comparison grid to {save_path}")

    grid_img = make_grid(grid, nrow=num_images)
    plt.figure(figsize=(num_images*2,4))
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.axis("off")
    plt.title("Top: Original | Bottom: Reconstructed")
    plt.show()
