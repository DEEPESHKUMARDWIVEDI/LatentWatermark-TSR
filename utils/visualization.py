# utils/visualization.py
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision.utils import make_grid

def show_tensor_image(x, title=None):
    if x.dim() == 4:
        x = make_grid(x)
    img = x.detach().cpu().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1.0) / 2.0
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

def save_sample(img_tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(img_tensor, nrow=4, normalize=True, scale_each=True)
    npimg = grid.detach().cpu().numpy()
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))
