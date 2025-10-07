import matplotlib.pyplot as plt
import torch

def show_images(images, titles=None, nrow=4, figsize=(12,8)):
    """
    Display batch of images in a grid.
    Args:
        images: torch tensor [B,C,H,W]
        titles: list of titles for each image
        nrow: number of images per row
    """
    images = images.cpu().detach()
    B, C, H, W = images.shape
    ncol = int(np.ceil(B / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten()
    for i in range(B):
        img = images[i].permute(1,2,0)  # C,H,W -> H,W,C
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    for i in range(B, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
