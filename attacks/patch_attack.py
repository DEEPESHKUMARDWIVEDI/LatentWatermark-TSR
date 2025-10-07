import torch

def patch_attack(images, patch, position=(0,0)):
    """
    Apply a physical patch to images
    Args:
        images: batch of images (B, C, H, W)
        patch: small patch (C, h, w)
        position: top-left corner where patch is applied
    Returns:
        patched_images: images with patch applied
    """
    patched_images = images.clone()
    x, y = position
    _, _, h_patch, w_patch = patch.unsqueeze(0).shape
    patched_images[:, :, x:x+h_patch, y:y+w_patch] = patch
    return patched_images
