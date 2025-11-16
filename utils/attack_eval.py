import torch
from utils.attacks import (
    fgsm_attack, pgd_attack, paste_patch, evaluate_patch_attack,
    add_gaussian_noise, add_salt_and_pepper,
    gaussian_blur_batch, motion_blur_batch,
    color_jitter_batch, random_affine_batch,
    random_perspective_batch
)

def compute_asr(model, dataloader, device, attack_fn):
    model.eval()
    total = correct = original_correct = correct_to_wrong = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        correct_mask = (preds == labels)
        original_correct += correct_mask.sum().item()

        adv_imgs = attack_fn(imgs)
        with torch.no_grad():
            adv_outputs = model(adv_imgs)
            _, adv_preds = torch.max(adv_outputs, 1)

        correct += (adv_preds == labels).sum().item()
        total += labels.size(0)
        correct_to_wrong += ((preds == labels) & (adv_preds != labels)).sum().item()

    acc = 100.0 * correct / total
    asr = 100.0 * correct_to_wrong / max(original_correct, 1)
    return acc, asr

def evaluate_clean(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    return acc

def evaluate_fgsm(model, dataloader, device, eps=0.03):
    return compute_asr(model, dataloader, device,
        lambda x: fgsm_attack(model, x, torch.argmax(model(x), 1), eps=eps, device=device)
    )

def evaluate_pgd(model, dataloader, device, eps=0.03, alpha=0.01, iters=10):
    return compute_asr(model, dataloader, device,
        lambda x: pgd_attack(model, x, torch.argmax(model(x), 1), eps=eps, alpha=alpha, iters=iters, device=device)
    )

def evaluate_patch(model, dataloader, patch, device, patch_size=0.2):
    return evaluate_patch_attack(model, dataloader, patch, device, patch_size)

def evaluate_noise(model, dataloader, device, sigma=0.05):
    return compute_asr(model, dataloader, device,
        lambda x: add_gaussian_noise(x, sigma=sigma)
    )

def evaluate_salt_pepper(model, dataloader, device, amount=0.01):
    return compute_asr(model, dataloader, device,
        lambda x: add_salt_and_pepper(x, amount=amount)
    )

def evaluate_blur(model, dataloader, device, mode='gaussian', ksize=5, motion_angle=None):
    if mode == 'gaussian':
        return compute_asr(model, dataloader, device,
            lambda x: gaussian_blur_batch(x, ksize=ksize)
        )
    else:
        return compute_asr(model, dataloader, device,
            lambda x: motion_blur_batch(x, kernel_size=ksize, angle=motion_angle)
        )

def evaluate_color_jitter(model, dataloader, device, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    return compute_asr(model, dataloader, device,
        lambda x: color_jitter_batch(x, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    )

def evaluate_affine(model, dataloader, device, degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10):
    return compute_asr(model, dataloader, device,
        lambda x: random_affine_batch(x, degrees=degrees, translate=translate, scale=scale, shear=shear)
    )

def evaluate_perspective(model, dataloader, device, distortion_scale=0.5):
    return compute_asr(model, dataloader, device,
        lambda x: random_perspective_batch(x, distortion_scale=distortion_scale)
    )
