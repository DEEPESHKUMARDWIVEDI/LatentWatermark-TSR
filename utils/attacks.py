# utils/attacks.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, eps=0.03, device='cpu'):
    images = images.clone().detach().to(device)
    images.requires_grad = True
    outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs[1]
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    adv_images = images + eps * grad.sign()
    return torch.clamp(adv_images, -1.0, 1.0)

def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, iters=10, device='cpu'):
    ori_images = images.clone().detach().to(device)
    adv_images = images.clone().detach().to(device)
    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        if isinstance(outputs, tuple):
            outputs = outputs[1]
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + eta, -1.0, 1.0).detach_()
    return adv_images
