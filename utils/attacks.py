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

def generate_patch(model, img_shape=(3,64,64), patch_size=32, steps=500, lr=0.1, target=None, device='cpu'):
    """
    Generates an adversarial patch by gradient ascent on the classification loss.
    - model: classifier
    - img_shape: (C,H,W)
    - patch_size: side length of square patch
    - steps: optimization steps
    - lr: learning rate for patch
    - target: if None -> untargeted (maximize loss), else targeted label int
    Returns patch tensor in [-1,1].
    """
    C,H,W = img_shape
    # initialize patch as uniform small noise
    patch = torch.rand(1, C, patch_size, patch_size, device=device, requires_grad=True) * 2 - 1
    optimizer = torch.optim.Adam([patch], lr=lr)

    # use a small batch of training images to optimize patch
    # pick random examples from train_loader
    data_iter = iter(train_loader)
    data_batch, label_batch = next(data_iter)
    data_batch = data_batch.to(device)
    label_batch = label_batch.to(device)

    # patch position (top-left) - you can randomize this
    x0, y0 = 16, 16  # example position
    for step in range(steps):
        optimizer.zero_grad()
        patched = data_batch.clone()
        # tile patch to batch
        p_tile = patch.repeat(patched.size(0),1,1,1)
        # paste
        patched[:,:, y0:y0+patch_size, x0:x0+patch_size] = p_tile
        outputs = model(patched)
        if isinstance(outputs, tuple):
            outputs = outputs[1]
        if target is None:
            loss = -F.cross_entropy(outputs, label_batch)  # maximize loss (untargeted)
        else:
            tgt = torch.ones(patched.size(0), dtype=torch.long, device=device) * int(target)
            loss = F.cross_entropy(outputs, tgt)  # targeted: force target class
        loss.backward()
        # gradient step
        optimizer.step()
        # clamp patch to [-1,1]
        with torch.no_grad():
            patch.clamp_(-1.0,1.0)
        if step % 100 == 0:
            print(f"Patch step {step}, loss {loss.item():.4f}")
    return patch.detach()

def apply_patch(imgs, patch, x0=16, y0=16):
    """
    imgs: tensor (B,C,H,W) in [-1,1]
    patch: (1,C,ps,ps)
    returns patched images
    """
    imgs = imgs.clone()
    ps = patch.shape[-1]
    p = patch.repeat(imgs.size(0),1,1,1)
    imgs[:,:, y0:y0+ps, x0:x0+ps] = p
    return imgs
