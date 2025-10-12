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

# ---------------------------
# Patch attack utilities
# ---------------------------
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

def paste_patch(imgs, patch, positions=None, scale=0.2):
    """
    Paste a patch onto a batch of images.
    imgs: tensor (B,C,H,W), values in [-1,1] or [0,1] depending on your pipeline.
    patch: tensor (C, Ph, Pw) in same value-range as imgs (recommend [-1,1] if imgs normalized that way)
    positions: list of (x_center, y_center) in normalized coords [0,1], length B or None -> random
    scale: fraction of image width used for patch (square)
    returns: patched images (B,C,H,W)
    """
    device = imgs.device
    B, C, H, W = imgs.shape
    # ensure patch is on same device
    patch = patch.to(device)
    # compute patch size
    psize = int(scale * W)
    # resize patch
    patch_resized = F.interpolate(patch.unsqueeze(0), size=(psize, psize), mode='bilinear', align_corners=False)[0]
    patched = imgs.clone()
    for i in range(B):
        if positions is None:
            # random position with some margin
            xc = random.uniform(0.15, 0.85)
            yc = random.uniform(0.15, 0.85)
        else:
            xc, yc = positions[i]
        # top-left coords
        x = int(xc * W - psize // 2)
        y = int(yc * H - psize // 2)
        x = max(0, min(W-psize, x))
        y = max(0, min(H-psize, y))
        # blend: simple overwrite (you can do alpha blending)
        patched[i, :, y:y+psize, x:x+psize] = patch_resized
    return patched

def train_adversarial_patch(model, dataloader, device, num_epochs=100, lr=0.1,
                            patch_size=0.2, target=None, init="random", budget=0.2):
    """
    Train an adversarial patch (white-box) to cause misclassification.
    - model: classifier (in eval() but we will call it for grads)
    - dataloader: data to optimize patch on (train subset)
    - device: 'cuda' or 'cpu'
    - num_epochs: optimization steps
    - lr: optimizer lr for the patch
    - patch_size: fraction of image width (square)
    - target: if int -> targeted attack (target class), else None -> untargeted (maximize loss)
    - init: "random" or "zeros" or "image" (init from mean)
    - budget: clamp patch range if you want smaller magnitude; patch assumed to be in same range as inputs
    Returns: learned patch tensor (C,Ph,Pw) in range matching inputs
    """
    model.eval()
    # assume images have shape (C,H,W) normalized in [-1,1]
    # create a patch parameter
    sample = next(iter(dataloader))[0][0]  # a single sample, (C,H,W)
    C = sample.shape[0]; H = sample.shape[1]; W = sample.shape[2]
    psize = int(patch_size * W)
    # initialize patch
    if init == "random":
        patch = torch.randn((1, C, psize, psize), device=device, requires_grad=True)
    elif init == "zeros":
        patch = torch.zeros((1, C, psize, psize), device=device, requires_grad=True)
    else:
        patch = torch.randn((1, C, psize, psize), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            B = imgs.size(0)
            # sample random positions
            positions = [(random.uniform(0.15,0.85), random.uniform(0.15,0.85)) for _ in range(B)]
            # apply patch
            patched = paste_patch(imgs, patch[0], positions=positions, scale=patch_size)
            # forward
            outputs = model(patched)
            # compute objective
            if target is None:
                # untargeted: maximize loss -> minimize negative loss (i.e., make model wrong)
                loss = -loss_fn(outputs, labels)
            else:
                tgt = torch.ones(B, dtype=torch.long, device=device) * int(target)
                loss = loss_fn(outputs, tgt)  # targeted: move to target
            optimizer.zero_grad()
            loss.backward()
            # gradient step
            optimizer.step()
            # optional: clamp patch to reasonable range (if inputs in [-1,1])
            with torch.no_grad():
                patch.clamp_(-1.0, 1.0)
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f"[Patch train] Epoch {epoch+1}/{num_epochs}, loss: {epoch_loss/len(dataloader):.4f}")
    # return patch (C,Ph,Pw)
    return patch.detach()[0].cpu()

def evaluate_patch_attack(model, dataloader, patch, device, patch_size=0.2, positions=None):
    """
    Apply a given patch to the test set and compute metrics (clean acc, adv acc, ASR).
    - patch: tensor (C,Ph,Pw) on cpu or device
    - positions: if None use random positions per sample
    """
    model.eval()
    patch = patch.to(device)
    total = 0; clean_correct = 0; adv_correct = 0; original_correct = 0; correct_to_wrong = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device); labels = labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        clean_correct += (preds == labels).sum().item()
        total += labels.size(0)
        # make patched images
        B = imgs.size(0)
        if positions is None:
            pos = None
        else:
            pos = positions  # should be list len B of tuples
        patched = paste_patch(imgs, patch, positions=None, scale=patch_size)
        with torch.no_grad():
            out_adv = model(patched)
            _, preds_adv = torch.max(out_adv, 1)
        adv_correct += (preds_adv == labels).sum().item()
        # ASR counters
        original_correct += (preds == labels).sum().item()
        correct_to_wrong += ((preds == labels) & (preds_adv != labels)).sum().item()
    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    asr = 100.0 * correct_to_wrong / max(original_correct, 1)
    return {"clean_acc": clean_acc, "adv_acc": adv_acc, "asr": asr}
