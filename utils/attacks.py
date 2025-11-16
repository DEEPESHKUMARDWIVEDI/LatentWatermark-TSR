# utils/attacks.py
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
import cv2
from torchvision.transforms import InterpolationMode

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

def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, iters=20, device='cpu'):
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

def paste_patch(imgs, patch, positions=None, scale=0.2):
  
    device = imgs.device
    B, C, H, W = imgs.shape

    patch = patch.to(device)
    psize = int(scale * W)
    patch_resized = F.interpolate(patch.unsqueeze(0), size=(psize, psize), mode='bilinear', align_corners=False)[0]
    patched = imgs.clone()
    for i in range(B):
        if positions is None:
            xc = random.uniform(0.15, 0.85)
            yc = random.uniform(0.15, 0.85)
        else:
            xc, yc = positions[i]
        x = int(xc * W - psize // 2)
        y = int(yc * H - psize // 2)
        x = max(0, min(W-psize, x))
        y = max(0, min(H-psize, y))
        patched[i, :, y:y+psize, x:x+psize] = patch_resized
    return patched

def train_adversarial_patch(model, dataloader, device, num_epochs=100, lr=0.1,
                            patch_size=0.2, target=None, init="random", budget=0.2):
    model.eval()
    sample = next(iter(dataloader))[0][0] 
    C = sample.shape[0]; H = sample.shape[1]; W = sample.shape[2]
    psize = int(patch_size * W)
    
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
            
            positions = [(random.uniform(0.15,0.85), random.uniform(0.15,0.85)) for _ in range(B)]
        
            patched = paste_patch(imgs, patch[0], positions=positions, scale=patch_size)
        
            outputs = model(patched)
            
            if target is None:
            
                loss = -loss_fn(outputs, labels)
            else:
                tgt = torch.ones(B, dtype=torch.long, device=device) * int(target)
                loss = loss_fn(outputs, tgt)  
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            with torch.no_grad():
                patch.clamp_(-1.0, 1.0)
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f"[Patch train] Epoch {epoch+1}/{num_epochs}, loss: {epoch_loss/len(dataloader):.4f}")
    return patch.detach()[0].cpu()

def evaluate_patch_attack(model, dataloader, patch, device, patch_size=0.2, positions=None):
  
    model.eval()
    patch = patch.to(device)
    total = 0; clean_correct = 0; adv_correct = 0; original_correct = 0; correct_to_wrong = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device); labels = labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        clean_correct += (preds == labels).sum().item()
        total += labels.size(0)
    
        B = imgs.size(0)
        if positions is None:
            pos = None
        else:
            pos = positions  
        patched = paste_patch(imgs, patch, positions=None, scale=patch_size)
        with torch.no_grad():
            out_adv = model(patched)
            _, preds_adv = torch.max(out_adv, 1)
        adv_correct += (preds_adv == labels).sum().item()
        original_correct += (preds == labels).sum().item()
        correct_to_wrong += ((preds == labels) & (preds_adv != labels)).sum().item()
    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    asr = 100.0 * correct_to_wrong / max(original_correct, 1)
    return {"clean_acc": clean_acc, "adv_acc": adv_acc, "asr": asr}

def _to_uint8(img_tensor):

    arr = img_tensor.detach().cpu().numpy()
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return arr

def _to_tensor_uint8(img_uint8, device):

    arr = img_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float().to(device)
    t = t * 2.0 - 1.0
    return t

def add_gaussian_noise(imgs, sigma=0.05):

    device = imgs.device
    x = imgs.clone()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
        noisy = x + torch.randn_like(x) * sigma
        noisy = torch.clamp(noisy, 0.0, 1.0)
        noisy = noisy * 2.0 - 1.0
    else:
        noisy = x + torch.randn_like(x) * sigma
        noisy = torch.clamp(noisy, 0.0, 1.0)
    return noisy.to(device)

def add_salt_and_pepper(imgs, amount=0.01, s_vs_p=0.5):
    
    device = imgs.device
    x = imgs.clone()
    B, C, H, W = x.shape
    out = x.clone()
    for i in range(B):
        num_pixels = int(amount * H * W)
        num_salt = int(num_pixels * s_vs_p)
        coords = (np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt))
        for c in range(C):
            out[i, c, coords[0], coords[1]] = 1.0 
        
        num_pepper = num_pixels - num_salt
        coords = (np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper))
        for c in range(C):
            out[i, c, coords[0], coords[1]] = -1.0 
    return out.to(device)

def gaussian_blur_batch(imgs, ksize=5):

    device = imgs.device
    out = []
    for img in imgs:
        im = _to_uint8(img)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(im, (ksize, ksize), 0)
        out.append(_to_tensor_uint8(blurred, device))
    return torch.stack(out, dim=0)

def motion_blur_batch(imgs, kernel_size=9, angle=None):

    device = imgs.device
    out = []
    for img in imgs:
        im = _to_uint8(img)
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if angle is None:
            ang = random.uniform(0, 180)
        else:
            ang = angle
        cx = kernel_size // 2
        cy = kernel_size // 2
        k[:, cx] = 1.0
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
        k = k / k.sum() if k.sum() != 0 else k
        blurred = cv2.filter2D(im, -1, k)
        out.append(_to_tensor_uint8(blurred, device))
    return torch.stack(out, dim=0)


def color_jitter_batch(imgs, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    
    device = imgs.device
    out = []
    for img in imgs:
        t = img.clone()
        if t.min() < 0:
            t = (t + 1.0) / 2.0

        b = 1.0 + random.uniform(-brightness, brightness)
        c = 1.0 + random.uniform(-contrast, contrast)
        s = 1.0 + random.uniform(-saturation, saturation)
        h = random.uniform(-hue, hue)
        t = TF.adjust_brightness(t, b)
        t = TF.adjust_contrast(t, c)
        t = TF.adjust_saturation(t, s)
        t = TF.adjust_hue(t, h)
        t = torch.clamp(t, 0.0, 1.0)
        if img.min() < 0:
            t = t * 2.0 - 1.0
        out.append(t.to(device))
    return torch.stack(out, dim=0)

def random_affine_batch(imgs, degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10):
    
    B = imgs.size(0)
    adv_list = []
    for i in range(B):
        img = imgs[i]
        angle = random.uniform(-degrees, degrees)
        trans_x = random.uniform(-translate[0], translate[0]) * img.size(1)
        trans_y = random.uniform(-translate[1], translate[1]) * img.size(2)
        scale_v = random.uniform(scale[0], scale[1])
        shear_v = random.uniform(-shear, shear)

        t = TF.affine(
            img,
            angle=angle,
            translate=(int(trans_x), int(trans_y)),
            scale=scale_v,
            shear=shear_v,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )
        t = torch.clamp(t, -1.0, 1.0)
        adv_list.append(t)
    return torch.stack(adv_list)

def random_perspective_batch(imgs, distortion_scale=0.5):
    B, C, H, W = imgs.shape
    transformed_imgs = []

    for i in range(B):
        img = imgs[i]
        half_height = H // 2
        half_width = W // 2
        topleft = [random.uniform(0, distortion_scale * half_width),
                   random.uniform(0, distortion_scale * half_height)]
        topright = [W - random.uniform(0, distortion_scale * half_width),
                    random.uniform(0, distortion_scale * half_height)]
        botright = [W - random.uniform(0, distortion_scale * half_width),
                    H - random.uniform(0, distortion_scale * half_height)]
        botleft = [random.uniform(0, distortion_scale * half_width),
                   H - random.uniform(0, distortion_scale * half_height)]

        startpoints = [[0, 0], [W, 0], [W, H], [0, H]]
        endpoints = [topleft, topright, botright, botleft]

        img_t = TF.perspective(img, startpoints, endpoints, interpolation=TF.InterpolationMode.BILINEAR)
        transformed_imgs.append(img_t.unsqueeze(0))

    return torch.cat(transformed_imgs, dim=0)

