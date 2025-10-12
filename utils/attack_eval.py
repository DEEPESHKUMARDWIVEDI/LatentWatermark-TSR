import torch
from utils.attacks import fgsm_attack, pgd_attack, paste_patch, evaluate_patch_attack

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
    model.eval()
    adv_correct, adv_total = 0, 0
    original_correct, correct_to_wrong = 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct_mask = (preds == labels)
        original_correct += correct_mask.sum().item()

        adv_imgs = fgsm_attack(model, imgs, labels, eps=eps, device=device)
        with torch.no_grad():
            adv_outputs = model(adv_imgs)
            _, adv_preds = torch.max(adv_outputs, 1)
        adv_correct += (adv_preds == labels).sum().item()
        adv_total += labels.size(0)
        correct_to_wrong += ((preds == labels) & (adv_preds != labels)).sum().item()
    adv_acc = 100.0 * adv_correct / adv_total
    asr = 100.0 * correct_to_wrong / max(original_correct, 1)
    return adv_acc, asr

def evaluate_pgd(model, dataloader, device, eps=0.03, alpha=0.01, iters=10):
    model.eval()
    adv_correct, adv_total = 0, 0
    original_correct, correct_to_wrong = 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct_mask = (preds == labels)
        original_correct += correct_mask.sum().item()

        adv_imgs = pgd_attack(model, imgs, labels, eps=eps, alpha=alpha, iters=iters, device=device)
        with torch.no_grad():
            adv_outputs = model(adv_imgs)
            _, adv_preds = torch.max(adv_outputs, 1)
        adv_correct += (adv_preds == labels).sum().item()
        adv_total += labels.size(0)
        correct_to_wrong += ((preds == labels) & (adv_preds != labels)).sum().item()
    adv_acc = 100.0 * adv_correct / adv_total
    asr = 100.0 * correct_to_wrong / max(original_correct, 1)
    return adv_acc, asr

def evaluate_patch(model, dataloader, patch, device, patch_size=0.2):
    return evaluate_patch_attack(model, dataloader, patch, device, patch_size)
