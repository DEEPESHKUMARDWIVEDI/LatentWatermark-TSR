import torch

def fgsm_attack(model, images, labels, epsilon):
    """
    Performs Fast Gradient Sign Method attack
    Args:
        model: neural network model
        images: input images (B, C, H, W)
        labels: true labels (B,)
        epsilon: attack strength
    Returns:
        perturbed_images: adversarial images
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images
