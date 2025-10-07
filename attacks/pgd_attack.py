import torch

def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=40):
    """
    Performs Projected Gradient Descent attack
    Args:
        model: neural network model
        images: input images
        labels: true labels
        epsilon: max perturbation
        alpha: step size
        iters: number of iterations
    Returns:
        adv_images: adversarial images
    """
    images = images.clone().detach()
    ori_images = images.clone().detach()
    
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()
    return images
