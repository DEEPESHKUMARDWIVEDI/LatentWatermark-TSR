import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    """
    L2 loss between original latent vector and reconstructed latent vector
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, z, z_hat):
        return self.criterion(z_hat, z)
