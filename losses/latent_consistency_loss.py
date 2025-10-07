import torch
import torch.nn as nn

class LatentConsistencyLoss(nn.Module):
    """
    Ensures the latent vector remains consistent even under attacks/distortions
    """
    def __init__(self):
        super(LatentConsistencyLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, z_original, z_attacked):
        return self.criterion(z_attacked, z_original)
