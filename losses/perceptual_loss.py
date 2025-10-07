import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3'], device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.layers = layers
        self.criterion = nn.MSELoss()

    def forward(self, input_img, target_img):
        input_features = self.get_features(input_img)
        target_features = self.get_features(target_img)
        loss = 0
        for f_in, f_tar in zip(input_features, target_features):
            loss += self.criterion(f_in, f_tar)
        return loss

    def get_features(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            layer_name = f'conv{name}_'  # dummy name for indexing
            if layer_name in self.layers:
                features.append(x)
        return features
