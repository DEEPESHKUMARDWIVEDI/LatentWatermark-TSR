import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network to extract latent vector z from traffic sign image
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # H/8
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*4*4, latent_dim)  # Assuming input size 32x32

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    """
    Decoder to reconstruct image from latent vector z
    """
    def __init__(self, latent_dim=128, out_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # H*2
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # H*4
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x_hat = self.deconv(x)
        return x_hat

class Autoencoder(nn.Module):
    """
    Full autoencoder combining Encoder and Decoder
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
