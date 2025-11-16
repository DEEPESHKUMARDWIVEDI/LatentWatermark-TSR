import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetWatermark(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent_to_map = nn.Sequential(
            nn.Linear(latent_dim, 32 * 32),
            nn.ReLU(inplace=True)
        )

        self.enc1 = DoubleConv(img_channels + 1, 64)   # 3 img + 1 latent map
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, img, latent):
       
        latent_map = self.latent_to_map(latent).view(-1, 1, 32, 32)

        x = torch.cat([img, latent_map], dim=1)  

        e1 = self.enc1(x)              
        e2 = self.enc2(self.pool(e1))  
        e3 = self.enc3(self.pool(e2))  

        
        b = self.bottleneck(self.pool(e3))  

        d1 = self.up1(b)              
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)             
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)            
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.out_conv(d3)      
        return out
    
if __name__ == "__main__":
    model = UNetWatermark(img_channels=3, latent_dim=128, out_channels=3)
    img = torch.randn(4, 3, 32, 32)      
    latent = torch.randn(4, 128)         
    out = model(img, latent)
    print("Output shape:", out.shape)
