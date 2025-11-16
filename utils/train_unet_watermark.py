# # # # utils/train_unet_watermark.py
import os
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.unet_watermark import UNetWatermark
from models.autoencoder import Autoencoder, load_checkpoint
from models.latent_decoder import LatentDecoder
from utils.perceptual import VGGPerceptualLoss
from utils.screen_shooting import ScreenShooting  


def train_unet_watermark(train_loader, device,
                         autoencoder_ckpt="results/checkpoints/autoencoder/autoencoder_best.pth",
                         latent_dim=128,
                         num_epochs=10,
                         lr=1e-4,
                         ckpt_dir="results/checkpoints/unet",
                         shooting_prob=0.2):


    os.makedirs(ckpt_dir, exist_ok=True)

    ae = Autoencoder(latent_dim=latent_dim).to(device)
    load_checkpoint(ae, None, autoencoder_ckpt, map_location=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    unet = UNetWatermark(latent_dim=latent_dim).to(device)
    latent_decoder = LatentDecoder(latent_dim=latent_dim).to(device)
    screen = ScreenShooting(apply_prob=shooting_prob).to(device)  

    optimizer_unet = optim.Adam(unet.parameters(), lr=lr)
    optimizer_latent = optim.Adam(latent_decoder.parameters(), lr=lr)

    vgg_loss = VGGPerceptualLoss(device=device)
    mse_loss = nn.MSELoss()
    cross_entropy = nn.BCELoss()
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        unet.train()
        latent_decoder.train()
        running_loss = 0.0
        total = 0

        for imgs, _ in tqdm(train_loader, desc=f"UNet Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)

            with torch.no_grad():
                z = ae.encoder(imgs)

            Iw = unet(imgs, z)

            Iw_shot = screen(Iw)
            
            z_pred = latent_decoder(Iw_shot)

            I_rec = ae.decoder(z_pred)
            loss_percep = vgg_loss(imgs, Iw)
            loss_qual = mse_loss(imgs, Iw)
            loss_latent =  mse_loss(z_pred, z)
            loss_recon = vgg_loss(imgs, I_rec)
            cos_loss = 1 - F.cosine_similarity(z_pred, z).mean()
       

            loss = 0.6 * loss_percep + 0.6 * loss_recon + loss_latent + 0.2 * loss_qual + 0.2 * cos_loss


            optimizer_unet.zero_grad()
            optimizer_latent.zero_grad()
            loss.backward()
            optimizer_unet.step()
            optimizer_latent.step()

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        avg_loss = running_loss / total
        print(f"[Epoch {epoch+1}] Avg Total Loss: {avg_loss:.6f}")
        unet_ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": unet.state_dict(),
            "loss": avg_loss
        }, unet_ckpt_path)

        latent_ckpt_path = os.path.join(ckpt_dir, f"latent_decoder_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": latent_decoder.state_dict(),
            "loss": avg_loss
        }, latent_ckpt_path)


        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save({
                "epoch": epoch+1,
                "model_state": unet.state_dict(),
                "loss": avg_loss
            }, os.path.join(ckpt_dir, "unet_best.pth"))

            torch.save({
                "epoch": epoch+1,
                "model_state": latent_decoder.state_dict(),
                "loss": avg_loss
            }, os.path.join(ckpt_dir, "latent_decoder_best.pth"))

            print("Saved best UNet and LatentDecoder checkpoints.")

    return unet, latent_decoder, best_loss







# # # utils/train_unet_watermark.py
# import os
# import torch
# from tqdm import tqdm
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F


# from models.unet_watermark import UNetWatermark
# from models.autoencoder import Autoencoder, load_checkpoint
# from models.latent_decoder import LatentDecoder
# from utils.perceptual import VGGPerceptualLoss
# from utils.screen_shooting import ScreenShooting  


# def train_unet_watermark(train_loader, device,
#                          autoencoder_ckpt="results/checkpoints/autoencoder/autoencoder_best.pth",
#                          latent_dim=128,
#                          num_epochs=10,
#                          lr=1e-4,
#                          ckpt_dir="results/checkpoints/unet",
#                          shooting_prob=0.2):


#     os.makedirs(ckpt_dir, exist_ok=True)

#     ae = Autoencoder(latent_dim=latent_dim).to(device)
#     load_checkpoint(ae, None, autoencoder_ckpt, map_location=device)
#     ae.eval()
#     for p in ae.parameters():
#         p.requires_grad = False

#     unet = UNetWatermark(latent_dim=latent_dim).to(device)
#     latent_decoder = LatentDecoder(latent_dim=latent_dim).to(device)
#     screen = ScreenShooting(apply_prob=shooting_prob).to(device) 
#     unet.eval() 

#     optimizer_latent = optim.Adam(latent_decoder.parameters(), lr=lr)

#     vgg_loss = VGGPerceptualLoss(device=device)
#     mse_loss = nn.MSELoss()
#     cross_entropy = nn.BCELoss()
#     best_loss = float("inf")
    
#     for epoch in range(num_epochs):
#         unet.train()
#         latent_decoder.train()
#         running_loss = 0.0
#         total = 0

#         for imgs, _ in tqdm(train_loader, desc=f"UNet Epoch {epoch+1}/{num_epochs}"):
#             imgs = imgs.to(device)

#             with torch.no_grad():
#                 z = ae.encoder(imgs)

#             Iw = unet(imgs, z)

#             Iw_shot = screen(Iw)
            
#             z_pred = latent_decoder(Iw_shot)

            
#             loss_latent =  mse_loss(z_pred, z)
            
#             cos_loss = 1 - F.cosine_similarity(z_pred, z).mean()
       

#             loss = loss_latent +  cos_loss


#             optimizer_latent.zero_grad()
#             loss.backward()
#             optimizer_latent.step()

#             running_loss += loss.item() * imgs.size(0)
#             total += imgs.size(0)

#         avg_loss = running_loss / total
#         print(f"[Epoch {epoch+1}] Avg Total Loss: {avg_loss:.6f}")
#         unet_ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch+1}.pth")
#         torch.save({
#             "epoch": epoch+1,
#             "model_state": unet.state_dict(),
#             "loss": avg_loss
#         }, unet_ckpt_path)

#         latent_ckpt_path = os.path.join(ckpt_dir, f"latent_decoder_epoch{epoch+1}.pth")
#         torch.save({
#             "epoch": epoch+1,
#             "model_state": latent_decoder.state_dict(),
#             "loss": avg_loss
#         }, latent_ckpt_path)


#         if avg_loss < best_loss:
#             best_loss = avg_loss

#             torch.save({
#                 "epoch": epoch+1,
#                 "model_state": unet.state_dict(),
#                 "loss": avg_loss
#             }, os.path.join(ckpt_dir, "unet_best.pth"))

#             torch.save({
#                 "epoch": epoch+1,
#                 "model_state": latent_decoder.state_dict(),
#                 "loss": avg_loss
#             }, os.path.join(ckpt_dir, "latent_decoder_best.pth"))

#             print("Saved best UNet and LatentDecoder checkpoints.")

#     return unet, latent_decoder, best_loss


























# # import os
# # import torch
# # from tqdm import tqdm
# # import torch.optim as optim
# # import torch.nn as nn

# # from models.unet_watermark import UNetWatermark
# # from models.autoencoder import Autoencoder, load_checkpoint
# # from models.latent_decoder import LatentDecoder
# # from utils.perceptual import VGGPerceptualLoss



# # def train_unet_watermark(train_loader, device,
# #                          autoencoder_ckpt="results/checkpoints/autoencoder/autoencoder_best.pth",
# #                          latent_dim=128,
# #                          num_epochs=10,
# #                          lr=1e-4,
# #                          ckpt_dir="results/checkpoints/unet"):
# #     """
# #     Train the UNet watermarking model and LatentDecoder jointly using perceptual, reconstruction,
# #     and latent consistency losses.

# #     Args:
# #         train_loader: DataLoader for training images
# #         device: torch.device ('cuda' or 'cpu')
# #         autoencoder_ckpt: path to pretrained autoencoder checkpoint
# #         latent_dim: latent dimension size
# #         num_epochs: number of training epochs
# #         lr: learning rate
# #         ckpt_dir: directory to save checkpoints
# #     """

# #     os.makedirs(ckpt_dir, exist_ok=True)

# #     # ---------------------------
# #     # 1. Load pretrained Autoencoder (frozen)
# #     # ---------------------------
# #     ae = Autoencoder(latent_dim=latent_dim).to(device)
# #     load_checkpoint(ae, None, autoencoder_ckpt, map_location=device)
# #     ae.eval()
# #     for p in ae.parameters():
# #         p.requires_grad = False

# #     # ---------------------------
# #     # 2. Initialize UNet and LatentDecoder
# #     # ---------------------------
# #     unet = UNetWatermark(latent_dim=latent_dim).to(device)
# #     latent_decoder = LatentDecoder(latent_dim=latent_dim).to(device)

# #     optimizer_unet = optim.Adam(unet.parameters(), lr=lr)
# #     optimizer_latent = optim.Adam(latent_decoder.parameters(), lr=lr)

# #     vgg_loss = VGGPerceptualLoss(device=device)
# #     mse_loss = nn.MSELoss()

# #     best_loss = float("inf")
    
# #     # ---------------------------
# #     # 3. Training Loop
# #     # ---------------------------
# #     for epoch in range(num_epochs):
# #         unet.train()
# #         latent_decoder.train()
# #         running_loss = 0.0
# #         total = 0

# #         for imgs, _ in tqdm(train_loader, desc=f"UNet Epoch {epoch+1}/{num_epochs}"):
# #             imgs = imgs.to(device)

# #             # Encode latent representation (z)
# #             with torch.no_grad():
# #                 z, _ = ae(imgs)

# #             # Generate watermarked image
# #             Iw = unet(imgs, z)

# #             # Predict latent from watermarked image
# #             z_pred = latent_decoder(Iw)

# #             # Reconstruct image from predicted latent
# #             I_rec = ae.decoder(z_pred)

# #             # ---------------------------
# #             # Compute Combined Loss
# #             # ---------------------------
# #             loss_percep = vgg_loss(imgs, Iw)
# #             loss_latent = mse_loss(z_pred, z)
# #             loss_recon = vgg_loss(imgs, I_rec)

# #             # Weighted total loss
# #             loss = loss_percep + 0.5 * loss_recon + 0.1 * loss_latent

# #             optimizer_unet.zero_grad()
# #             optimizer_latent.zero_grad()
# #             loss.backward()
# #             optimizer_unet.step()
# #             optimizer_latent.step()

# #             running_loss += loss.item() * imgs.size(0)
# #             total += imgs.size(0)

# #         avg_loss = running_loss / total
# #         print(f"[Epoch {epoch+1}] Avg Total Loss: {avg_loss:.6f}")

# #         # ---------------------------
# #         # 4. Save Model Checkpoints
# #         # ---------------------------
# #         unet_ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch+1}.pth")
# #         torch.save({
# #             "epoch": epoch+1,
# #             "model_state": unet.state_dict(),
# #             "loss": avg_loss
# #         }, unet_ckpt_path)

# #         latent_ckpt_path = os.path.join(ckpt_dir, f"latent_decoder_epoch{epoch+1}.pth")
# #         torch.save({
# #             "epoch": epoch+1,
# #             "model_state": latent_decoder.state_dict(),
# #             "loss": avg_loss
# #         }, latent_ckpt_path)

# #         # Save best models
# #         if avg_loss < best_loss:
# #             best_loss = avg_loss

# #             torch.save({
# #                 "epoch": epoch+1,
# #                 "model_state": unet.state_dict(),
# #                 "loss": avg_loss
# #             }, os.path.join(ckpt_dir, "unet_best.pth"))

# #             torch.save({
# #                 "epoch": epoch+1,
# #                 "model_state": latent_decoder.state_dict(),
# #                 "loss": avg_loss
# #             }, os.path.join(ckpt_dir, "latent_decoder_best.pth"))

# #             print("Saved best UNet and LatentDecoder checkpoints.")

# #     return unet, latent_decoder, best_loss
