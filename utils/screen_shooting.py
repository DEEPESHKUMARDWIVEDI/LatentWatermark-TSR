#screen_shooting.py
import torch
import numpy as np
import cv2
import random

class ScreenShooting(torch.nn.Module):
    """
    Simulates real-world screen shooting artifacts:
    - Perspective distortion
    - Moiré noise
    - Gaussian & motion blur
    - Illumination changes
    - Color manipulation (hue, saturation)
    - JPEG compression

    Input:  (B, C, H, W) tensor in [0,1] or [-1,1]
    Output: same shape, same device, same range
    """

    def __init__(self, apply_prob=0.7):
        super(ScreenShooting, self).__init__()
        self.apply_prob = apply_prob

    @torch.no_grad()
    def forward(self, imgs):
        if random.random() > self.apply_prob:
            return imgs  # randomly skip

        imgs_out = []
        for img in imgs:
            # Convert tensor → numpy uint8 image
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            img_np = ((img_np + 1) / 2.0 * 255).astype(np.uint8) if img_np.min() < 0 else (img_np * 255).astype(np.uint8)

            # Apply combined screen effects
            img_np = self.apply_screen_effect(img_np)

            # Convert back to torch tensor in [-1, 1]
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            img_t = img_t * 2 - 1
            imgs_out.append(img_t)

        imgs_out = torch.stack(imgs_out, dim=0).to(imgs.device)
        return imgs_out

    # --------------------------------------------------
    # Main screen shooting distortion pipeline
    # --------------------------------------------------
    def apply_screen_effect(self, img):
        h, w = img.shape[:2]

        # --- Perspective warp ---
        src = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        dst = src + np.random.normal(0, 3, src.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # --- Moiré pattern ---
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        moire_freq = random.randint(20, 80)
        moire_intensity = random.uniform(10, 30)
        moire = (moire_intensity * np.sin(2 * np.pi * X / moire_freq)).astype(np.float32)
        moire = np.stack([moire] * 3, axis=-1)
        img = np.clip(img.astype(np.float32) + moire, 0, 255).astype(np.uint8)

        # --- Gaussian blur ---
        k = random.choice([1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)

        # --- Motion blur ---
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5, 7])
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            xs, ys = random.choice([(1, 0), (0, 1), (1, 1)])
            if xs:
                kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            elif ys:
                kernel_motion_blur[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
            else:
                np.fill_diagonal(kernel_motion_blur, 1)
            kernel_motion_blur /= kernel_size
            img = cv2.filter2D(img, -1, kernel_motion_blur)

        # --- Color manipulation ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180  # hue shift
        hsv[..., 1] *= random.uniform(0.8, 1.2)  # saturation
        hsv[..., 2] *= random.uniform(0.9, 1.1)  # brightness
        hsv = np.clip(hsv, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # --- Illumination jitter (contrast + brightness) ---
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # --- JPEG compression ---
        if random.random() < 0.8:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
            _, enc_img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

        return img
 