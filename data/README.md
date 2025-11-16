# Latent Watermark for Traffic Sign Recognition (TSR)

## Project Overview
This project builds upon **StegaStamp** to develop a **latent-space watermarking defense** for Traffic Sign Recognition (TSR) systems.  
Unlike traditional image-level watermarking, our approach embeds watermarks in the **semantic latent vector space** of the sign image.  
This makes the watermark **robust against adversarial attacks, screen-shooting distortions**, and **image transformations** such as perspective, illumination, and Moiré distortions.

---

## Objectives
- Implement a **latent-space watermark encoder-decoder** pipeline.  
- Simulate **screen-shooting noise layers** (Perspective, Illumination, Moiré distortions).  
- Evaluate watermark robustness under **FGSM/PGD attacks** and distortions.  
- Integrate watermarking defense in a **TSR model** trained on the GTSRB dataset.

---

## Architecture
Our model consists of:
1. **Encoder (U-Net-based)** – embeds watermark into the latent space.  
2. **Noise Layer** – applies distortions such as perspective, illumination, and moiré.  
3. **Decoder** – reconstructs and verifies the latent watermark.  
4. **TSR Classifier** – trained to classify traffic signs while maintaining watermark integrity.

Total loss:
\[
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{latent} + \lambda_3 \mathcal{L}_{detect} + \lambda_4 \mathcal{L}_{robust}
\]

---

## Dataset
We use the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.  
It is publicly available and contains over 50,000 labeled images across 43 sign classes.

---

## Current Progress (Phase-II)
-  Dataset preprocessed and loaded into training pipeline  
-  U-Net Encoder and Decoder implemented  
-  Screen noise distortion layer integrated  
-  Initial training and watermark embedding verified  
-  Next: Evaluate adversarial robustness and TSR accuracy impact  

---

##  How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/LatentWatermark-TSR.git
   cd LatentWatermark-TSR

2. Install dependencies

   pip install -r requirements.txt

3. Open the notebook

   jupyter notebook LatentWatermark_TSR.ipynb

