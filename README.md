# Latent Watermark for Traffic Sign Recognition (TSR)

## Project Overview
This project implements a **latent-space watermarking framework** for traffic sign recognition (TSR) systems. Unlike traditional watermarking, our method embeds the **latent representation** of a traffic sign directly into the image, ensuring the watermark is **imperceptible**, **robust to digital and physical distortions**, and **preserves classification performance**.

---

## Key features:
- **Latent-space embedding:** Protects images without altering pixel-level information.  
- **Physical-world robustness:** Handles noise, blur, color jitter, and perspective changes.  
- **End-to-end recovery:** Recovers the latent vector and reconstructs the image, ensuring watermark reliability.  
- **Plug-and-play:** Compatible with any TSR model without retraining.

---

## Architecture
![Architecture](SignGuard.drawio.png) 

*Figure:Latent watermarking pipeline for traffic sign images.* The autoencoder extracts the latent representations of the traffic sign images. The U-Net watermarking module embeds these latent representations into the traffic sign images imperceptibly. A noise layer simulation ensures robustness against both digital and physical level perturbations. Finally, the latent extractor recovers the embedded latent representations from the distorted traffic sign images, which are reconstructed by the autoencoder back to the traffic sign image, which is used by the TSR system for accurate recognition.

---

## Dataset
We use the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.  
It is publicly available and contains over 50,000 labeled images across 43 sign classes.(https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

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
   ---

data/
├── meta/
├── train/ 
├── test/
├── Meta.csv
├── Train.csv
└── Test.csv