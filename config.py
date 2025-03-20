import os
import torch

# 1. Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Paths to style and content folders
STYLE_DIR = "Chinese_art_dataset/style_images"
CONTENT_DIR = "Chinese_art_dataset/content_images"

# 3. Image parameters
IMG_SIZE = 256       # 128 or 256 based on GPU memory
BATCH_SIZE = 1       # Typically 1 or 2 for CycleGAN

# 4. Training parameters
EPOCHS = 20
LR = 2e-4       # Learning rate
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_CYCLE = 10.0  # Weight for cycle consistency loss
LAMBDA_ID = 5.0      # Weight for identity loss

# 5. Folder to save all weights
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
