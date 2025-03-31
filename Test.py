import os
import glob
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import GeneratorResNet


WEIGHTS_DIR = "weights"
CHECKPOINT_EPOCH = 20  # whichever epoch you want to load

# Path to the generator checkpoint
GENERATOR_PATH = os.path.join(WEIGHTS_DIR, f"G_content2style_{CHECKPOINT_EPOCH}.pth")

# Define paths
SOURCE_FOLDER = "test_folder"       # Folder with images to stylize
OUTPUT_FOLDER = "stylized_results"  # Folder to save stylized images
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load your generator
G_content2style = GeneratorResNet()
G_content2style.load_state_dict(torch.load(GENERATOR_PATH, map_location="cpu"))
G_content2style.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Inference function
def stylize_image(input_image_path, output_image_path):
    img = Image.open(input_image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        fake_style = G_content2style(input_tensor)
        # Denormalize output from [-1,1] back to [0,1]
        fake_style = 0.5 * (fake_style + 1.0)
        # Convert to PIL
        fake_style_img = transforms.ToPILImage()(fake_style.squeeze())

    # Save result
    fake_style_img.save(output_image_path)

# 5. Process all images in SOURCE_FOLDER
image_paths = glob.glob(os.path.join(SOURCE_FOLDER, "*.jpg"))
for img_path in image_paths:
    filename = os.path.basename(img_path)  # e.g. "example.jpg"
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    stylize_image(img_path, out_path)
    print(f"Saved stylized image to {out_path}")
